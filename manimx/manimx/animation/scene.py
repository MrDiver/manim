"""Scenes: authoring code compiled once, evaluated at any time.

    class Intro(Scene):
        def construct(self):
            square = Square(color=BLUE, fill_opacity=1)
            self.play(FadeIn(square))
            self.play(square.animate.shift(RIGHT * 2), run_time=2)
            self.wait()

    prepared = Intro().prepare()      # pass 1: runs construct() once
    frame = prepared.state_at(1.5)    # pass 2: any time, any order (a render object tree)

``construct()`` never renders and never sleeps. Each ``play`` places
animations on the scene's local timeline at the current cursor and moves
the cursor; ``wait`` only moves the cursor. ``start`` begins animations
at the cursor without moving it, and ``stop`` ends them wherever the
cursor is then, so they run while the scene goes on:

    hops = self.start(MoveAlongPath(dot, path))   # from here ...
    self.play(Write(title))                       # ... meanwhile ...
    self.cue("voice.circle")
    self.stop(hops)                               # ... to here

With cues at both ends, a started animation lasts exactly from one
spoken word to another, however the voice is paced. A started
animation that is not stopped ends with its segment (at the next
``wait_for``, or the end of the scene). ``start(..., run_time=0.5)``
runs a short effect alongside: it ends by itself after its run time. The result is a
:class:`PreparedScene`: tracks of placed animations per signal, plus the
nodes in the order they were added (later ones are drawn on top, as in
Manim).

Segments: ``wait_for("EVENT")`` ends the current segment; the scene holds
there until the event fires on the global timeline (timing.py), then the
next segment begins. ``emit("EVENT")`` fires an event at the current point
of the scene, for other segments or scenes to wait for.

Segments are laid out on the scene's compiled time axis one after another,
separated by a gap of :data:`SEGMENT_GAP` in which nothing is ever placed:

    compiled:  [seg 0: 0 - 4] gap [seg 1: 5 - 8] gap [seg 2: 9 - ...]

Tracks and animations need no notion of segments at all; a segment only
decides which compiled time to evaluate. A hold evaluates the exact end of
the previous segment, which lies before the next segment's start, so
nothing that happens at the very start of the next segment (an ``add``,
an instantaneous change) leaks into the hold. Retiming a segment on the
global timeline only changes the mapping to compiled time.
"""

from __future__ import annotations

import bisect
import gc
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..renderer.config import ViewState
from ..robjects import RGroup, new_uid
from .animations import (
    Animation,
    AnimationBuilder,
    AnimationGroup,
    SetValue,
    SignalChange,
    as_animation,
)
from .audio import DEFAULT_SAMPLE_RATE, Sound, SoundEvent, mix
from .authoring import _active
from .nodes.camera import Camera
from .nodes.group import Group
from .nodes.node import Node
from .signals.signal import RootSignal, Signal, TimeSignals
from .signals.signal_tracks import Change, Placement, RateFunction, SignalTrack, linear, refresh
from .simulation import Simulation

if TYPE_CHECKING:
    from .plan import ScenePlan

SEGMENT_GAP = 1.0
"""Compiled time between two segments; nothing is ever placed inside it."""


@dataclass(frozen=True, slots=True)
class Emit:
    event: str
    time: float
    """Time within the segment."""


@dataclass(frozen=True, slots=True)
class Cue:
    """A sync point: the clip's time map passes through it when the bound
    event fires (see timing.py), without holding."""

    name: str
    time: float
    """Time within the segment."""


@dataclass(frozen=True, slots=True)
class Segment:
    """The stretch of a scene between two ``wait_for`` calls."""

    index: int
    offset: float
    """Where the segment starts on the scene's compiled time axis."""
    duration: float
    waits_for: str | None
    """The event that starts it (None for the first segment)."""
    emits: tuple[Emit, ...] = ()
    play_offset: float = 0.0
    """Playing time (compiled time without gaps) at the segment's start."""
    cues: tuple[Cue, ...] = ()
    sounds: tuple[SoundEvent, ...] = ()

    @property
    def end(self) -> float:
        return self.offset + self.duration


@contextmanager
def _no_cycle_collection() -> Iterator[None]:
    """Keep Python's cycle collector off a scene while and after it is prepared.

    construct() makes millions of objects for a large scene (100 000 dice:
    half a million nodes and their signals), and the collector walks all
    objects alive each time enough new ones pile up: 60% of preparing went
    there, and as much of building the scene's plan and rendering after it.
    A prepared scene is long-lived data, like a cache, that the collector
    need not look through again and again. So it is paused while preparing
    (nothing made then is garbage before it ends), and what is alive
    afterwards is frozen (``gc.freeze``): moved to a generation the
    collector skips, at no cost.

    Each prepare unfreezes first: scenes a reload replaced (``manimx ui``)
    become collectable again, and what is still alive is frozen anew.
    https://docs.python.org/3/library/gc.html#gc.freeze
    """
    enabled = gc.isenabled()
    gc.disable()
    gc.unfreeze()
    try:
        yield
    finally:
        gc.freeze()
        if enabled:
            gc.enable()


@dataclass(eq=False)
class Running:
    """Animations begun with :meth:`Scene.start`, until :meth:`Scene.stop`."""

    placements: list[tuple[Placement, float, float]]
    """Each animation's placement, and its start and length in the group
    (fractions of the run time, which ``stop`` decides)."""
    begin: float
    """The cursor when they started."""
    segment: int
    """The segment they started in (they end with it at the latest)."""
    animations: list[Animation] = field(default_factory=list)
    stopped: bool = False


class Scene:
    """Subclass and override :meth:`construct`, then call :meth:`prepare`."""

    def __init__(self) -> None:
        self.time = TimeSignals()
        self._camera = Camera("camera")
        self._compiler: SceneCompiler | None = None
        self._prepared: PreparedScene | None = None

    @property
    def camera(self) -> Camera:
        """The scene's one camera: move it like any node, e.g.
        ``self.play(self.camera.animate.shift(RIGHT))``. It is never drawn
        and needs no ``add``.
        """
        return self._camera

    def construct(self) -> None:
        """Describe the scene with ``add``, ``remove``, ``play`` and ``wait``."""

    def prepare(self) -> PreparedScene:
        """Run :meth:`construct` once and return the evaluable scene."""
        if self._prepared is None:
            with _no_cycle_collection():
                self._prepared = self._prepare()
        return self._prepared

    def _prepare(self) -> PreparedScene:
        compiler = SceneCompiler(self.time)
        # Camera changes are recorded at their time, like changes of
        # nodes in the scene, without the camera being added.
        compiler.timeline_signals.update(signal.id for signal in self._camera.signals())
        self._compiler = compiler
        token = _active.set(compiler)
        try:
            self.construct()
        finally:
            _active.reset(token)
            self._compiler = None
        prepared = compiler.finish(self.time, self._camera)
        nodes = {node.uid: node for root in prepared.roots for node in root.family()}
        if any([node.after_prepare(prepared) for node in nodes.values()]):
            prepared.refresh()
        return prepared

    # -- authoring API (only inside construct) --------------------------------

    def add(self, *nodes: Node) -> None:
        """Put nodes into the scene at the current time."""
        self._authoring().add(nodes)

    def remove(self, *nodes: Node) -> None:
        """Take nodes out of the scene at the current time."""
        self._authoring().remove(nodes)

    def play(
        self,
        *animations: Animation | AnimationBuilder | AnimationGroup,
        run_time: float | None = None,
        rate: RateFunction | None = None,
        lag_ratio: float = 0.0,
    ) -> None:
        """Place animations at the current time and wait until they end.

        ``run_time`` and ``rate`` override the animations' own (a group's
        run time is fitted to ``run_time``; ``rate`` reaches the animations
        in it). With ``lag_ratio`` > 0 each animation starts that fraction
        of its run time after the previous one (Manim's LaggedStart).
        """
        self._authoring().play([a if isinstance(a, AnimationGroup) else as_animation(a) for a in animations], run_time, rate, lag_ratio)

    def start(
        self,
        *animations: Animation | AnimationBuilder | AnimationGroup,
        run_time: float | None = None,
        rate: RateFunction | None = None,
    ) -> Running:
        """Begin animations at the current time without waiting for them:
        they run until :meth:`stop` (from start to stop is their run time)
        while the scene goes on; with ``run_time``, they end by themselves
        after it, like a ``play`` that does not wait. ``rate`` overrides
        their own rates."""
        return self._authoring().start([a if isinstance(a, AnimationGroup) else as_animation(a) for a in animations], run_time, rate)

    def stop(self, *running: Running) -> None:
        """End animations begun with :meth:`start` at the current time
        (none given: every one still running)."""
        self._authoring().stop(running)

    def wait(self, duration: float = 1.0) -> None:
        """Let ``duration`` seconds pass."""
        self._authoring().wait(duration)

    def wait_until(self, t: float) -> None:
        """Let time pass until ``scene_time`` is ``t``: the next animation
        starts exactly then, however long the ones before it took (e.g. on
        a beat of the music). Raises if the scene is already past ``t``."""
        self._authoring().wait_until(t)

    def wait_for(self, event: str) -> None:
        """End the current segment: hold until ``event`` fires, then go on."""
        self._authoring().wait_for(event)

    def emit(self, event: str) -> None:
        """Fire ``event`` at this point of the scene."""
        self._authoring().emit(event)

    def cue(self, name: str) -> None:
        """Mark a sync point: when the event bound to ``name`` fires (by
        default the event of the same name, e.g. a spoken word), the scene
        is at this point. Unlike ``wait_for`` it never holds; the clip's
        time map speeds up or slows down smoothly to get there."""
        self._authoring().cue(name)

    def play_sound(self, sound: str | Path | Sound, volume: float = 1.0) -> Sound:
        """Start a sound (a file or a :class:`Sound`) at this point; the
        cursor does not move (``self.wait(sound.duration)`` waits for it)."""
        sound = sound if isinstance(sound, Sound) else Sound.load(sound)
        self._authoring().play_sound(sound, volume)
        return sound

    def play_wave(
        self, samples: np.ndarray | str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE, volume: float = 1.0
    ) -> Sound:
        """Start a wave at this point, like :meth:`play_sound`: an array (mono
        ``(count,)`` or stereo ``(count, 2)``, values in [-1, 1], at
        ``sample_rate``) or a sound file (its own sample rate applies).
        ``volume`` scales the samples (1: as recorded)."""
        sound = Sound.load(samples) if isinstance(samples, (str, Path)) else Sound(samples, sample_rate)
        return self.play_sound(sound, volume)

    def _authoring(self) -> SceneCompiler:
        if self._compiler is None:
            raise RuntimeError("add, remove, play, wait, wait_until, wait_for and emit can only be used inside construct()")
        return self._compiler


class SceneCompiler:
    """Builds the timeline while ``construct()`` runs (see authoring.py)."""

    def __init__(self, time: TimeSignals | None = None) -> None:
        self.time = time or TimeSignals()
        self.cursor = 0.0
        self.order = 0
        self.tracks: dict[int, SignalTrack] = {}
        self.roots: list[Node] = []
        self._root_uids: set[int] = set()  # roots by uid: a list test per add made adding n nodes O(n^2)
        self._held_by: dict[int, dict[int, Node]] = {}  # member uid -> the roots holding it, by uid
        self._watched: set[int] = set()  # groups whose new children are indexed in _held_by
        self.present: set[int] = set()
        self.timeline_signals: set[int] = set()
        self.placements: list[Placement] = []
        self.segments: list[Segment] = []
        self.segment_start = 0.0
        self.segment_event: str | None = None
        self.segment_emits: list[Emit] = []
        self.segment_cues: list[Cue] = []
        self.segment_sounds: list[SoundEvent] = []
        self.running: list[Running] = []
        """Animations begun with ``start`` and not stopped yet."""
        self.started = False
        """Whether any animation was started: their placements' start
        values are computed again at the end, with their real lengths."""

    # -- used by authoring.py -------------------------------------------------

    def on_timeline(self, signal: Signal) -> bool:
        """Whether changes of ``signal`` must be recorded at the cursor."""
        return signal.id in self.timeline_signals or signal.id in self.tracks

    def record_instant(self, signal: RootSignal, mode: Change, value: object) -> None:
        self.place(SetValue(SignalChange(signal, mode, value)), self.cursor, 0.0)

    def current_value(self, signal: Signal) -> object:
        track = self.tracks.get(signal.id)
        return track.final_value() if track is not None else signal.value

    # -- authoring operations -------------------------------------------------

    def add(self, nodes: Sequence[Node]) -> None:
        for node in nodes:
            if node.uid in self.present:
                continue
            self._hold(node, node)
            for member in node.family():
                self.timeline_signals.update(signal.id for signal in member.signals())
                # A member already in the scene on its own is drawn through
                # this node from now on; until now it stays on its own.
                if member is not node and member.uid in self.present:
                    self.remove([member])
            if node.uid not in self._root_uids:
                self._root_uids.add(node.uid)
                self.roots.append(node)
            self.present.add(node.uid)
            self.record_instant(node.present, Change.ABSOLUTE, True)

    def _hold(self, root: Node, part: Node) -> None:
        """Index ``part``'s family as held by ``root``; groups among them
        keep the index up to date when they gain children later."""
        for member in part.family():
            if member is not root:
                self._held_by.setdefault(member.uid, {})[root.uid] = root
            if isinstance(member, Group) and member.uid not in self._watched:
                self._watched.add(member.uid)
                member.on_structure_change(self._grown)

    def _grown(self, group: Group) -> None:
        """``group`` gained children: the roots holding it hold them too."""
        roots = list(self._held_by.get(group.uid, {}).values())
        if group.uid in self._root_uids:
            roots.append(group)
        for root in roots:
            self._hold(root, group)

    def remove(self, nodes: Sequence[Node]) -> None:
        for node in nodes:
            if node.uid in self.present:
                self.present.discard(node.uid)
                self.record_instant(node.present, Change.ABSOLUTE, False)

    def play(self, animations: Sequence[Animation | AnimationGroup], run_time: float | None, rate: RateFunction | None, lag_ratio: float) -> None:
        if not animations:
            raise ValueError("play() needs at least one animation")
        start = self.cursor
        end = start
        # Simulations are baked last, so they see the other animations of
        # this call as inputs (e.g. an animated paddle pushing a ball).
        order = sorted(range(len(animations)), key=lambda i: isinstance(animations[i], Simulation))
        for index in order:
            animation = animations[index]
            simulated = isinstance(animation, Simulation)
            duration = animation.run_time if run_time is None or simulated else run_time
            begin = start + index * lag_ratio * duration
            if isinstance(animation, AnimationGroup):
                for member, offset, length in animation.schedule(duration):
                    if isinstance(member, Simulation):
                        raise TypeError("a Simulation cannot be in an AnimationGroup; play it on its own")
                    self._place_one(member, begin + offset, length, rate or member.rate)
            elif simulated:
                self.cursor = begin
                self.add([m for m in animation.nodes() if not self._is_inside_present(m)])
                duration = animation.bake(self, begin)
                for event, at in animation.events:
                    self.segment_emits.append(Emit(event, begin + at - self.segment_start))
                for sound, volume, at in animation.sounds:
                    self.play_sound(sound, volume, at=begin + at)
                self.place(animation, begin, duration, linear)
                self._remove_after(animation, begin + duration)
            else:
                self._place_one(animation, begin, duration, rate or animation.rate)
            end = max(end, begin + duration)
        self.cursor = end

    def start(self, animations: Sequence[Animation | AnimationGroup], run_time: float | None, rate: RateFunction | None) -> Running:
        """Place ``animations`` at the cursor now (their place in authoring
        order), with no end yet: :meth:`stop` gives them their length.
        With ``run_time``: a ``play`` that leaves the cursor where it is."""
        if not animations:
            raise ValueError("start() needs at least one animation")
        begin = self.cursor
        if run_time is not None:
            self.play(animations, run_time, rate, 0.0)
            self.cursor = begin
            return Running([], begin, len(self.segments), stopped=True)
        running = Running([], begin, len(self.segments))
        for animation in animations:
            members = animation.schedule(1.0) if isinstance(animation, AnimationGroup) else [(animation, 0.0, 1.0)]
            for member, offset, length in members:
                if isinstance(member, Simulation):
                    raise TypeError("a Simulation cannot be started; play it")
                self.add([m for m in member.nodes() if not self._is_inside_present(m)])
                # No end until it stops (stop sets start and length again).
                placement = self.place(member, begin, math.inf, rate or member.rate)
                running.placements.append((placement, offset, length))
                running.animations.append(member)
        self.cursor = begin
        self.running.append(running)
        self.started = True
        return running

    def stop(self, running: Sequence[Running]) -> None:
        for item in running or list(self.running):
            if item.segment != len(self.segments):
                raise ValueError("these animations ended with their segment (at a wait_for) already")
            if item.stopped:
                raise ValueError("these animations were stopped already")
            self._end(item, self.cursor)

    def _end(self, running: Running, at: float) -> None:
        """Give ``running``'s animations their length: from their start to ``at``."""
        running.stopped = True
        self.running.remove(running)
        duration = at - running.begin
        cursor = self.cursor
        for (placement, offset, length), animation in zip(running.placements, running.animations, strict=True):
            placement.start = running.begin + offset * duration
            placement.duration = length * duration
            for sound, volume, when in animation.audio():
                self.play_sound(sound, volume, at=placement.start + when * placement.duration)
            self._remove_after(animation, placement.end)
        self.cursor = cursor

    def _place_one(self, animation: Animation, begin: float, duration: float, rate: RateFunction) -> None:
        # Nodes appear when their own animation begins: a lagged FadeIn
        # must not show its node before its fade starts.
        self.cursor = begin
        self.add([m for m in animation.nodes() if not self._is_inside_present(m)])
        self.place(animation, begin, duration, rate)
        for sound, volume, when in animation.audio():
            self.play_sound(sound, volume, at=begin + when * duration)
        self._remove_after(animation, begin + duration)

    def _remove_after(self, animation: Animation, at: float) -> None:
        if animation.removes():
            self.cursor = at
            self.remove(animation.removes())

    def wait(self, duration: float) -> None:
        if duration < 0:
            raise ValueError("cannot wait a negative time")
        self.cursor += duration

    def scene_time(self) -> float:
        """``scene_time`` at the cursor: the closed segments plus the time
        since the current one started (holds at events do not count)."""
        return sum(segment.duration for segment in self.segments) + self.cursor - self.segment_start

    def wait_until(self, t: float) -> None:
        now = self.scene_time()
        if t < now - 1e-9:
            raise ValueError(f"cannot wait until scene time {t:g} s: the scene is already at {now:g} s")
        self.cursor += max(t - now, 0.0)

    def wait_for(self, event: str) -> None:
        self._close_segment()
        self.cursor += SEGMENT_GAP
        self.segment_start = self.cursor
        self.segment_event = event

    def emit(self, event: str) -> None:
        self.segment_emits.append(Emit(event, self.cursor - self.segment_start))

    def cue(self, name: str) -> None:
        self.segment_cues.append(Cue(name, self.cursor - self.segment_start))

    def play_sound(self, sound: Sound, volume: float, at: float | None = None) -> None:
        time = self.cursor if at is None else at
        self.segment_sounds.append(SoundEvent(sound, time - self.segment_start, volume))

    def _close_segment(self) -> None:
        for running in list(self.running):
            self._end(running, self.cursor)  # started and never stopped: they end with the segment
        # A segment lasts until its cursor and until its last animation ends.
        end = max([self.cursor, *(p.end for p in self.placements if p.start >= self.segment_start)])
        play_offset = sum(segment.duration for segment in self.segments)
        self.segments.append(
            Segment(
                len(self.segments), self.segment_start, end - self.segment_start,
                self.segment_event, tuple(self.segment_emits), play_offset, tuple(self.segment_cues),
                tuple(self.segment_sounds),
            )
        )  # fmt: skip
        self.cursor = end
        self.segment_emits = []
        self.segment_cues = []
        self.segment_sounds = []

    def place(self, animation: Animation, start: float, duration: float, rate: RateFunction | None = None) -> Placement:
        placement = Placement(animation, start, duration, self.order, rate or animation.rate)
        self.order += 1
        for target in animation.targets():
            if not isinstance(target.signal, RootSignal):
                raise TypeError(f"{target.signal.name} is derived from other signals and cannot be animated")
            track = self.tracks.get(target.signal.id)
            if track is None:
                track = self.tracks[target.signal.id] = SignalTrack(target.signal)
            track.add(placement, target.mode)
        self.placements.append(placement)
        return placement

    def finish(self, time: TimeSignals, camera: Camera) -> PreparedScene:
        self._close_segment()
        if self.started:
            refresh(self.placements)  # start values seen while started animations had no end yet
        duration = self.segments[-1].end
        return PreparedScene(
            duration, list(self.tracks.values()), list(self.roots), time, camera, list(self.placements), list(self.segments)
        )

    def set_time(self, t: float) -> None:
        """Set the time signals for compiled time ``t`` of the scene being
        compiled (used while baking simulations; the global time is not
        known yet, so it equals the scene time)."""
        playing = sum(segment.duration for segment in self.segments) + max(t - self.segment_start, 0.0)
        self.time.set(scene_time=playing, global_time=playing)

    def _is_inside_present(self, node: Node) -> bool:
        """Present itself, or part of a present group."""
        if node.uid in self.present:
            return True
        # The roots holding it (_hold): looking through every root instead
        # made animating n new nodes O(n^2).
        return any(root.uid in self.present and node in root.family() for root in self._held_by.get(node.uid, {}).values())


@dataclass(eq=False)
class PreparedScene:
    """A compiled scene: evaluate it at any compiled time with :meth:`state_at`."""

    duration: float
    """End of the last segment on the compiled time axis."""
    tracks: list[SignalTrack]
    roots: list[Node]
    time: TimeSignals
    camera: Camera
    """The scene's camera; its signals are set by :meth:`evaluate` like all others."""
    placements: list[Placement] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    uid: int = field(default_factory=new_uid, init=False, repr=False)
    """Unique per prepared scene in a process: the uid of its render
    object trees' root."""
    _plan: ScenePlan | None = field(default=None, init=False, repr=False)

    @property
    def plan(self) -> ScenePlan:
        """The scene in the core's evaluator (plan.py), made on first use
        and again after :meth:`refresh`."""
        if self._plan is None:
            from .plan import ScenePlan

            self._plan = ScenePlan(self)
        return self._plan

    @property
    def playing_time(self) -> float:
        """Total duration of all segments (compiled time without gaps)."""
        return sum(segment.duration for segment in self.segments)

    def state_at(self, t: float, global_time: float | None = None) -> RGroup:
        """The render object tree at compiled time ``t`` (pure: no
        dependence on earlier calls), built by the core (:attr:`plan`).

        ``scene_time`` is the playing time at ``t`` (frozen inside the gaps
        between segments); ``global_time`` defaults to ``t``. Every
        animated signal has its value at ``t``, derived signals follow; the
        tree holds the present nodes in the order they were added. The
        Python signals are not set (see :meth:`evaluate`).
        """
        return self.plan.state_at(t, global_time)

    def frame_at(self, t: float) -> tuple[ViewState, ...]:
        """The render object tree at compiled time ``t`` seen through the
        scene's camera (the same shape as ``Timeline.frame_at``).
        """
        return (self.plan.frame_at(t),)

    def sounds(self) -> list[tuple[float, Sound, float]]:
        """``(compiled time, sound, volume)`` of every sound in the scene."""
        return [(segment.offset + event.time, event.sound, event.volume) for segment in self.segments for event in segment.sounds]

    def audio(self, start: float = 0.0, end: float | None = None, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """The scene's sounds mixed on its compiled time axis (what
        ``state_at`` shows), stereo float32 for ``[start, end)``."""
        placed = [(time, 0.0, sound, volume) for time, sound, volume in self.sounds()]
        return mix(placed, start, self.duration if end is None else end, sample_rate)

    def evaluate(self, t: float, global_time: float | None = None) -> None:
        """Set every Python signal to its value at compiled time ``t``
        (to read values, e.g. in tests and tools)."""
        self.time.set(scene_time=self.playing_time_at(t), global_time=t if global_time is None else global_time)
        for track in self.tracks:
            track.signal.set(track.value_at(t))

    def refresh(self) -> None:
        """Recompute cached start values after placements were edited (the
        plan is made again on next use)."""
        refresh(self.placements)
        self._plan = None

    def playing_time_at(self, t: float) -> float:
        """Compiled time ``t`` without the gaps before it."""
        if not self.segments:
            return t
        index = max(0, bisect.bisect_right([s.offset for s in self.segments], t) - 1)
        segment = self.segments[index]
        return segment.play_offset + min(max(t - segment.offset, 0.0), segment.duration)

    def compiled_time_at(self, playing: float) -> float:
        """The compiled time showing playing time ``playing`` (clamped).

        A playing time on a boundary maps to the end of the earlier
        segment, which is what the scene shows while holding there.
        """
        if not self.segments:
            return min(max(playing, 0.0), self.duration)
        playing = min(max(playing, 0.0), self.playing_time)
        index = max(0, bisect.bisect_left([s.play_offset for s in self.segments], playing) - 1)
        segment = self.segments[index]
        return segment.offset + min(playing - segment.play_offset, segment.duration)
