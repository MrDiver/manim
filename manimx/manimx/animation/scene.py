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
the cursor; ``wait`` only moves the cursor. The result is a
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
from collections.abc import Sequence
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
from .nodes.node import Node
from .signals.signal import RootSignal, Signal, TimeSignals
from .signals.signal_tracks import Change, Placement, RateFunction, SignalTrack, linear
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
            self._prepared = compiler.finish(self.time, self._camera)
        return self._prepared

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
        self.present: set[int] = set()
        self.timeline_signals: set[int] = set()
        self.placements: list[Placement] = []
        self.segments: list[Segment] = []
        self.segment_start = 0.0
        self.segment_event: str | None = None
        self.segment_emits: list[Emit] = []
        self.segment_cues: list[Cue] = []
        self.segment_sounds: list[SoundEvent] = []

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
            for member in node.family():
                self.timeline_signals.update(signal.id for signal in member.signals())
                # A member already in the scene on its own is drawn through
                # this node from now on; until now it stays on its own.
                if member is not node and member.uid in self.present:
                    self.remove([member])
            if node not in self.roots:
                self.roots.append(node)
            self.present.add(node.uid)
            self.record_instant(node.present, Change.ABSOLUTE, True)

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
        return node.uid in self.present or any(
            node in root.family() and root.uid in self.present for root in self.roots
        )


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
        for track in self.tracks:
            track.refresh()
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
