"""Timing: placing scene clips on the global timeline and resolving events.

Section 6 of docs/animation_engine_design.md. A :class:`Timeline` holds
clips (prepared scenes placed at a global start time) and event times.
Each clip's *time map* says which compiled scene time is shown at which
global time. It is made of one :class:`MonotoneMap` per segment, with holds
in between:

* the first segment starts at the clip's start;
* a later segment starts at the time of the event it waits for, or when
  the previous segment ends if that is later (a click that arrives early
  is queued);
* inside a segment, time runs at the segment's scale, slowed down or sped
  up inside slow-motion ranges, and is fitted smoothly (PCHIP) through
  every *cue* whose event is known, so each cue happens exactly at its
  event (for example a spoken word);
* a *cut* trims the clip to a range of its playing time; segments outside
  it are skipped and their emits do not fire.

Events come from :meth:`Timeline.fire` (a fixed time, an editor, a
presenter's click) or from scenes' ``emit`` calls, whose global time
depends on their own segment's time map. Everything is resolved together
by iterating to a fixed point.

Editing (adding clips, firing events, changing a clip) resolves
leniently, so clips can be added in any order. :meth:`Timeline.resolve` is
the strict check an offline render runs: it rejects events that never
fire (including cycles) and cues that would make time run backwards. In
live mode (a running presentation) segments waiting for unfired events
hold until they fire.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from ..objects import Group as RenderGroup
from ..objects import Quaternion, Transform, new_uid
from ..renderer.config import Camera as RenderCamera
from ..renderer.config import ViewState
from .audio import DEFAULT_SAMPLE_RATE, Sound, mix
from .cameras import Camera, View
from .scene import PreparedScene, Segment
from .time_map import MonotoneMap
from .worlds import World, World3D

NEVER = float("inf")
"""Start time of a segment whose event has not fired (yet)."""

_MAX_PASSES = 10_000


class UnresolvedEvents(ValueError):
    """Events that segments wait for but nothing fires (or that form a cycle)."""


class InvalidCue(ValueError):
    """A cue whose event would make the scene's time run backwards."""


@dataclass(frozen=True, slots=True)
class Ramp:
    """Play the scene ``factor`` times slower between two playing times
    (factor 0.5 plays it twice as fast)."""

    start: float
    end: float
    factor: float

    def __post_init__(self) -> None:
        if self.end <= self.start or self.factor <= 0.0:
            raise ValueError("a ramp needs start < end and a positive factor")


@dataclass(eq=False)
class SegmentTiming:
    """Where one segment of a clip landed on the global timeline."""

    index: int
    start: float
    end: float
    compiled_start: float
    compiled_end: float
    map: MonotoneMap | None
    """Global -> compiled time; None for segments of zero length."""

    def compiled_at(self, t: float) -> float:
        if self.map is None:
            return self.compiled_start
        return self.map(t)

    def global_at(self, compiled: float) -> float:
        if self.map is None:
            return self.start
        return self.map.inverse(compiled)

    def key(self) -> tuple:
        knots = () if self.map is None else (tuple(self.map.x), tuple(self.map.y))
        return (self.start, self.end, knots)


@dataclass(eq=False)
class Clip:
    """A prepared scene placed in a world and on the global timeline.

    Where: the clip's ``world``, its ``transform`` in that world, and its
    ``layer`` (higher layers are drawn on top; equal layers in the order
    the clips were added). When: its ``start``, its cut and its time map.

    Changing a clip's timing (``cut``, ``slow_motion``, ``bind_cue``,
    ``set_scale``) marks its timeline for re-resolution. Editing methods
    return the clip, so they chain.
    """

    scene: PreparedScene
    start: float = 0.0
    name: str = ""
    world: World | None = None
    transform: Transform = field(default_factory=Transform)
    layer: int = 0
    uid: int = field(default_factory=new_uid, init=False, repr=False)
    """Render identity of the clip's group, the same in every snapshot."""
    scales: dict[int, float] = field(default_factory=dict)
    """Time scale per segment index (default 1: real time; 2: half speed)."""
    ramps: list[Ramp] = field(default_factory=list)
    cue_events: dict[str, str] = field(default_factory=dict)
    """Event bound to each cue (default: the event with the cue's name)."""
    cut_in: float | None = None
    cut_out: float | None = None
    """Cut range in playing time (None: from the start / to the end)."""
    timings: list[SegmentTiming | None] = field(default_factory=list, init=False)
    """Per segment: its timing, or None if skipped by the cut or not resolved."""
    timeline: Timeline | None = field(default=None, init=False, repr=False)

    # -- editing ------------------------------------------------------------

    def cut(self, start: float | None = None, end: float | None = None) -> Clip:
        """Trim the clip to playing times ``[start, end]``."""
        if start is not None and end is not None and end <= start:
            raise ValueError("a cut needs start < end")
        self.cut_in, self.cut_out = start, end
        return self._changed()

    def slow_motion(self, start: float, end: float, factor: float) -> Clip:
        """Play playing times ``[start, end]`` ``factor`` times slower."""
        self.ramps.append(Ramp(start, end, factor))
        return self._changed()

    def bind_cue(self, cue: str, event: str) -> Clip:
        """Sync ``cue`` to ``event`` (e.g. a transcript word)."""
        self.cue_events[cue] = event
        return self._changed()

    def set_scale(self, segment: int, scale: float) -> Clip:
        if scale <= 0.0:
            raise ValueError("a segment's scale must be positive")
        self.scales[segment] = scale
        return self._changed()

    def place(
        self,
        position: object = None,
        rotation: Quaternion | None = None,
        scale: object = None,
        layer: int | None = None,
    ) -> Clip:
        """Place the clip in its world; arguments left out stay unchanged.

        The transform applies to the whole scene, so one scene can appear
        in several places through several clips (each with its own
        prepared scene).
        """
        transform = Transform(
            position=self.transform.position if position is None else _vector3(position),
            rotation=self.transform.rotation if rotation is None else rotation,
            scale=self.transform.scale if scale is None else _vector3(scale),
        )
        if self.world is not None:
            self.world.check_transform(transform)
        self.transform = transform
        if layer is not None:
            self.layer = layer
        return self

    def _changed(self) -> Clip:
        if self.timeline is not None:
            self.timeline._dirty = True
        return self

    # -- queries ------------------------------------------------------------

    def still(self, at: float, global_time: float | None = None) -> RenderGroup:
        """The clip's render state at playing time ``at``, placed by its
        transform: a still frame, independent of the clip's start, cut and
        time map. ``global_time`` (for idle motion) defaults to ``at``."""
        compiled = self.scene.compiled_time_at(at)
        state = self.scene.state_at(compiled, global_time=at if global_time is None else global_time)
        return self._group(state.children)

    def _group(self, children: list) -> RenderGroup:
        return RenderGroup(self.name, transform=self.transform, children=children, uid=self.uid)

    def event_of(self, cue: str) -> str:
        return self.cue_events.get(cue, cue)

    @property
    def segment_starts(self) -> list[float]:
        """Global start per segment (NEVER if skipped or not yet placed)."""
        if self.timeline is not None:
            self.timeline._ensure_resolved()
        return [NEVER if timing is None else timing.start for timing in self.timings]

    @property
    def end(self) -> float:
        """Global end time (NEVER while a segment waits for an unfired event)."""
        if self.timeline is not None:
            self.timeline._ensure_resolved()
        included = [timing for timing in self.timings if timing is not None]
        if not included or len(included) < self._included_count():
            return NEVER
        return included[-1].end

    def compiled_time(self, t: float) -> float | None:
        """The clip's compiled time at global time ``t``, or None outside it.

        Inside a segment, the segment's time map applies; between segments
        the clip holds at the end of the previous one.
        """
        included = [timing for timing in self.timings if timing is not None]
        if not included or t < included[0].start or t > self.end:
            return None
        index = max(0, bisect.bisect_right([timing.start for timing in included], t) - 1)
        timing = included[index]
        if t >= timing.end:
            return timing.compiled_end  # holding until the next segment
        return timing.compiled_at(t)

    def held_time(self, t: float) -> float:
        """Like :meth:`compiled_time`, but holding the first state before
        the clip starts and the last one after it ends (a camera keeps its
        placement when its camera scene is not playing)."""
        compiled = self.compiled_time(t)
        if compiled is not None:
            return compiled
        included = [timing for timing in self.timings if timing is not None]
        if not included:
            return 0.0
        return included[0].compiled_start if t < included[0].start else included[-1].compiled_end

    def placed_sounds(self) -> list[tuple[float, float, Sound, float]]:
        """``(global time, skip, sound, gain)`` of the clip's sounds.

        A sound starts at the global time its scene point is shown and
        then plays at its own speed (time maps never change pitch). Sounds
        in skipped segments or after the cut's end are left out; a sound
        that started before the cut's start plays on from where the cut
        enters it.
        """
        if self.timeline is not None:
            self.timeline._ensure_resolved()
        placed = []
        for timing, segment in zip(self.timings, self.scene.segments, strict=True):
            compiled = self._compiled_range(segment)
            if timing is None or compiled is None:
                continue
            for event in segment.sounds:
                at = segment.offset + event.time
                if at > compiled[1]:
                    continue
                if at < compiled[0]:
                    placed.append((timing.start, compiled[0] - at, event.sound, event.gain))
                else:
                    placed.append((timing.global_at(at), 0.0, event.sound, event.gain))
        return placed

    def _compiled_range(self, segment: Segment) -> tuple[float, float] | None:
        """The part of ``segment`` inside the cut, in compiled time."""
        first = 0.0 if self.cut_in is None else self.cut_in - segment.play_offset
        last = segment.duration if self.cut_out is None else self.cut_out - segment.play_offset
        first, last = max(first, 0.0), min(last, segment.duration)
        if last < first or (last == first and segment.duration > 0.0):
            return None
        return segment.offset + first, segment.offset + last

    def _included_count(self) -> int:
        return sum(1 for segment in self.scene.segments if self._compiled_range(segment) is not None)


@dataclass(eq=False)
class Timeline:
    """Clips on one global timeline, with their worlds and events."""

    clips: list[Clip] = field(default_factory=list)
    events: dict[str, float] = field(default_factory=dict)
    """Events with a time set directly (fixed, or fired by a presenter)."""
    live: bool = False
    """Live mode: unfired events hold instead of being an error."""
    worlds: list[World] = field(default_factory=lambda: [World3D("world")])
    """The worlds clips play in; the first one is the default."""
    _dirty: bool = field(default=True, repr=False)

    def add(
        self,
        scene: PreparedScene,
        start: float = 0.0,
        name: str = "",
        scales: dict[int, float] | None = None,
        world: World | None = None,
        transform: Transform | None = None,
        layer: int = 0,
    ) -> Clip:
        """Place ``scene`` in ``world`` (default: the first world) at global
        time ``start``.

        A prepared scene evaluates by setting its signals, so it can only
        back one clip; to show a scene twice, prepare it twice.
        """
        if any(clip.scene is scene for clip in self.clips):
            raise ValueError("this prepared scene already backs a clip; prepare the scene again to reuse it")
        world = world or self.worlds[0]
        if world not in self.worlds:
            self.worlds.append(world)
        if transform is not None:
            world.check_transform(transform)
        clip = Clip(scene, start, name or f"clip-{len(self.clips)}", world=world, layer=layer, scales=dict(scales or {}))
        if transform is not None:
            clip.transform = transform
        clip.timeline = self
        self.clips.append(clip)
        self._dirty = True
        return clip

    def world(self, name: str) -> World:
        for world in self.worlds:
            if world.name == name:
                return world
        raise KeyError(f"no world named {name!r}")

    def fire(self, event: str, time: float) -> None:
        """Set the time of ``event`` (a presenter's click, or a fixed time)."""
        self.events[event] = time
        self._dirty = True

    def resolve(self) -> dict[str, float]:
        """Resolve all timing; returns the event times.

        Outside live mode, raises :class:`UnresolvedEvents` if a segment
        waits for an event that never fires, and :class:`InvalidCue` if a
        cue's event would make time run backwards.
        """
        return self._resolve(strict=not self.live)

    # -- queries ------------------------------------------------------------

    @property
    def duration(self) -> float:
        self._ensure_resolved()
        ends = [clip.end for clip in self.clips if clip.end != NEVER]
        return max(ends, default=0.0)

    def state_at(self, t: float, world: World | None = None) -> RenderGroup:
        """Render state of ``world`` (default: the first world) at global
        time ``t``: one group per active clip, placed by the clip's
        transform, in layer order (higher layers on top; equal layers in
        the order the clips were added)."""
        self._ensure_resolved()
        world = world or self.worlds[0]
        children = []
        for clip in sorted(self.clips, key=lambda clip: clip.layer):  # stable
            if clip.world is not world:
                continue
            compiled = clip.compiled_time(t)
            if compiled is not None:
                state = clip.scene.state_at(compiled, global_time=t)
                children.append(clip._group(state.children))
        return RenderGroup(world.name, children=children, uid=world.uid)

    def audio(self, start: float = 0.0, end: float | None = None, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """Every clip's sounds mixed on the global timeline, stereo float32
        for ``[start, end)`` (default: the whole timeline)."""
        placed = [entry for clip in self.clips for entry in clip.placed_sounds()]
        return mix(placed, start, self.duration if end is None else end, sample_rate)

    def view_at(self, view: View, t: float) -> ViewState:
        """The view's world and camera at global time ``t``."""
        root = self.state_at(t, view.world)
        clip = self._clip_of(view.camera)
        if clip is not None and clip.world is not view.world:
            raise ValueError(f"camera {view.camera.name!r} plays in world {clip.world.name!r}, not {view.world.name!r}")
        return ViewState(root, self.camera_at(view.camera, t))

    def view(self, view: View) -> Callable[[float], ViewState]:
        """``t -> view_at(view, t)``, a scene function for the FrameComposer."""
        return lambda t: self.view_at(view, t)

    def camera_at(self, camera: Camera, t: float) -> RenderCamera:
        """The renderer camera of ``camera`` at global time ``t``.

        The camera's clip (the clip whose scene contains it) is evaluated
        at its held time, so a camera keeps its placement before and after
        its camera scene plays; the clip's transform places it in the
        world. A camera in no clip keeps its initial values.
        """
        self._ensure_resolved()
        clip = self._clip_of(camera)
        if clip is None:
            return camera.render_camera()
        clip.scene.evaluate(clip.held_time(t), global_time=t)
        return camera.render_camera(clip.transform)

    def _clip_of(self, camera: Camera) -> Clip | None:
        for clip in self.clips:
            if any(camera in root.family() for root in clip.scene.roots):
                return clip
        return None

    def segments(self) -> Iterable[tuple[Clip, int, float, float]]:
        """``(clip, segment index, global start, global end)`` of every
        placed segment, for display."""
        self._ensure_resolved()
        for clip in self.clips:
            for timing in clip.timings:
                if timing is not None:
                    yield clip, timing.index, timing.start, timing.end

    # -- resolution ---------------------------------------------------------

    def _ensure_resolved(self) -> None:
        if self._dirty:
            self._resolve(strict=False)

    def _resolve(self, strict: bool) -> dict[str, float]:
        """Iterate segment timings and event times to a fixed point.

        Each pass places the segments whose events are known and fits their
        maps through the cues whose events are known; emits of placed
        segments may reveal events for the next pass. Values only settle
        from "unknown" towards their final values, so this terminates.
        """
        for clip in self.clips:
            clip.timings = [None] * len(clip.scene.segments)
        known = dict(self.events)
        for _ in range(_MAX_PASSES):
            changed = False
            for clip in self.clips:
                changed |= self._place(clip, known, strict)
            if not changed:
                break
        if strict:
            missing = sorted(
                {
                    segment.waits_for
                    for clip in self.clips
                    for index, segment in enumerate(clip.scene.segments)
                    if segment.waits_for
                    and clip.timings[index] is None
                    and clip._compiled_range(segment) is not None
                    and segment.waits_for not in known
                }
            )
            if missing:
                raise UnresolvedEvents(f"events never fire (missing or waiting on each other): {missing}")
        self._dirty = False
        return known

    def _place(self, clip: Clip, known: dict[str, float], strict: bool) -> bool:
        """Place ``clip``'s segments with the events known so far."""
        changed = False
        previous_end: float | None = None
        for index, segment in enumerate(clip.scene.segments):
            compiled = clip._compiled_range(segment)
            if compiled is None:
                continue  # outside the cut
            if previous_end is None:
                start = clip.start  # the first segment inside the cut
            elif segment.waits_for is None or segment.waits_for not in known:
                break  # waits for an event that has not fired (yet)
            else:
                start = max(known[segment.waits_for], previous_end)
            timing = _segment_timing(clip, index, segment, start, compiled, known, strict)
            old = clip.timings[index]
            if old is None or old.key() != timing.key():
                clip.timings[index] = timing
                changed = True
            for emit in segment.emits:
                at = segment.offset + emit.time
                if not compiled[0] <= at <= compiled[1]:
                    continue  # cut away
                time = timing.global_at(at)
                if emit.event not in self.events and known.get(emit.event, NEVER) > time:
                    known[emit.event] = time  # the earliest emit wins
                    changed = True
            previous_end = timing.end
        return changed


def _vector3(value: object) -> tuple[float, float, float]:
    """Three floats from a vector (a scalar is used for all three)."""
    x, y, z = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,))
    return float(x), float(y), float(z)


def _segment_timing(
    clip: Clip,
    index: int,
    segment: Segment,
    start: float,
    compiled: tuple[float, float],
    known: dict[str, float],
    strict: bool,
) -> SegmentTiming:
    """Fit one segment's time map from its start, ramps and cues."""
    first, last = compiled
    if last == first:
        return SegmentTiming(index, start, start, first, last, None)
    scale = clip.scales.get(index, 1.0)

    def to_compiled(playing: float) -> float:
        return segment.offset + playing - segment.play_offset

    # Speed changes at ramp edges (compiled time -> slowdown factor).
    edges: dict[float, None] = {}
    ramps = [(to_compiled(r.start), to_compiled(r.end), r.factor) for r in clip.ramps]
    for ramp_start, ramp_end, _ in ramps:
        for edge in (ramp_start, ramp_end):
            if first < edge < last:
                edges[edge] = None

    def factor_at(position: float) -> float:
        """Slowdown of the stretch starting at ``position``."""
        result = 1.0
        for ramp_start, ramp_end, factor in ramps:
            if ramp_start <= position < ramp_end:
                result *= factor
        return result

    cues = {
        segment.offset + cue.time: known[clip.event_of(cue.name)]
        for cue in segment.cues
        if clip.event_of(cue.name) in known and first < segment.offset + cue.time < last
    }

    knots_x, knots_y = [start], [first]
    for position in sorted({*edges, *cues}):
        if position in cues:
            time = cues[position]
            if time <= knots_x[-1]:
                if strict:
                    raise InvalidCue(f"cue at scene time {position} of {clip.name} would make time run backwards")
                continue
        else:
            time = knots_x[-1] + (position - knots_y[-1]) * scale * factor_at(knots_y[-1])
        knots_x.append(time)
        knots_y.append(position)
    end = knots_x[-1] + (last - knots_y[-1]) * scale * factor_at(knots_y[-1])
    knots_x.append(end)
    knots_y.append(last)
    return SegmentTiming(index, start, end, first, last, MonotoneMap(knots_x, knots_y))
