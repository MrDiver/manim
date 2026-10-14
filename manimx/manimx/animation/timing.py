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
presenter's click), from the words of the scripts on the audio tracks
(``voice.square``: when "square" is said, see manimx/voice/script.py)
or from scenes' ``emit`` calls, whose global time depends on their own
segment's time map. Everything is resolved together by iterating to a
fixed point.

Editing (adding clips, firing events, changing a clip) resolves
leniently, so clips can be added in any order. :meth:`Timeline.resolve` is
the strict check an offline render runs: it rejects events that never
fire (including cycles) and cues that would make time run backwards. In
live mode (a running presentation) segments waiting for unfired events
hold until they fire.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..renderer.config import ViewState
from ..robjects import RGroup, new_uid
from .audio import DEFAULT_SAMPLE_RATE, Sound, mix
from .scene import PreparedScene, Segment
from .time_map import MonotoneMap

if TYPE_CHECKING:
    from ..voice.script import Script
    from ..voice.transcript import Subtitle

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
    """A prepared scene placed on the global timeline.

    When: its ``start``, its cut and its time map. Its ``layer`` orders
    clips that play at the same time: each clip is rendered through its
    scene's camera, and higher layers are drawn over lower ones (equal
    layers in the order the clips were added).

    Changing a clip's timing (``cut``, ``slow_motion``, ``bind_cue``,
    ``set_scale``) marks its timeline for re-resolution. Editing methods
    return the clip, so they chain.
    """

    scene: PreparedScene
    start: float = 0.0
    name: str = ""
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

    def bind_cues(self, cues: dict[str, str]) -> Clip:
        """Sync each cue to its event; every other cue to the event of its own name."""
        self.cue_events = dict(cues)
        return self._changed()

    def move(self, start: float) -> Clip:
        """Start the clip at global time ``start``."""
        self.start = float(start)
        return self._changed()

    def set_layer(self, layer: int) -> Clip:
        """Draw the clip over clips of lower layers."""
        self.layer = int(layer)
        return self._changed()

    def set_speed(self, speed: float) -> Clip:
        """Play every segment ``speed`` times as fast (0.5: slow motion)."""
        if speed <= 0.0:
            raise ValueError("a clip's speed must be positive")
        self.scales = dict.fromkeys(range(len(self.scene.segments)), 1.0 / speed)
        return self._changed()

    def replace_scene(self, scene: PreparedScene) -> Clip:
        """Show ``scene`` instead (the same scene prepared again after its
        file changed); start, cut, layer and speed stay."""
        speeds = set(self.scales.values())
        self.scene = scene
        self.timings = []
        if len(speeds) == 1:  # one speed for the whole clip: the new segments get it too
            self.scales = dict.fromkeys(range(len(scene.segments)), speeds.pop())
        return self._changed()

    def set_scale(self, segment: int, scale: float) -> Clip:
        if scale <= 0.0:
            raise ValueError("a segment's scale must be positive")
        self.scales[segment] = scale
        return self._changed()

    def _changed(self) -> Clip:
        if self.timeline is not None:
            self.timeline._dirty = True
        return self

    # -- queries ------------------------------------------------------------

    def still(self, at: float, global_time: float | None = None) -> RGroup:
        """The clip's render state at playing time ``at``: a still frame,
        independent of the clip's start, cut and time map. ``global_time`` (for idle motion) defaults to ``at``.
        """
        compiled = self.scene.compiled_time_at(at)
        return self.scene.plan.state_at(compiled, global_time=at if global_time is None else global_time, uid=self.uid)

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
        if self.timeline is not None:
            self.timeline._ensure_resolved()  # before reading the timings
        included = [timing for timing in self.timings if timing is not None]
        if not included or t < included[0].start or t > self.end:
            return None
        index = max(0, bisect.bisect_right([timing.start for timing in included], t) - 1)
        timing = included[index]
        if t >= timing.end:
            return timing.compiled_end  # holding until the next segment
        return timing.compiled_at(t)

    def global_time(self, compiled: float) -> float | None:
        """The global time at which the clip shows compiled time
        ``compiled``, or None where the cut or an unfired event skips it."""
        if self.timeline is not None:
            self.timeline._ensure_resolved()
        for timing in self.timings:
            if timing is not None and timing.compiled_start <= compiled <= timing.compiled_end:
                return timing.global_at(compiled)
        return None

    def placed_sounds(self) -> list[tuple[float, float, Sound, float]]:
        """``(global time, skip, sound, volume)`` of the clip's sounds.

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
                    placed.append((timing.start, compiled[0] - at, event.sound, event.volume))
                else:
                    placed.append((timing.global_at(at), 0.0, event.sound, event.volume))
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
class AudioClip:
    """A sound placed on the timeline's audio tracks, outside any scene
    (a voice-over, music, a recording): it plays from ``start`` at its own
    speed, ``cut_in`` to ``cut_out`` seconds of the sound.

    A part of a script (``script`` and ``part``, see manimx/voice/script.py)
    is an audio clip too: before it is recorded it has no sound and lasts
    ``planned`` seconds; once recorded, its take is the sound and the part
    starts ``offset`` seconds into it. Either way its words are events."""

    sound: Sound | None
    start: float = 0.0
    track: int = 0
    cut_in: float | None = None
    cut_out: float | None = None
    volume: float = 1.0
    name: str = ""
    script: Script | None = None
    part: int = 0
    offset: float = 0.0
    """Where the script part starts in the sound (after the countdown of its take)."""
    planned: float = 0.0
    """How long a script part without a sound lasts."""
    fade_in: float = 0.0
    """Seconds over which it rises from silence at its start."""
    fade_out: float = 0.0
    """Seconds over which it falls to silence at its end."""

    @property
    def source_length(self) -> float:
        """Seconds of the sound (or of the planned part)."""
        return self.sound.duration if self.sound is not None else self.planned

    @property
    def length(self) -> float:
        """Seconds the clip plays."""
        cut_in = self.cut_in or 0.0
        cut_out = self.source_length if self.cut_out is None else min(self.cut_out, self.source_length)
        return max(0.0, cut_out - cut_in)

    @property
    def end(self) -> float:
        return self.start + self.length

    def global_time(self, seconds: float) -> float | None:
        """The global time of second ``seconds`` of the sound, None if cut away."""
        cut_in = self.cut_in or 0.0
        if not cut_in <= seconds <= cut_in + self.length:
            return None
        return self.start + seconds - cut_in

    def events(self) -> dict[str, float]:
        """The global times of its script part's words that it plays."""
        if self.script is None:
            return {}
        events = {}
        names = self.script.event_names()[self.part]
        for word, word_names in zip(self.script.parts[self.part].words, names, strict=True):
            time = self.global_time(self.offset + word.start)
            if time is not None:
                events.update(dict.fromkeys(word_names, time))
        return events

    def subtitles(self) -> list[Subtitle]:
        """Its script part's subtitles that it plays, in global time (a
        subtitle partly cut away keeps the part that plays)."""
        from ..voice.transcript import Subtitle

        if self.script is None:
            return []
        cut_in = self.cut_in or 0.0
        shown = []
        for subtitle in self.script.subtitles(self.part):
            start = max(self.offset + subtitle.start, cut_in)
            end = min(self.offset + subtitle.end, cut_in + self.length)
            if start < end:
                shown.append(Subtitle(self.start + start - cut_in, self.start + end - cut_in, subtitle.text))
        return shown

    def placed(self) -> tuple[float, float, Sound, float, float, float] | None:
        """``(global time, skip, sound, volume, fade in, fade out)``, as
        :func:`mix` takes it (None without a sound)."""
        sound = self.sound
        if sound is None:
            return None
        if self.cut_out is not None:
            sound = Sound(sound.samples[: round(self.cut_out * sound.sample_rate)], sound.sample_rate)
        return (self.start, self.cut_in or 0.0, sound, self.volume, self.fade_in, self.fade_out)


@dataclass
class Track:
    """How a track of the timeline plays: its name, and whether it is
    shown, heard and how loud. Scene tracks are the clips' layers, audio
    tracks the sound files' tracks, script tracks those of the script
    parts (their recorded takes); all mute and solo (a scene track's
    sounds are its scenes' play_sound calls)."""

    name: str = ""
    hidden: bool = False
    """Scene tracks: their clips are not drawn."""
    locked: bool = False
    """Not edited in the preview (it only guards against slips)."""
    muted: bool = False
    solo: bool = False
    """Some tracks solo: only those are heard."""
    volume: float = 1.0


_DEFAULT_TRACK = Track()


@dataclass(eq=False)
class Timeline:
    """Clips on one global timeline, with their events, and the audio
    clips on its audio tracks."""

    clips: list[Clip] = field(default_factory=list)
    audio_clips: list[AudioClip] = field(default_factory=list)
    """Sounds on the audio tracks. The words of script parts are events:
    call :meth:`audio_changed` after changing the list or a clip in it."""
    events: dict[str, float] = field(default_factory=dict)
    """Events with a time set directly (fixed, or fired by a presenter)."""
    scene_tracks: dict[int, Track] = field(default_factory=dict)
    """Per layer, how its track plays (missing: as a default Track)."""
    audio_tracks: dict[int, Track] = field(default_factory=dict)
    """Per audio track, how it plays (missing: as a default Track)."""
    script_tracks: dict[int, Track] = field(default_factory=dict)
    """Per script track (the tracks script parts are placed on), how it plays."""
    live: bool = False
    """Live mode: unfired events hold instead of being an error."""
    uid: int = field(default_factory=new_uid, init=False, repr=False)
    """Render identity of the root group of :meth:`state_at`."""
    _dirty: bool = field(default=True, repr=False)

    def add(
        self,
        scene: PreparedScene,
        start: float = 0.0,
        name: str = "",
        scales: dict[int, float] | None = None,
        layer: int = 0,
    ) -> Clip:
        """Place ``scene`` at global time ``start``.

        A prepared scene evaluates by setting its signals, so it can only
        back one clip; to show a scene twice, prepare it twice.
        """
        if any(clip.scene is scene for clip in self.clips):
            raise ValueError("this prepared scene already backs a clip; prepare the scene again to reuse it")
        clip = Clip(scene, start, name or f"clip-{len(self.clips)}", layer=layer, scales=dict(scales or {}))
        clip.timeline = self
        self.clips.append(clip)
        self._dirty = True
        return clip

    def remove(self, clip: Clip) -> None:
        """Take ``clip`` off the timeline."""
        self.clips.remove(clip)
        clip.timeline = None
        self._dirty = True

    def insert(self, index: int, clip: Clip) -> None:
        """Put a removed ``clip`` back at position ``index`` (an undo)."""
        if clip.timeline is not None or any(other.scene is clip.scene for other in self.clips):
            raise ValueError("this clip, or its prepared scene, is on a timeline already")
        clip.timeline = self
        self.clips.insert(index, clip)
        self._dirty = True

    def fire(self, event: str, time: float) -> None:
        """Set the time of ``event`` (a presenter's click, or a fixed time)."""
        self.events[event] = time
        self._dirty = True

    def audio_changed(self) -> None:
        """The audio clips changed: their words' events move with them."""
        self._dirty = True

    def word_events(self) -> dict[str, float]:
        """The events of every script part's words (the first audio
        clip with an event decides its time)."""
        events: dict[str, float] = {}
        for audio in self.audio_clips:
            for event, time in audio.events().items():
                events.setdefault(event, time)
        return events

    def subtitles(self) -> list[Subtitle]:
        """Every audio clip's subtitles, by time."""
        return sorted((subtitle for audio in self.audio_clips for subtitle in audio.subtitles()), key=lambda subtitle: subtitle.start)

    def unfire(self, event: str) -> None:
        """Forget the time set for ``event``: clips waiting for it wait again."""
        if self.events.pop(event, None) is not None:
            self._dirty = True

    def event_times(self) -> dict[str, float]:
        """The time of every event known so far: set (:meth:`fire`), said
        (:meth:`word_events`) or emitted by a placed segment. Unlike
        :meth:`resolve`, never raises for events that do not fire."""
        return self._resolve(strict=False)

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
        ends = [clip.end for clip in self.clips if clip.end != NEVER] + [audio.end for audio in self.audio_clips]
        return max(ends, default=0.0)

    def frame_at(self, t: float) -> tuple[ViewState, ...]:
        """What the frame at global time ``t`` shows: one view per active
        clip, its render state with its scene's camera, in layer order
        (lowest first; equal layers in the order the clips were added).
        The renderer draws each view over the ones before it.
        """
        self._ensure_resolved()
        views = []
        for clip in sorted(self.clips, key=lambda clip: clip.layer):  # stable
            if self.scene_tracks.get(clip.layer, _DEFAULT_TRACK).hidden:
                continue
            compiled = clip.compiled_time(t)
            if compiled is not None:
                views.append(clip.scene.plan.frame_at(compiled, global_time=t, uid=clip.uid))
        return tuple(views)

    def state_at(self, t: float) -> RGroup:
        """The render states of :meth:`frame_at` in one group, one child per
        active clip (without their cameras).
        """
        return RGroup(children=[view.root for view in self.frame_at(t)], uid=self.uid)

    def audio(self, start: float = 0.0, end: float | None = None, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """Every clip's sounds and the audio clips mixed on the global timeline, stereo float32
        for ``[start, end)`` (default: the whole timeline)."""
        soloing = any(track.solo for tracks in (self.scene_tracks, self.audio_tracks, self.script_tracks) for track in tracks.values())

        def heard(track: Track) -> float:
            """How loud a track is heard: 0 when muted, or when others solo."""
            return 0.0 if track.muted or (soloing and not track.solo) else track.volume

        placed = []
        for clip in self.clips:
            loudness = heard(self.scene_tracks.get(clip.layer, _DEFAULT_TRACK))
            placed += [(time, skip, sound, volume * loudness) for time, skip, sound, volume in clip.placed_sounds()]
        for audio in self.audio_clips:
            entry = audio.placed()
            if entry is not None:
                time, skip, sound, volume, fade_in, fade_out = entry
                tracks = self.audio_tracks if audio.script is None else self.script_tracks
                placed.append((time, skip, sound, volume * heard(tracks.get(audio.track, _DEFAULT_TRACK)), fade_in, fade_out))
        return mix([entry for entry in placed if entry[3] > 0.0], start, self.duration if end is None else end, sample_rate)

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
        known = self.word_events() | self.events
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
