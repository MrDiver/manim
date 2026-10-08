"""Timing: placing scene clips on the global timeline and resolving events.

Section 6 of docs/animation_engine_design.md. A :class:`Timeline` holds
clips (prepared scenes placed at a global start time) and event times. It
decides when each segment of each clip starts:

* the first segment starts at the clip's start;
* a later segment starts at the time of the event it waits for, or when
  the previous segment ends if that is later (like a presenter click that
  arrives while the previous part is still playing);
* an event's time is either set directly (:meth:`Timeline.fire`: a fixed
  time from a config or an editor, or a presenter's click) or the global
  time of an ``emit`` in some segment, which depends on when that segment
  starts. Events are therefore resolved in dependency order.

Between segments a clip holds: it shows the end of the previous segment
(``scene_time`` stands still, ``global_time`` keeps running). A segment
can be stretched with a time scale; a scale of 2 plays it at half speed.

Editing (adding clips, firing events) always resolves leniently, so clips
can be added in any order: an event emitted by a clip added later simply
is not known yet. :meth:`Timeline.resolve` is the strict check an offline
render runs: it rejects events that never fire, including cycles where
segments wait for each other's emits. In live mode (a running
presentation) such segments hold until their event fires.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..objects import Group as RenderGroup
from .scene import PreparedScene

NEVER = float("inf")
"""Start time of a segment whose event has not fired (yet)."""


class UnresolvedEvents(ValueError):
    """Events that segments wait for but nothing fires (or that form a cycle)."""


@dataclass(eq=False)
class Clip:
    """A prepared scene placed on the global timeline."""

    scene: PreparedScene
    start: float = 0.0
    name: str = ""
    scales: dict[int, float] = field(default_factory=dict)
    """Time scale per segment index (default 1: real time)."""
    segment_starts: list[float] = field(default_factory=list, init=False)
    """Global start time per segment, filled in by :meth:`Timeline.resolve`."""

    def scale(self, segment: int) -> float:
        return self.scales.get(segment, 1.0)

    def segment_end(self, index: int) -> float:
        segment = self.scene.segments[index]
        return self.segment_starts[index] + segment.duration * self.scale(index)

    @property
    def end(self) -> float:
        return self.segment_end(len(self.scene.segments) - 1)

    def compiled_time(self, t: float) -> float | None:
        """The clip's compiled time at global time ``t``, or None outside it.

        Inside a segment, compiled time runs at the segment's scale; between
        segments it holds at the end of the previous one.
        """
        if t < self.start or t > self.end:
            return None
        index = max(0, bisect.bisect_right(self.segment_starts, t) - 1)
        segment = self.scene.segments[index]
        elapsed = (t - self.segment_starts[index]) / self.scale(index)
        return segment.offset + min(elapsed, segment.duration)


@dataclass(eq=False)
class Timeline:
    """Clips on one global timeline, with their events."""

    clips: list[Clip] = field(default_factory=list)
    events: dict[str, float] = field(default_factory=dict)
    """Events with a time set directly (fixed, or fired by a presenter)."""
    live: bool = False
    """Live mode: unfired events hold instead of being an error."""

    def add(self, scene: PreparedScene, start: float = 0.0, name: str = "", scales: dict[int, float] | None = None) -> Clip:
        clip = Clip(scene, start, name or f"clip-{len(self.clips)}", dict(scales or {}))
        self.clips.append(clip)
        self._resolve(strict=False)
        return clip

    def fire(self, event: str, time: float) -> None:
        """Set the time of ``event`` (a presenter's click, or a fixed time)."""
        self.events[event] = time
        self._resolve(strict=False)

    def resolve(self) -> dict[str, float]:
        """Compute every segment's start; returns the resolved event times.

        Outside live mode, raises :class:`UnresolvedEvents` if a segment
        waits for an event that never fires.
        """
        return self._resolve(strict=not self.live)

    def _resolve(self, strict: bool) -> dict[str, float]:
        """Compute every segment's start; returns the resolved event times.

        Iterates to a fixed point: each pass places the segments whose
        events are known, which may reveal emitted events for the next
        pass. Event times only ever come from earlier segments, so this
        terminates after at most one pass per segment.
        """
        known = dict(self.events)
        for clip in self.clips:
            clip.segment_starts = [clip.start] + [NEVER] * (len(clip.scene.segments) - 1)
        changed = True
        while changed:
            changed = False
            for clip in self.clips:
                for index, segment in enumerate(clip.scene.segments):
                    start = clip.segment_starts[index]
                    if index > 0:
                        previous_end = clip.segment_end(index - 1)
                        start = max(known.get(segment.waits_for, NEVER), previous_end)
                        if start != clip.segment_starts[index]:
                            clip.segment_starts[index] = start
                            changed = True
                    if start == NEVER:
                        continue
                    for emit in segment.emits:
                        time = start + emit.time * clip.scale(index)
                        if emit.event not in self.events and known.get(emit.event, NEVER) > time:
                            # The earliest emit of an event wins.
                            known[emit.event] = time
                            changed = True
        if strict:
            missing = sorted(
                {s.waits_for for c in self.clips for s in c.scene.segments if s.waits_for and s.waits_for not in known}
            )
            if missing:
                raise UnresolvedEvents(f"events never fire (missing or waiting on each other): {missing}")
        return known

    @property
    def duration(self) -> float:
        ends = [clip.end for clip in self.clips if clip.end != NEVER]
        return max(ends, default=0.0)

    def state_at(self, t: float) -> RenderGroup:
        """Render state of all clips at global time ``t``; later clips are
        drawn on top of earlier ones."""
        children = []
        for clip in self.clips:
            compiled = clip.compiled_time(t)
            if compiled is not None:
                state = clip.scene.state_at(compiled, global_time=t)
                children.append(RenderGroup(clip.name, children=state.children))
        return RenderGroup("timeline", children=children)

    def segments(self) -> Iterable[tuple[Clip, int, float, float]]:
        """``(clip, segment index, global start, global end)`` for display."""
        for clip in self.clips:
            for index in range(len(clip.scene.segments)):
                yield clip, index, clip.segment_starts[index], clip.segment_end(index)
