"""Scenes: authoring code compiled once, evaluated at any time.

    class Intro(Scene):
        def construct(self):
            square = Square(color=BLUE, fill_opacity=1)
            self.play(FadeIn(square))
            self.play(square.animate.shift(RIGHT * 2), run_time=2)
            self.wait()

    prepared = Intro().prepare()      # pass 1: runs construct() once
    frame = prepared.state_at(1.5)    # pass 2: any time, any order

``construct()`` never renders and never sleeps. Each ``play`` places
animations on the scene's local timeline at the current cursor and moves
the cursor; ``wait`` only moves the cursor. The result is a
:class:`PreparedScene`: tracks of placed animations per signal, plus the
mobjects in the order they were added (later ones are drawn on top, as in
Manim).

Events (``wait_for``, ``emit``) and several scenes on one timeline come
with the session layer; this module covers a single scene clip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..objects import Group as RenderGroup
from .animations import Animation, AnimationBuilder, SetValue, SignalChange, as_animation
from .authoring import _active
from .mobjects import Mobject
from .signals import RootSignal, Signal, TimeSignals
from .tracks import Change, Placement, RateFunction, Track


class Scene:
    """Subclass and override :meth:`construct`, then call :meth:`prepare`."""

    def __init__(self) -> None:
        self.time = TimeSignals()
        self._compiler: SceneCompiler | None = None
        self._prepared: PreparedScene | None = None

    def construct(self) -> None:
        """Describe the scene with ``add``, ``remove``, ``play`` and ``wait``."""

    def prepare(self) -> PreparedScene:
        """Run :meth:`construct` once and return the evaluable scene."""
        if self._prepared is None:
            compiler = SceneCompiler()
            self._compiler = compiler
            token = _active.set(compiler)
            try:
                self.construct()
            finally:
                _active.reset(token)
                self._compiler = None
            self._prepared = compiler.finish(self.time)
        return self._prepared

    # -- authoring API (only inside construct) --------------------------------

    def add(self, *mobjects: Mobject) -> None:
        """Put mobjects into the scene at the current time."""
        self._authoring().add(mobjects)

    def remove(self, *mobjects: Mobject) -> None:
        """Take mobjects out of the scene at the current time."""
        self._authoring().remove(mobjects)

    def play(
        self,
        *animations: Animation | AnimationBuilder,
        run_time: float | None = None,
        rate: RateFunction | None = None,
        lag_ratio: float = 0.0,
    ) -> None:
        """Place animations at the current time and wait until they end.

        ``run_time`` and ``rate`` override the animations' own. With
        ``lag_ratio`` > 0 each animation starts that fraction of its run
        time after the previous one (Manim's LaggedStart).
        """
        self._authoring().play([as_animation(a) for a in animations], run_time, rate, lag_ratio)

    def wait(self, duration: float = 1.0) -> None:
        """Let ``duration`` seconds pass."""
        self._authoring().wait(duration)

    def _authoring(self) -> SceneCompiler:
        if self._compiler is None:
            raise RuntimeError("add, remove, play and wait can only be used inside construct()")
        return self._compiler


class SceneCompiler:
    """Builds the timeline while ``construct()`` runs (see authoring.py)."""

    def __init__(self) -> None:
        self.cursor = 0.0
        self.order = 0
        self.tracks: dict[int, Track] = {}
        self.roots: list[Mobject] = []
        self.present: set[int] = set()
        self.timeline_signals: set[int] = set()
        self.placements: list[Placement] = []

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

    def add(self, mobjects: Sequence[Mobject]) -> None:
        for mobject in mobjects:
            if mobject.uid in self.present:
                continue
            for member in mobject.family():
                self.timeline_signals.update(signal.id for signal in member.signals())
                # A member already in the scene on its own is drawn through
                # this mobject from now on; until now it stays on its own.
                if member is not mobject and member.uid in self.present:
                    self.remove([member])
            if mobject not in self.roots:
                self.roots.append(mobject)
            self.present.add(mobject.uid)
            self.record_instant(mobject.present, Change.ABSOLUTE, True)

    def remove(self, mobjects: Sequence[Mobject]) -> None:
        for mobject in mobjects:
            if mobject.uid in self.present:
                self.present.discard(mobject.uid)
                self.record_instant(mobject.present, Change.ABSOLUTE, False)

    def play(self, animations: Sequence[Animation], run_time: float | None, rate: RateFunction | None, lag_ratio: float) -> None:
        if not animations:
            raise ValueError("play() needs at least one animation")
        start = self.cursor
        end = start
        for index, animation in enumerate(animations):
            duration = animation.run_time if run_time is None else run_time
            begin = start + index * lag_ratio * duration
            # Mobjects appear when their own animation begins: a lagged
            # FadeIn must not show its mobject before its fade starts.
            self.cursor = begin
            self.add([m for m in animation.mobjects() if not self._is_inside_present(m)])
            self.place(animation, begin, duration, rate or animation.rate)
            end = max(end, begin + duration)
            if animation.removes():
                self.cursor = begin + duration
                self.remove(animation.removes())
        self.cursor = end

    def wait(self, duration: float) -> None:
        if duration < 0:
            raise ValueError("cannot wait a negative time")
        self.cursor += duration

    def place(self, animation: Animation, start: float, duration: float, rate: RateFunction | None = None) -> Placement:
        placement = Placement(animation, start, duration, self.order, rate or animation.rate)
        self.order += 1
        for target in animation.targets():
            if not isinstance(target.signal, RootSignal):
                raise TypeError(f"{target.signal.name} is derived from other signals and cannot be animated")
            track = self.tracks.get(target.signal.id)
            if track is None:
                track = self.tracks[target.signal.id] = Track(target.signal)
            track.add(placement, target.mode)
        self.placements.append(placement)
        return placement

    def finish(self, time: TimeSignals) -> PreparedScene:
        duration = max([self.cursor, *(track.end for track in self.tracks.values())])
        return PreparedScene(duration, list(self.tracks.values()), list(self.roots), time, list(self.placements))

    def _is_inside_present(self, mobject: Mobject) -> bool:
        """Present itself, or part of a present group."""
        return mobject.uid in self.present or any(
            mobject in root.family() and root.uid in self.present for root in self.roots
        )


@dataclass(eq=False)
class PreparedScene:
    """A compiled scene: evaluate it at any local time with :meth:`state_at`."""

    duration: float
    tracks: list[Track]
    roots: list[Mobject]
    time: TimeSignals
    placements: list[Placement] = field(default_factory=list)

    def state_at(self, t: float) -> RenderGroup:
        """Render state at local time ``t`` (pure: no dependence on earlier calls).

        Every animated root signal is set to its value at ``t``; derived
        signals follow by propagation; then the present mobjects are
        snapshotted in the order they were added.
        """
        self.time.set(scene_time=t, global_time=t)
        for track in self.tracks:
            track.signal.set(track.value_at(t))
        children = [snapshot for mobject in self.roots if mobject.present.value and (snapshot := mobject.snapshot()) is not None]
        return RenderGroup("scene", children=children)

    def refresh(self) -> None:
        """Recompute cached start values after placements were edited."""
        for track in self.tracks:
            track.refresh()
