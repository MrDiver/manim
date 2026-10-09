"""Scene plans: a prepared scene as data, evaluated by the core.

    plan = ScenePlan(prepared)        # once, after Scene.prepare()
    plan.evaluate(2.5)                # any time, any order
    plan.value(square.position.x)     # what the signal has at 2.5

A :class:`ScenePlan` turns a :class:`~.scene.PreparedScene` into the
tables of the core's evaluator (``manimx._core.Evaluator``, see CORE.md):

* **signals:** every signal the scene can read, each with a slot: the
  time signals, the signals of every track, of every node in the scene
  and of the camera, and every input of a derived signal. A derived
  signal comes after its inputs, so the evaluator computes them in slot
  order.
* **tracks:** per animated root signal its base value and its entries.
  An entry's contribution is data when its animation describes it
  (:meth:`~.animations.Animation.planned_contribution`), else a call back
  to ``Animation.contribution()``. Rates are data for ``linear`` and
  ``smooth``, else called back.
* **derived signals:** offsets of :meth:`~.node.PositionMixin.follow` are
  data; every other function is called back with the input values.

Evaluating sets the evaluator's slots, not the Python signals: the
:class:`~.scene.PreparedScene` keeps its own state, and both can be
compared value by value (tests/test_plan.py).
"""

from __future__ import annotations

from typing import Any

from .. import _core
from .contributions import By, Contribution, Keyframes, Power, Ramp, To, Turn
from .kinds import (
    ArrayKind,
    ColorKind,
    ColorsKind,
    DiscreteKind,
    FactorKind,
    PathKind,
    RotationKind,
    ScalarKind,
    ScalarsKind,
    SignalKind,
)
from .node import Offset
from .rates import linear, smooth
from .scene import PreparedScene
from .signal_tracks import Change, Entry, RateFunction, SignalTrack
from .signals import DerivedSignal, Signal

_KIND_CODES: dict[type, str] = {
    ScalarKind: "scalar",
    FactorKind: "factor",
    RotationKind: "rotation",
    ColorKind: "color",
    ColorsKind: "colors",
    ScalarsKind: "scalars",
    PathKind: "path",
    ArrayKind: "array",
    DiscreteKind: "discrete",
}
"""Kinds the core implements, by exact type (a subclass may change any
operation, so it is called back like any other kind)."""


class ScenePlan:
    """A prepared scene in the core's evaluator (see the module docs)."""

    def __init__(self, scene: PreparedScene) -> None:
        self.scene = scene
        self._slots: dict[int, int] = {}
        self._signals: list[Signal] = []
        for signal in _readable_signals(scene):
            self._add_signal(signal)
        self.evaluator = _core.Evaluator(
            [_kind_code(signal.kind) for signal in self._signals],
            [signal.value for signal in self._signals],
            self.slot(scene.time.scene_time),
            self.slot(scene.time.global_time),
        )
        for track in scene.tracks:
            self.evaluator.add_track(self.slot(track.signal), track.base, [_entry(entry, track) for entry in track.entries])
        for signal in self._signals:
            if isinstance(signal, DerivedSignal):
                self.evaluator.add_derived(self.slot(signal), [self.slot(i) for i in signal.inputs], _derivation(signal))

    @property
    def signals(self) -> list[Signal]:
        """Every signal in the plan, in slot order."""
        return list(self._signals)

    def slot(self, signal: Signal) -> int:
        return self._slots[signal.id]

    def evaluate(self, t: float, global_time: float | None = None) -> None:
        """Set every slot to its value at compiled time ``t``, like
        :meth:`PreparedScene.evaluate` sets the signals.
        """
        self.evaluator.evaluate(t, self.scene.playing_time_at(t), t if global_time is None else global_time)

    def value(self, signal: Signal) -> Any:
        """The value of ``signal`` at the last evaluated time."""
        return self.evaluator.value(self.slot(signal))

    def version(self, signal: Signal) -> int:
        """Increases exactly when the value of ``signal`` changes."""
        return self.evaluator.version(self.slot(signal))

    def _add_signal(self, signal: Signal) -> None:
        if signal.id in self._slots:
            return
        if isinstance(signal, DerivedSignal):
            for source in signal.inputs:
                self._add_signal(source)  # inputs first: dependency order
        self._slots[signal.id] = len(self._signals)
        self._signals.append(signal)


def _readable_signals(scene: PreparedScene) -> list[Signal]:
    """Every signal evaluation sets or a snapshot reads."""
    signals: list[Signal] = [scene.time.scene_time, scene.time.global_time]
    signals += [track.signal for track in scene.tracks]
    for root in scene.roots:
        for node in root.family():
            signals += node.signals()
    signals += scene.camera.signals()
    return signals


def _kind_code(kind: SignalKind) -> tuple[str, SignalKind]:
    return (_KIND_CODES.get(type(kind), "python"), kind)


def _entry(entry: Entry, track: SignalTrack) -> tuple:
    placement = entry.placement
    planned = placement.animation.planned_contribution(track.signal)  # type: ignore[attr-defined]
    contribution = _contribution(planned) if planned is not None else ("python", placement.animation, track.signal)
    return (
        placement.start,
        placement.duration,
        _rate(placement.rate),
        entry.mode is Change.RELATIVE,
        entry.start_value,
        contribution,
    )


def _rate(rate: RateFunction) -> tuple:
    if rate is linear:
        return ("linear",)
    if rate is smooth:
        return ("smooth", 10.0)
    return ("python", rate)


def _contribution(planned: Contribution) -> tuple:
    if isinstance(planned, To):
        return ("to", planned.value)
    if isinstance(planned, By):
        return ("by", planned.change)
    if isinstance(planned, Ramp):
        return ("ramp", planned.factor, planned.remaining)
    if isinstance(planned, Power):
        return ("power", planned.base)
    if isinstance(planned, Turn):
        return ("turn", planned.axis, planned.angle)
    if isinstance(planned, Keyframes):
        return ("keyframes", planned.run_time, list(planned.times), list(planned.values))
    raise TypeError(f"unknown contribution {planned!r}")


def _derivation(signal: DerivedSignal) -> tuple:
    function = signal.function
    if isinstance(function, Offset):
        return ("offset", function.offset)
    return ("python", function)


__all__ = ["ScenePlan"]
