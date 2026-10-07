"""Signals: the reactive values every animatable property is made of.

There are two kinds of signal (section 3.2 of
docs/animation_engine_design.md):

* :class:`RootSignal` holds a value that is set from outside: by authoring
  code, and over time by animations and baked physics.
* :class:`DerivedSignal` is a pure function of other signals. It replaces
  Manim's updaters: ``label_x = derive(lambda x: x + 0.5, dot_x)``.

Propagation is **push-based invalidation with pull evaluation**. Setting a
root marks everything that depends on it as dirty (and notifies
subscribers); a dirty derived signal recomputes only when it is read. So
setting many roots and then building one snapshot computes every derived
value at most once, in dependency order.

Every signal has a process-unique :attr:`Signal.id` and a :attr:`version`
that increases exactly when its value changes. Renderer caches use these
versions: an unchanged signal never causes recomputation downstream.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from typing import Any

from .kinds import SCALAR, SignalKind

_next_id = itertools.count(1)

Listener = Callable[["Signal"], None]


class Signal:
    """Base of root and derived signals."""

    __slots__ = ("id", "name", "kind", "_dependents", "_listeners", "_version", "_value")

    def __init__(self, kind: SignalKind, name: str) -> None:
        self.id = next(_next_id)
        self.name = name or f"signal-{self.id}"
        self.kind = kind
        self._dependents: list[DerivedSignal] = []
        self._listeners: list[Listener] = []
        self._version = 0
        self._value: Any = None

    @property
    def value(self) -> Any:
        raise NotImplementedError

    @property
    def version(self) -> int:
        """Increases exactly when :attr:`value` changes (reading may update it)."""
        self.value  # noqa: B018 - bring derived signals up to date first
        return self._version

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Call ``listener(signal)`` whenever this signal may have changed.

        Root signals notify when set to a new value; derived signals notify
        when one of their inputs did (read :attr:`value` to get the new
        value). Returns a function that unsubscribes.
        """
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def dependents(self) -> Iterator[DerivedSignal]:
        """All signals computed from this one, directly or indirectly."""
        seen: set[int] = set()
        stack = list(self._dependents)
        while stack:
            signal = stack.pop()
            if signal.id not in seen:
                seen.add(signal.id)
                yield signal
                stack.extend(signal._dependents)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.kind.name})"


class RootSignal(Signal):
    """A signal whose value is set from outside."""

    __slots__ = ()

    def __init__(self, value: Any, kind: SignalKind = SCALAR, name: str = "") -> None:
        super().__init__(kind, name)
        self._value = kind.validate(value)

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self.set(value)

    def set(self, value: Any) -> bool:
        """Store ``value``; returns True if it differs from the current one.

        Setting an equal value changes nothing: no version bump, no
        invalidation, no notification.
        """
        value = self.kind.validate(value)
        if self.kind.equal(value, self._value):
            return False
        self._value = value
        self._version += 1
        self._notify()
        for dependent in self._dependents:
            dependent._invalidate()
        return True


class DerivedSignal(Signal):
    """A signal computed by a pure function from other signals.

    The function receives the input values in order. It must not have side
    effects or read anything but its inputs: evaluation is lazy and may
    happen in any order, at any time.
    """

    __slots__ = ("_function", "_inputs", "_dirty")

    def __init__(
        self,
        function: Callable[..., Any],
        inputs: tuple[Signal, ...],
        kind: SignalKind = SCALAR,
        name: str = "",
    ) -> None:
        super().__init__(kind, name)
        self._function = function
        self._inputs: tuple[Signal, ...] = ()
        self._dirty = True
        self._connect(inputs)

    @property
    def inputs(self) -> tuple[Signal, ...]:
        return self._inputs

    @property
    def value(self) -> Any:
        if self._dirty:
            self._recompute()
        return self._value

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Like :meth:`Signal.subscribe`. A derived signal notifies when it
        goes stale after having been read, so a listener should read the
        value once after subscribing (and after each notification)."""
        return super().subscribe(listener)

    def rebind(self, function: Callable[..., Any], inputs: tuple[Signal, ...]) -> None:
        """Replace the function and inputs (e.g. a label now follows another
        object). Rejected if it would make the signal depend on itself."""
        for signal in inputs:
            if signal is self or any(dependent is signal for dependent in self.dependents()):
                raise ValueError(f"{self.name} cannot depend on {signal.name}: that would create a cycle")
        self._disconnect()
        self._function = function
        self._connect(inputs)
        self._invalidate()

    def _connect(self, inputs: tuple[Signal, ...]) -> None:
        self._inputs = tuple(inputs)
        for signal in self._inputs:
            signal._dependents.append(self)

    def _disconnect(self) -> None:
        for signal in self._inputs:
            signal._dependents.remove(self)
        self._inputs = ()

    def _invalidate(self) -> None:
        if self._dirty:
            return  # dependents were already invalidated with us
        self._dirty = True
        self._notify()
        for dependent in self._dependents:
            dependent._invalidate()

    def _recompute(self) -> None:
        value = self.kind.validate(self._function(*(signal.value for signal in self._inputs)))
        self._dirty = False
        if self._version == 0 or not self.kind.equal(value, self._value):
            self._value = value
            self._version += 1


def derive(function: Callable[..., Any], *inputs: Signal, kind: SignalKind = SCALAR, name: str = "") -> DerivedSignal:
    """Shorthand for :class:`DerivedSignal`: ``derive(lambda x: x + 1, x_signal)``."""
    return DerivedSignal(function, inputs, kind, name)


class Vector3Signal:
    """Three scalar root signals ``x``, ``y``, ``z`` that read as a tuple.

    Positions and scales are split per axis, so an animation only touches
    the axes it changes: "move right" and "move down" running at the same
    time do not interfere (section 3.1 of the design doc).
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, value: tuple[float, float, float] = (0.0, 0.0, 0.0), name: str = "") -> None:
        x, y, z = value
        self.x = RootSignal(x, name=f"{name}.x")
        self.y = RootSignal(y, name=f"{name}.y")
        self.z = RootSignal(z, name=f"{name}.z")

    @property
    def axes(self) -> tuple[RootSignal, RootSignal, RootSignal]:
        return (self.x, self.y, self.z)

    @property
    def value(self) -> tuple[float, float, float]:
        return (self.x.value, self.y.value, self.z.value)

    @value.setter
    def value(self, value: tuple[float, float, float]) -> None:
        self.set(value)

    def set(self, value: tuple[float, float, float]) -> bool:
        """Set all three axes; returns True if any of them changed."""
        changed = [axis.set(component) for axis, component in zip(self.axes, value)]
        return any(changed)

    @property
    def version(self) -> tuple[int, int, int]:
        return (self.x.version, self.y.version, self.z.version)


class TimeSignals:
    """The two built-in time roots (section 3.3 of the design doc).

    ``scene_time`` is the scene's local time and stands still while the
    scene holds at a ``wait_for``; ``global_time`` is the session time and
    keeps running, so idle motion continues during holds. Evaluation sets
    both before reading anything, which keeps ``state_at(t)`` deterministic.
    """

    __slots__ = ("scene_time", "global_time")

    def __init__(self) -> None:
        self.scene_time = RootSignal(0.0, name="scene_time")
        self.global_time = RootSignal(0.0, name="global_time")

    def set(self, scene_time: float, global_time: float) -> None:
        self.scene_time.set(scene_time)
        self.global_time.set(global_time)
