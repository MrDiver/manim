"""Animations: objects on the timeline, evaluated with a progress ``alpha``.

An animation references the node it animates (it never creates a copy
or a "target node") and is placed on the timeline with a start time and
a duration. It declares which signals it touches and how
(:meth:`Animation.targets`), and at evaluation the tracks of those signals
call :meth:`Animation.contribution` with the rate-mapped progress.

Contract for custom animations: ``targets`` and ``contribution`` must be
pure. ``contribution`` may read only its arguments and the animation's own
parameters, never other signals or global state, so that ``state_at(t)``
gives the same answer for every query order. Built-in animations are
dataclasses, so their parameters can be inspected (and shown in an
editor).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .contributions import By, Contribution, To
from .rates import smooth
from .signal_tracks import Change, RateFunction
from .signals import RootSignal, Signal

if TYPE_CHECKING:
    from .node import Node


@dataclass(slots=True)
class Target:
    """One signal an animation changes, and how."""

    signal: RootSignal
    mode: Change


class Animation:
    """Base class: override :meth:`targets` and :meth:`contribution`."""

    run_time: float = 1.0
    rate: RateFunction = staticmethod(smooth)  # type: ignore[assignment]

    def targets(self) -> Sequence[Target]:
        raise NotImplementedError

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        """The change (relative targets) or value (absolute targets) of
        ``signal`` at progress ``alpha``; ``start_value`` is the signal's
        value when this animation starts."""
        raise NotImplementedError

    def planned_contribution(self, signal: RootSignal) -> Contribution | None:
        """:meth:`contribution` for ``signal`` as data (see
        contributions.py), or None to be called back. Override it together
        with :meth:`contribution`; both must give the same values."""
        return None

    def nodes(self) -> Sequence[Node]:
        """Nodes that must be in the scene while this animation plays
        (added automatically by ``play``)."""
        return ()

    def removes(self) -> Sequence[Node]:
        """Nodes removed from the scene when this animation ends."""
        return ()


# --------------------------------------------------------------------------
# Plain signal changes and the .animate builder
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SignalChange:
    """A plain change of one signal: by an amount, or to a value."""

    signal: RootSignal
    mode: Change
    value: Any

    def then(self, mode: Change, value: Any) -> SignalChange:
        """This change followed by another one on the same signal."""
        kind = self.signal.kind
        if mode is Change.ABSOLUTE:
            return SignalChange(self.signal, Change.ABSOLUTE, value)
        if self.mode is Change.ABSOLUTE:
            return SignalChange(self.signal, Change.ABSOLUTE, kind.combine(kind.validate(self.value), value))
        return SignalChange(self.signal, Change.RELATIVE, kind.combine(self.value, value))

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        kind = self.signal.kind
        if self.mode is Change.RELATIVE:
            return kind.scale(self.value, alpha)
        return kind.interpolate(start_value, kind.validate(self.value), alpha)

    def planned(self) -> Contribution:
        if self.mode is Change.RELATIVE:
            return By(self.value)
        return To(self.signal.kind.validate(self.value))


@dataclass(eq=False)
class ChangeSignals(Animation):
    """Plain changes of a node's signals: what ``node.animate`` builds.

    Several changes of the same signal merge into one (two shifts add up, a
    move_to followed by a shift moves to the shifted point).
    """

    node: Node
    changes: dict[int, SignalChange] = field(default_factory=dict)
    run_time: float = 1.0
    rate: RateFunction = smooth

    def add(self, signal: Signal, mode: Change, value: Any) -> None:
        if not isinstance(signal, RootSignal):
            raise TypeError(f"{signal.name} is derived from other signals and cannot be animated")
        existing = self.changes.get(signal.id)
        self.changes[signal.id] = existing.then(mode, value) if existing else SignalChange(signal, mode, value)

    def targets(self) -> Sequence[Target]:
        return [Target(change.signal, change.mode) for change in self.changes.values()]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        return self.changes[signal.id].contribution(signal, alpha, start_value)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return self.changes[signal.id].planned()

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


class AnimationBuilder:
    """``node.animate``: chain changes, then pass the result to ``play``.

        self.play(square.animate.shift(RIGHT).rotate(PI / 4), run_time=2)

    Every method records a change of the node's own signals; nothing is
    copied. ``run_time`` and ``rate`` can be set here or in ``play``.
    """

    def __init__(self, node: Node) -> None:
        self._node = node
        self.animation = ChangeSignals(node)

    def __getattr__(self, name: str) -> Any:
        method = getattr(type(self._node), name, None)
        changes = getattr(method, "_signal_changes", None)
        if changes is None:
            getattr(self._node, name)  # a missing mixin part explains itself
            raise AttributeError(f"{type(self._node).__name__}.{name} cannot be animated")

        def record(*args: Any, **kwargs: Any) -> AnimationBuilder:
            for signal, mode, value in changes(self._node, *args, **kwargs):
                self.animation.add(signal, mode, value)
            return self

        return record

    def set_run_time(self, run_time: float) -> AnimationBuilder:
        self.animation.run_time = run_time
        return self

    def set_rate(self, rate: RateFunction) -> AnimationBuilder:
        self.animation.rate = rate
        return self


def as_animation(item: Animation | AnimationBuilder) -> Animation:
    return item.animation if isinstance(item, AnimationBuilder) else item


@dataclass(eq=False)
class SetValue(Animation):
    """An instantaneous change, e.g. ``square.shift(RIGHT)`` between plays
    (see authoring.py), or a node being added or removed."""

    change: SignalChange
    run_time: float = 0.0

    def targets(self) -> Sequence[Target]:
        return [Target(self.change.signal, self.change.mode)]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        return self.change.contribution(signal, alpha, start_value)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return self.change.planned()
