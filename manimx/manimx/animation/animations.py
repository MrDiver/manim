"""Animations: objects on the timeline, evaluated with a progress ``alpha``.

An animation references the node it animates (it never creates a copy
or a "target node") and is placed on the timeline with a start time and
a duration. It declares which signals it touches and how
(:meth:`Animation.targets`), and at evaluation
:meth:`Animation.contributions` gives what it does to all of them at the
rate-mapped progress, in one call.

Contract for custom animations: ``targets`` and ``contributions`` must be
pure. ``contributions`` may read only its arguments and the animation's
own parameters, never other signals or global state, so that
``state_at(t)`` gives the same answer for every query order. Built-in animations are
dataclasses, so their parameters can be inspected (and shown in an
editor).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import authoring
from .signals.contributions import By, Contribution, To
from .signals.rates import smooth
from .signals.signal import RootSignal, Signal
from .signals.signal_tracks import Change, RateFunction

if TYPE_CHECKING:
    from .audio import Sound
    from .nodes.node import Node


@dataclass(slots=True)
class Target:
    """One signal an animation changes, and how."""

    signal: RootSignal
    mode: Change


class Animation:
    """Base class: override :meth:`targets` and :meth:`contributions`."""

    run_time: float = 1.0
    rate: RateFunction = staticmethod(smooth)  # type: ignore[assignment]

    def targets(self) -> Sequence[Target]:
        raise NotImplementedError

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        """The change (relative targets) or value (absolute targets) of
        each target at progress ``alpha``, in the order of
        :meth:`targets`; ``start_values`` holds each target's value when
        this animation starts, in the same order."""
        raise NotImplementedError

    def planned_contributions(self) -> Sequence[Contribution | None]:
        """:meth:`contributions` as data (see contributions.py), one per
        target in the order of :meth:`targets`, None for a target that is
        called back. Override it together with :meth:`contributions`;
        both must give the same values."""
        return [None] * len(self.targets())

    def nodes(self) -> Sequence[Node]:
        """Nodes that must be in the scene while this animation plays
        (added automatically by ``play``)."""
        return ()

    def removes(self) -> Sequence[Node]:
        """Nodes removed from the scene when this animation ends."""
        return ()

    def audio(self) -> Sequence[tuple[Sound, float, float]]:
        """Sounds this animation plays: ``(sound, volume, when)``, ``when``
        a fraction of its run time (0: as it starts), so they stay on
        their moments whatever run time ``play`` gives it."""
        return ()


class AnimationGroup:
    """Animations played as one: each starts ``lag_ratio`` of the one
    before it's run time after that one starts (0: all together, 1: one
    after another). ``run_time`` stretches or squeezes the whole group to
    that length (default: as long as its animations take). Groups nest.

    A group is not placed on the timeline itself: ``play`` places each
    animation in it at its own start, with its own rate. So a group has no
    rate of its own, and a Simulation cannot be in one.
    """

    def __init__(self, *animations: Animation | AnimationBuilder | AnimationGroup, lag_ratio: float = 0.0, run_time: float | None = None) -> None:
        if not animations:
            raise ValueError(f"{type(self).__name__} needs at least one animation")
        self.animations = [item if isinstance(item, AnimationGroup) else as_animation(item) for item in animations]
        self.lag_ratio = float(lag_ratio)
        self._run_time = run_time

    @property
    def run_time(self) -> float:
        if self._run_time is not None:
            return float(self._run_time)
        return max(start + length for _, start, length in self._natural())

    def schedule(self, run_time: float | None = None) -> list[tuple[Animation, float, float]]:
        """``(animation, start, run time)`` of every animation in the group
        (nested groups opened), starts counted from the group's start,
        fitted to ``run_time`` (default: the group's)."""
        natural = self._natural()
        span = max(start + length for _, start, length in natural)
        factor = (run_time if run_time is not None else self.run_time) / span if span > 0 else 1.0
        return [(animation, start * factor, length * factor) for animation, start, length in natural]

    def _natural(self) -> list[tuple[Animation, float, float]]:
        entries: list[tuple[Animation, float, float]] = []
        start = 0.0
        for item in self.animations:
            inner = item.schedule() if isinstance(item, AnimationGroup) else [(item, 0.0, float(item.run_time))]
            entries += [(animation, start + offset, length) for animation, offset, length in inner]
            start += self.lag_ratio * max(offset + length for _, offset, length in inner)
        return entries


class Succession(AnimationGroup):
    """Animations one after another."""

    def __init__(self, *animations: Animation | AnimationBuilder | AnimationGroup, run_time: float | None = None) -> None:
        super().__init__(*animations, lag_ratio=1.0, run_time=run_time)


class LaggedStart(AnimationGroup):
    """Animations started one after another, each ``lag_ratio`` of the
    previous one's run time after it, so they overlap."""

    def __init__(self, *animations: Animation | AnimationBuilder | AnimationGroup, lag_ratio: float = 0.05, run_time: float | None = None) -> None:
        super().__init__(*animations, lag_ratio=lag_ratio, run_time=run_time)


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

    def at(self, alpha: float, start_value: Any) -> Any:
        """The change, or value, at progress ``alpha``."""
        kind = self.signal.kind
        if self.mode is Change.RELATIVE:
            return kind.scale(self.value, alpha)
        return kind.interpolate(start_value, kind.validate(self.value), alpha)

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        """As an animation of its one signal (tracks in tests take one)."""
        return [self.at(alpha, start_values[0])]

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
    methods: list[str] = field(default_factory=list)
    """The names of the recorded methods, in order (for display)."""

    def add(self, signal: Signal, mode: Change, value: Any) -> None:
        if not isinstance(signal, RootSignal):
            raise TypeError(f"{signal.name} is derived from other signals and cannot be animated")
        existing = self.changes.get(signal.id)
        self.changes[signal.id] = existing.then(mode, value) if existing else SignalChange(signal, mode, value)

    def targets(self) -> Sequence[Target]:
        return [Target(change.signal, change.mode) for change in self.changes.values()]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        return [change.at(alpha, start) for change, start in zip(self.changes.values(), start_values, strict=True)]

    def planned_contributions(self) -> Sequence[Contribution]:
        return [change.planned() for change in self.changes.values()]

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


class AnimationBuilder:
    """``node.animate``: chain changes, then pass the result to ``play``.

        self.play(square.animate.shift(RIGHT).rotate(PI / 4), run_time=2)

    Every method records a change of the node's own signals; nothing is
    copied. The chain describes the end state, step by step: each method
    sees the values the methods before it set. ``run_time`` and ``rate`` can be set here or in ``play``.
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
            self.animation.methods.append(name)
            # Each step sees the end state of the steps before it.
            with authoring.chained(self.animation.changes):
                recorded = list(changes(self._node, *args, **kwargs))
            for signal, mode, value in recorded:
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

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        return [self.change.at(alpha, start_values[0])]

    def planned_contributions(self) -> Sequence[Contribution]:
        return [self.change.planned()]
