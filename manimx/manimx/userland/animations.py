"""The built-in animations: fades, turns, pulses, growing and shrinking.

They are userland: each one only implements the Animation contract
(``targets()``, ``contribution()``, ``nodes()``, ``removes()``) on the
public signals of a node, exactly as a project's own animation would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import quaternion

from ..animation import Animation, Change, Node, PositionMixin, RootSignal, RotationMixin, ScalableMixin, Target
from ..animation.authoring import current as authoring_current
from ..animation.contributions import Contribution, Power, Ramp, Turn
from ..animation.node import require
from ..animation.rates import smooth
from ..animation.signal_tracks import RateFunction
from ..robjects.transform import quaternion_from_axis_angle


@dataclass(eq=False)
class FadeIn(Animation):
    """Fade a node in: its opacity multiplier goes from 0 to 1, optionally
    while it shifts in from ``shift`` away."""

    node: Node
    shift: Sequence[float] | None = None
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        if self.shift is not None:
            require(self.node, PositionMixin, "FadeIn(shift=...)")

    def targets(self) -> Sequence[Target]:
        targets = [Target(self.node.alpha, Change.ABSOLUTE)]
        if self.shift is not None:
            targets += [Target(axis, Change.RELATIVE) for axis in self.node.position.axes]
        return targets

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        if signal is self.node.alpha:
            return alpha
        offset = self.shift[self.node.position.axes.index(signal)]
        return -offset * (1.0 - alpha)  # arrive at the authored position

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        if signal is self.node.alpha:
            return Ramp(1.0)
        return Ramp(float(-self.shift[self.node.position.axes.index(signal)]), remaining=True)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class FadeOut(Animation):
    """Fade a node out (opacity multiplier from 1 to 0) and remove it."""

    node: Node
    shift: Sequence[float] | None = None
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        if self.shift is not None:
            require(self.node, PositionMixin, "FadeOut(shift=...)")

    def targets(self) -> Sequence[Target]:
        targets = [Target(self.node.alpha, Change.ABSOLUTE)]
        if self.shift is not None:
            targets += [Target(axis, Change.RELATIVE) for axis in self.node.position.axes]
        return targets

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        if signal is self.node.alpha:
            return 1.0 - alpha
        return self.shift[self.node.position.axes.index(signal)] * alpha

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        if signal is self.node.alpha:
            return Ramp(1.0, remaining=True)
        return Ramp(float(self.shift[self.node.position.axes.index(signal)]))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)

    def removes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Rotate(Animation):
    """Rotate by ``angle`` about ``axis``. Unlike ``animate.rotate`` (which
    slerps along the shortest way), this sweeps the full angle, so a turn by
    2 pi spins once around."""

    node: Node
    angle: float
    axis: Sequence[float] = (0.0, 0.0, 1.0)
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        require(self.node, RotationMixin, "Rotate")

    def targets(self) -> Sequence[Target]:
        return [Target(self.node.rotation, Change.RELATIVE)]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> quaternion.quaternion:
        return quaternion_from_axis_angle(self.axis, self.angle * alpha)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        x, y, z = (float(c) for c in self.axis)
        return Turn((x, y, z), float(self.angle))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Indicate(Animation):
    """Briefly scale a node up and back (Manim's Indicate, without the
    color flash)."""

    node: Node
    scale_factor: float = 1.2
    run_time: float = 1.0
    rate: RateFunction = staticmethod(lambda t: float(np.sin(np.pi * t)))  # type: ignore[assignment]

    def __post_init__(self) -> None:
        require(self.node, ScalableMixin, "Indicate")

    def targets(self) -> Sequence[Target]:
        return [Target(axis, Change.RELATIVE) for axis in self.node.scale.axes]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        return float(self.scale_factor**alpha)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return Power(float(self.scale_factor))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class GrowFromCenter(Animation):
    """Grow a node from nothing (scale 0) to its authored scale."""

    node: Node
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        require(self.node, ScalableMixin, "GrowFromCenter")
        # The authored scale at the moment the animation is created.
        self._target = tuple(authoring_current(axis) for axis in self.node.scale.axes)

    def targets(self) -> Sequence[Target]:
        return [Target(axis, Change.ABSOLUTE) for axis in self.node.scale.axes]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        return self._target[self.node.scale.axes.index(signal)] * alpha

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return Ramp(float(self._target[self.node.scale.axes.index(signal)]))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class ShrinkToCenter(Animation):
    """Shrink a node to nothing and remove it."""

    node: Node
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        require(self.node, ScalableMixin, "ShrinkToCenter")

    def targets(self) -> Sequence[Target]:
        return [Target(axis, Change.ABSOLUTE) for axis in self.node.scale.axes]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        return start_value * (1.0 - alpha)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return Ramp(None, remaining=True)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)

    def removes(self) -> Sequence[Node]:
        return (self.node,)


__all__ = ["FadeIn", "FadeOut", "GrowFromCenter", "Indicate", "Rotate", "ShrinkToCenter"]
