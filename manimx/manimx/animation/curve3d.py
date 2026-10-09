"""Curve3D: a stroke-only path in 3D space, and the curves built on it.

A :class:`Curve3D` snapshots to an :class:`~manimx.robjects.RCurve3D`: cubic
Bezier curves whose stroke ribbon always faces the camera. It is a leaf.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from ..robjects import RCurve3D, RNode, Stroke
from ..utils.color import WHITE, ParsableManimColor
from .kinds import COLOR, DISCRETE, PATH
from .node import (
    ABS,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    ScalableMixin,
    _rgb,
    animatable,
)
from .signals import RootSignal, Signal
from .snapshot import version_token


class Curve3D(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Points ``(N, 3)`` in the RBezierPath layout (four per cubic curve)
    with a stroke. The stroke width is in world units."""

    def __init__(
        self,
        points: Any = (),
        name: str | None = None,
        *,
        color: ParsableManimColor = WHITE,
        stroke_width: float = 0.04,
        stroke_opacity: float = 1.0,
        cast_shadows: bool = True,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        prefix = self.name
        self.points = RootSignal(np.zeros((0, 3)) if len(points) == 0 else points, PATH, name=f"{prefix}.points")
        self.stroke_color = RootSignal(_rgb(color), COLOR, name=f"{prefix}.stroke_color")
        self.stroke_width = RootSignal(stroke_width, name=f"{prefix}.stroke_width")
        self.stroke_opacity = RootSignal(stroke_opacity, name=f"{prefix}.stroke_opacity")
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.points, self.stroke_color, self.stroke_width, self.stroke_opacity, self.cast_shadows)

    @animatable
    def set_stroke(self, color: ParsableManimColor | None = None, width: float | None = None, opacity: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.stroke_color, ABS, _rgb(color)))
        if width is not None:
            changes.append((self.stroke_width, ABS, width))
        if opacity is not None:
            changes.append((self.stroke_opacity, ABS, opacity))
        return changes

    @animatable
    def set_color(self, color: ParsableManimColor) -> ChangeList:
        return [(self.stroke_color, ABS, _rgb(color))]

    @animatable
    def set_cast_shadows(self, cast_shadows: bool) -> ChangeList:
        return [(self.cast_shadows, ABS, cast_shadows)]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alphas = (*alphas, self.alpha)
        alpha = math.prod(signal.value for signal in alphas)
        return RCurve3D(
            uid=self.uid,
            points=self.points.value,
            stroke=Stroke(
                np.array(self.stroke_color.value),
                opacity=min(1.0, max(0.0, self.stroke_opacity.value * alpha)),
                width=max(0.0, self.stroke_width.value),
            ),
            cast_shadows=bool(self.cast_shadows.value),
            transform=self._transform(),
            versions={
                "points": version_token(self.points),
                "stroke": version_token(self.stroke_color, self.stroke_width, self.stroke_opacity, *alphas),
            },
        )


class ParametricCurve3D(Curve3D):
    """A smooth curve through ``function(t)`` (a 3D point) at ``samples``
    evenly spaced ``t`` in ``t_range`` (see RCurve3D.parametric)."""

    def __init__(
        self,
        function: Callable[[float], Any],
        t_range: tuple[float, float] = (0.0, 1.0),
        samples: int = 64,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(RCurve3D.parametric(function, t_range, samples).points, name, **style)


__all__ = ["Curve3D", "ParametricCurve3D"]
