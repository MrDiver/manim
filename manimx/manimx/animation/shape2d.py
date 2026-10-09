"""Shape2D: a vector shape in its local plane, and the shapes built on it.

A :class:`Shape2D` snapshots to an :class:`~manimx.objects.RShape2D`: a
path with fill and stroke. It is a leaf; to attach other nodes to it, put
them in a :class:`~manimx.animation.Group` together.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from ..objects import Fill, RNode, RShape2D, Stroke
from ..objects.geometry import arc_points, rectangle, regular_polygon
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


class Shape2D(PositionMixin, RotationMixin, ScalableMixin, Node):
    """A path in the local XY plane with fill and stroke.

    ``color`` sets fill and stroke color together. The fill is transparent
    unless ``fill_opacity`` is given.
    """

    def __init__(
        self,
        points: Any = (),
        name: str | None = None,
        *,
        color: ParsableManimColor = WHITE,
        fill_color: ParsableManimColor | None = None,
        fill_opacity: float = 0.0,
        stroke_color: ParsableManimColor | None = None,
        stroke_width: float = 4.0,
        stroke_opacity: float = 1.0,
        cast_shadows: bool = True,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        prefix = self.name
        self.points = RootSignal(np.zeros((0, 2)) if len(points) == 0 else points, PATH, name=f"{prefix}.points")
        self.fill_color = RootSignal(_rgb(fill_color or color), COLOR, name=f"{prefix}.fill_color")
        self.fill_opacity = RootSignal(fill_opacity, name=f"{prefix}.fill_opacity")
        self.stroke_color = RootSignal(_rgb(stroke_color or color), COLOR, name=f"{prefix}.stroke_color")
        self.stroke_width = RootSignal(stroke_width, name=f"{prefix}.stroke_width")
        self.stroke_opacity = RootSignal(stroke_opacity, name=f"{prefix}.stroke_opacity")
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.points, self.fill_color, self.fill_opacity, self.stroke_color, self.stroke_width, self.stroke_opacity, self.cast_shadows)

    @animatable
    def set_fill(self, color: ParsableManimColor | None = None, opacity: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.fill_color, ABS, _rgb(color)))
        if opacity is not None:
            changes.append((self.fill_opacity, ABS, opacity))
        return changes

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
        return [(self.fill_color, ABS, _rgb(color)), (self.stroke_color, ABS, _rgb(color))]

    @animatable
    def set_cast_shadows(self, cast_shadows: bool) -> ChangeList:
        """Whether this shape blocks light for lit meshes behind it."""
        return [(self.cast_shadows, ABS, cast_shadows)]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alphas = (*alphas, self.alpha)
        alpha = math.prod(signal.value for signal in alphas)
        return RShape2D(
            uid=self.uid,
            points=self.points.value,
            fill=Fill(np.array(self.fill_color.value), opacity=min(1.0, max(0.0, self.fill_opacity.value * alpha))),
            stroke=Stroke(
                np.array(self.stroke_color.value),
                opacity=min(1.0, max(0.0, self.stroke_opacity.value * alpha)),
                width=max(0.0, self.stroke_width.value),
            ),
            cast_shadows=bool(self.cast_shadows.value),
            transform=self._transform(),
            versions={
                "points": version_token(self.points),
                "fill": version_token(self.fill_color, self.fill_opacity, *alphas),
                "stroke": version_token(self.stroke_color, self.stroke_width, self.stroke_opacity, *alphas),
            },
        )


class Circle(Shape2D):
    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(arc_points(radius), name, **style)


class Dot(Shape2D):
    def __init__(self, radius: float = 0.08, name: str | None = None, **style: Any) -> None:
        style.setdefault("fill_opacity", 1.0)
        style.setdefault("stroke_width", 0.0)
        super().__init__(arc_points(radius), name, **style)


class Rectangle(Shape2D):
    def __init__(self, width: float = 4.0, height: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(rectangle(width, height).points, name, **style)


class Square(Shape2D):
    def __init__(self, side_length: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(rectangle(side_length, side_length).points, name, **style)


class RegularPolygon(Shape2D):
    def __init__(self, n: int = 6, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(regular_polygon(n, radius).points, name, **style)


class Polyline(Shape2D):
    """Straight segments through ``corners`` (closed into a polygon if asked)."""

    def __init__(self, corners: Sequence[Sequence[float]], closed: bool = False, name: str | None = None, **style: Any) -> None:
        super().__init__(RShape2D.from_corners(corners, closed=closed).points, name, **style)


class Triangle(RegularPolygon):
    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(3, radius, name, **style)


__all__ = ["Circle", "Dot", "Polyline", "Rectangle", "RegularPolygon", "Shape2D", "Square", "Triangle"]
