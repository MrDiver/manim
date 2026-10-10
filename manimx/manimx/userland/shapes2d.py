"""Shapes built on Shape2D, each with its parameters as signals.

A shape here is a proxy for its parameters: ``Circle.radius``,
``Rectangle.width`` and so on are signals, and the shape's ``points`` are
derived from them (Shape2D._derive_points), so animating a parameter
reshapes the path at every frame::

    circle = Circle(1.0)
    self.play(circle.animate.set_radius(2.0))

They are userland: a project defines its own shapes the same way, with
the point generators of manimx.userland.geometry. No shape picks a color.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..animation import (
    ARRAY,
    DISCRETE,
    SCALAR,
    RootSignal,
    Shape2D,
    Signal,
    SignalKind,
    animatable,
)
from ..animation.node import ABS, ChangeList
from .geometry import (
    arc_points,
    corner_points,
    rectangle_points,
    regular_polygon_points,
)


def parameter(node: Shape2D, field: str, value: Any, kind: SignalKind = SCALAR) -> RootSignal:
    """A parameter signal of ``node`` named ``<node>.<field>``."""
    return RootSignal(value, kind, name=f"{node.name}.{field}")


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


class Circle(Shape2D):
    """A circle of ``radius`` around the origin."""

    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.radius = parameter(self, "radius", _positive("radius", radius))
        self._derive_points(arc_points, self.radius)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.radius

    @animatable
    def set_radius(self, radius: float) -> ChangeList:
        return [(self.radius, ABS, _positive("radius", radius))]


class Dot(Circle):
    """A small filled circle without an outline."""

    def __init__(self, radius: float = 0.08, name: str | None = None, **style: Any) -> None:
        style.setdefault("fill_opacity", 1.0)
        style.setdefault("stroke_width", 0.0)
        super().__init__(radius, name, **style)


class Rectangle(Shape2D):
    """A ``width`` by ``height`` rectangle centered on the origin."""

    def __init__(self, width: float = 4.0, height: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.width = parameter(self, "width", _positive("width", width))
        self.height = parameter(self, "height", _positive("height", height))
        self._derive_points(rectangle_points, self.width, self.height)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.width, self.height)

    @animatable
    def set_size(self, width: float | None = None, height: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if width is not None:
            changes.append((self.width, ABS, _positive("width", width)))
        if height is not None:
            changes.append((self.height, ABS, _positive("height", height)))
        return changes


class Square(Shape2D):
    """A square with sides ``side_length`` long, centered on the origin."""

    def __init__(self, side_length: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.side_length = parameter(self, "side_length", _positive("side_length", side_length))
        self._derive_points(_square_points, self.side_length)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.side_length

    @animatable
    def set_side_length(self, side_length: float) -> ChangeList:
        return [(self.side_length, ABS, _positive("side_length", side_length))]


class RegularPolygon(Shape2D):
    """A regular polygon with ``n`` corners on a circle of ``radius``;
    odd ones have a corner pointing up."""

    def __init__(self, n: int = 6, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.n = parameter(self, "n", _corners(n), DISCRETE)
        self.radius = parameter(self, "radius", _positive("radius", radius))
        self._derive_points(regular_polygon_points, self.n, self.radius)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.n, self.radius)

    @animatable
    def set_radius(self, radius: float) -> ChangeList:
        return [(self.radius, ABS, _positive("radius", radius))]


class Triangle(RegularPolygon):
    """An equilateral triangle on a circle of ``radius``, a corner up."""

    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(3, radius, name, **style)


class Polyline(Shape2D):
    """Straight segments through ``corners`` (2D or 3D), closed into a
    polygon if asked. ``corners`` is a signal: moving corners animates
    them one by one (the count stays)."""

    def __init__(self, corners: Sequence[Sequence[float]], closed: bool = False, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.corners = parameter(self, "corners", _corner_array(corners), ARRAY)
        self.closed = parameter(self, "closed", bool(closed), DISCRETE)
        self._derive_points(_polyline_points, self.corners, self.closed)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.corners, self.closed)

    @animatable
    def set_corners(self, corners: Sequence[Sequence[float]]) -> ChangeList:
        array = _corner_array(corners)
        if array.shape != np.shape(self.corners.value):
            raise ValueError(f"{self.name} has {len(self.corners.value)} corners; set_corners keeps the count")
        return [(self.corners, ABS, array)]


def _square_points(side_length: float) -> np.ndarray:
    return rectangle_points(side_length, side_length)


def _polyline_points(corners: np.ndarray, closed: bool) -> np.ndarray:
    return corner_points(corners, closed=closed)


def _corners(n: int) -> int:
    if int(n) != n or n < 3:
        raise ValueError(f"a polygon needs at least 3 corners, got {n}")
    return int(n)


def _corner_array(corners: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(corners, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3) or len(array) < 2:
        raise ValueError(f"corners must be two or more 2D or 3D points, got shape {array.shape}")
    return array


__all__ = ["Circle", "Dot", "Polyline", "Rectangle", "RegularPolygon", "Square", "Triangle", "parameter"]
