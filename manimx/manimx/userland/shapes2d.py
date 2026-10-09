"""Shapes built on Shape2D: each one only computes initial points.

They are userland: a project could define its own shapes the same way,
by subclassing Shape2D and passing points from manimx.userland.geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..animation import Shape2D
from .geometry import arc_points, corner_points, rectangle_points, regular_polygon_points


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
        super().__init__(rectangle_points(width, height), name, **style)


class Square(Shape2D):
    def __init__(self, side_length: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(rectangle_points(side_length, side_length), name, **style)


class RegularPolygon(Shape2D):
    def __init__(self, n: int = 6, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(regular_polygon_points(n, radius), name, **style)


class Polyline(Shape2D):
    """Straight segments through ``corners`` (closed into a polygon if asked)."""

    def __init__(self, corners: Sequence[Sequence[float]], closed: bool = False, name: str | None = None, **style: Any) -> None:
        super().__init__(corner_points(corners, closed=closed), name, **style)


class Triangle(RegularPolygon):
    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(3, radius, name, **style)


__all__ = ["Circle", "Dot", "Polyline", "Rectangle", "RegularPolygon", "Square", "Triangle"]
