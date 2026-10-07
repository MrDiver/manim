"""Basic shapes, ported from ``manim.mobject.geometry`` point generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .shape2d import Shape2D


def arc_points(
    radius: float = 1.0,
    start_angle: float = 0.0,
    angle: float = math.tau,
    num_curves: int = 8,
) -> np.ndarray:
    """Cubic Bezier approximation of a circular arc (Manim's ``Arc``).

    Each piece spans ``angle / num_curves``; its handles lie on the tangents
    at distance ``4/3 * tan(theta / 4)`` times the radius.
    """
    angles = start_angle + np.linspace(0.0, angle, num_curves + 1)
    theta = angle / num_curves
    handle_length = 4 / 3 * math.tan(theta / 4) * radius

    anchors = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    tangents = np.column_stack([-np.sin(angles), np.cos(angles)])
    curves = np.stack(
        [
            anchors[:-1],
            anchors[:-1] + handle_length * tangents[:-1],
            anchors[1:] - handle_length * tangents[1:],
            anchors[1:],
        ],
        axis=1,
    )
    return curves.reshape(-1, 2)


def circle(id: str, radius: float = 1.0, **kwargs: Any) -> Shape2D:
    return Shape2D(id, points=arc_points(radius), **kwargs)


def regular_polygon(
    id: str, n: int, radius: float = 1.0, start_angle: float | None = None, **kwargs: Any
) -> Shape2D:
    """Manim's ``RegularPolygon``: odd ``n`` start with a vertex pointing up."""
    if start_angle is None:
        start_angle = (n % 2) * math.pi / 2
    angles = start_angle + np.arange(n) * math.tau / n
    corners = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return Shape2D.from_corners(id, corners, closed=True, **kwargs)


def triangle(id: str, **kwargs: Any) -> Shape2D:
    return regular_polygon(id, 3, **kwargs)


def rectangle(id: str, width: float = 4.0, height: float = 2.0, **kwargs: Any) -> Shape2D:
    """Corners in Manim's order: UR, UL, DL, DR."""
    w, h = width / 2, height / 2
    return Shape2D.from_corners(id, [(w, h), (-w, h), (-w, -h), (w, -h)], closed=True, **kwargs)


def square(id: str, side_length: float = 2.0, **kwargs: Any) -> Shape2D:
    return rectangle(id, side_length, side_length, **kwargs)
