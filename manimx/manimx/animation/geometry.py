"""Points of basic shapes, in the RBezierPath layout (four per cubic curve).

The shape nodes (Circle, Rectangle, RegularPolygon, ...) are built from
these, and code that builds render objects directly can use them too:
``RShape2D(points=arc_points(1.0))``.
"""

from __future__ import annotations

import math

import numpy as np

from ..robjects import RShape2D


def arc_points(
    radius: float = 1.0,
    start_angle: float = 0.0,
    angle: float = math.tau,
    num_curves: int = 8,
) -> np.ndarray:
    """Cubic Bezier approximation of a circular arc.

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


def regular_polygon_points(n: int, radius: float = 1.0, start_angle: float | None = None) -> np.ndarray:
    """A closed regular polygon; odd ``n`` start with a vertex pointing up."""
    if start_angle is None:
        start_angle = (n % 2) * math.pi / 2
    angles = start_angle + np.arange(n) * math.tau / n
    corners = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return RShape2D.from_corners(corners, closed=True).points


def rectangle_points(width: float = 4.0, height: float = 2.0) -> np.ndarray:
    """A closed rectangle centered at the origin, corners in the order
    upper right, upper left, lower left, lower right."""
    w, h = width / 2, height / 2
    return RShape2D.from_corners([(w, h), (-w, h), (-w, -h), (w, -h)], closed=True).points


__all__ = ["arc_points", "rectangle_points", "regular_polygon_points"]
