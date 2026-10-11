"""Solids built on Shape3D and lines built on Curve3D: each one only
computes its initial mesh or points with manimx.userland.geometry
(userland, like manimx.userland.shapes2d). Round solids stand along the
z axis, centered at the origin."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..animation import Curve3D, Group, Shape3D
from ..animation.nodes.node import ABS, ChangeList, animatable
from .geometry import (
    box_mesh,
    cone_mesh,
    cylinder_mesh,
    prism_mesh,
    sphere_mesh,
    surface_mesh,
    torus_mesh,
)


class Box(Shape3D):
    """An axis-aligned box centered at the origin."""

    def __init__(self, width: float = 2.0, height: float = 2.0, depth: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(box_mesh(width, height, depth), name, **style)


class Cube(Box):
    """A cube with sides ``side_length`` long, centered at the origin."""

    def __init__(self, side_length: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(side_length, side_length, side_length, name, **style)


class Prism(Shape3D):
    """A right prism ``height`` tall over a regular ``n``-gon of
    circumradius ``radius`` (a corner on +x)."""

    def __init__(self, n: int = 6, radius: float = 1.0, height: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(prism_mesh(n, radius, height), name, **style)


class Cylinder(Shape3D):
    """A cylinder of ``radius`` and ``height``, closed at both ends unless
    ``caps`` is false."""

    def __init__(self, radius: float = 1.0, height: float = 2.0, segments: int = 48, caps: bool = True, name: str | None = None, **style: Any) -> None:
        super().__init__(cylinder_mesh(radius, height, segments, caps), name, **style)


class Cone(Shape3D):
    """A cone with a base of ``radius`` at ``z = -height / 2`` (closed
    unless ``cap`` is false) and its apex at ``z = height / 2``."""

    def __init__(self, radius: float = 1.0, height: float = 2.0, segments: int = 48, cap: bool = True, name: str | None = None, **style: Any) -> None:
        super().__init__(cone_mesh(radius, height, segments, cap), name, **style)


class Sphere(Shape3D):
    """A UV sphere around the z axis."""

    def __init__(self, radius: float = 1.0, segments: int = 48, rings: int = 24, name: str | None = None, **style: Any) -> None:
        super().__init__(sphere_mesh(radius, segments, rings), name, **style)


class Dot3D(Sphere):
    """A small sphere to mark a point: ``Dot3D().move_to(point)``."""

    def __init__(self, radius: float = 0.08, name: str | None = None, **style: Any) -> None:
        super().__init__(radius, 16, 8, name, **style)


class Torus(Shape3D):
    """A torus around the z axis."""

    def __init__(
        self,
        major_radius: float = 1.0,
        minor_radius: float = 0.35,
        segments: int = 64,
        sides: int = 32,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(torus_mesh(major_radius, minor_radius, segments, sides), name, **style)


class ParametricSurface(Shape3D):
    """The surface through ``function(u, v)`` (a 3D point), sampled on a
    grid of ``resolution`` cells (see geometry.surface_mesh). Surfaces
    of the same resolution have corresponding vertices."""

    def __init__(
        self,
        function: Callable[[float, float], Any],
        u_range: tuple[float, float] = (0.0, 1.0),
        v_range: tuple[float, float] = (0.0, 1.0),
        resolution: tuple[int, int] = (32, 32),
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(surface_mesh(function, u_range, v_range, resolution), name, **style)


class Line3D(Curve3D):
    """A straight line from ``start`` to ``end`` in space; ``set_ends``
    moves its ends (animatable)."""

    def __init__(self, start: Sequence[float] = (0.0, 0.0, 0.0), end: Sequence[float] = (1.0, 0.0, 0.0), name: str | None = None, **style: Any) -> None:
        super().__init__(_segment(start, end), name, **style)

    @animatable
    def set_ends(self, start: Sequence[float], end: Sequence[float]) -> ChangeList:
        return [(self.points, ABS, _segment(start, end))]


class Arrow3D(Group):
    """An arrow from ``start`` to ``end`` in space: a ``line`` (Line3D,
    ending where the tip starts) and a ``tip`` (Cone ``tip_length`` long,
    ``tip_radius`` wide) pointing at ``end``. ``color`` colors both;
    ``style`` goes to the line and ``tip_style`` to the tip."""

    def __init__(
        self,
        start: Sequence[float] = (0.0, 0.0, 0.0),
        end: Sequence[float] = (1.0, 0.0, 0.0),
        tip_length: float = 0.3,
        tip_radius: float = 0.1,
        color: Any = None,
        tip_style: dict[str, Any] | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        start_point, end_point = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        direction = end_point - start_point
        length = float(np.linalg.norm(direction))
        if length <= tip_length:
            raise ValueError(f"an Arrow3D {length:.3g} long has no room for a tip {tip_length} long")
        unit = direction / length
        colored = {} if color is None else {"color": color}
        self.line = Line3D(start_point, end_point - unit * tip_length, name=f"{self.name}.line", **(colored | style))
        self.tip = Cone(tip_radius, tip_length, 24, name=f"{self.name}.tip", **(colored | (tip_style or {})))
        axis = np.cross((0.0, 0.0, 1.0), unit)
        if np.linalg.norm(axis) > 1e-9:
            self.tip.rotate(math.acos(float(np.clip(unit[2], -1.0, 1.0))), axis / np.linalg.norm(axis))
        elif unit[2] < 0:
            self.tip.rotate(math.pi, (1.0, 0.0, 0.0))
        self.tip.move_to(end_point - unit * tip_length / 2)
        self.add(self.line, self.tip)


def _segment(start: Sequence[float], end: Sequence[float]) -> np.ndarray:
    """One straight cubic curve from ``start`` to ``end``."""
    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    return np.array([a, (2 * a + b) / 3, (a + 2 * b) / 3, b])


__all__ = ["Arrow3D", "Box", "Cone", "Cube", "Cylinder", "Dot3D", "Line3D", "ParametricSurface", "Prism", "Sphere", "Torus"]
