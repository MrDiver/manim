"""Solids built on Shape3D: each one only computes its initial mesh with
manimx.userland.geometry (userland, like manimx.userland.shapes2d)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..animation import Shape3D
from .geometry import box_mesh, sphere_mesh, surface_mesh, torus_mesh


class Box(Shape3D):
    """An axis-aligned box centered at the origin."""

    def __init__(self, width: float = 2.0, height: float = 2.0, depth: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(box_mesh(width, height, depth), name, **style)


class Sphere(Shape3D):
    """A UV sphere around the z axis."""

    def __init__(self, radius: float = 1.0, segments: int = 48, rings: int = 24, name: str | None = None, **style: Any) -> None:
        super().__init__(sphere_mesh(radius, segments, rings), name, **style)


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


__all__ = ["Box", "ParametricSurface", "Sphere", "Torus"]
