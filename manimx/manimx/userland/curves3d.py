"""Curves built on Curve3D (userland, like manimx.userland.shapes2d)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..animation import Curve3D
from .geometry import parametric_points


class ParametricCurve3D(Curve3D):
    """A smooth curve through ``function(t)`` (a 3D point) at ``samples``
    evenly spaced ``t`` in ``t_range`` (see geometry.parametric_points)."""

    def __init__(
        self,
        function: Callable[[float], Any],
        t_range: tuple[float, float] = (0.0, 1.0),
        samples: int = 64,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(parametric_points(function, t_range, samples), name, **style)


__all__ = ["ParametricCurve3D"]
