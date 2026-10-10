"""Operations on point arrays of cubic Bezier curves that animations use
while they play: the part of a path drawn so far (:func:`partial_path`,
:class:`DrawnPart`).

Points are laid out as in ``RBezierPath``: four per curve (anchor,
handle, handle, anchor), a new subpath wherever a curve does not start
where the previous one ended. The core computes the same in Rust
(core/src/paths.rs) with the same operations in the same order, so both
agree; keep them in step.
"""

from __future__ import annotations

import numpy as np

from .kinds import PATH, SCALAR
from .signals import DerivedSignal, RootSignal, Signal

POINTS_PER_CURVE = 4


def split_curve(curve: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The curve ``(4, D)`` cut at parameter ``t`` (de Casteljau)."""
    p0, p1, p2, p3 = curve
    a = p0 + (p1 - p0) * t
    b = p1 + (p2 - p1) * t
    c = p2 + (p3 - p2) * t
    d = a + (b - a) * t
    e = b + (c - b) * t
    f = d + (e - d) * t
    return np.array([p0, a, d, f]), np.array([f, e, c, p3])


def partial_curve(curve: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """The part of one curve between parameters ``lower`` and ``upper``."""
    if lower >= 1.0:
        return np.repeat(curve[3:], POINTS_PER_CURVE, axis=0)
    _, rest = split_curve(curve, lower)
    part, _ = split_curve(rest, (upper - lower) / (1.0 - lower))
    return part


def partial_path(points: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """The part of a path from ``lower`` to ``upper`` (fractions of its
    curves), as Manim's ``pointwise_become_partial`` computes it: whole
    curves in between, the curves at both ends cut. Nothing is left when
    ``upper <= lower``, so a path not started yet draws no dot.
    """
    count = len(points) // POINTS_PER_CURVE
    if lower <= 0.0 and upper >= 1.0:
        return points
    if count == 0 or upper <= lower:
        return points[:0]
    curves = points.reshape(count, POINTS_PER_CURVE, -1)
    lower_index, lower_residue = _curve_at(count, lower)
    upper_index, upper_residue = _curve_at(count, upper)
    if lower_index == upper_index:
        return partial_curve(curves[lower_index], lower_residue, upper_residue)
    return np.concatenate(
        [
            partial_curve(curves[lower_index], lower_residue, 1.0),
            curves[lower_index + 1 : upper_index].reshape(-1, points.shape[1]),
            partial_curve(curves[upper_index], 0.0, upper_residue),
        ]
    )


def _curve_at(count: int, alpha: float) -> tuple[int, float]:
    """Manim's ``integer_interpolate(0, count, alpha)``: which curve the
    fraction ``alpha`` of a path lies on, and how far along it.
    """
    if alpha >= 1.0:
        return count - 1, 1.0
    if alpha <= 0.0:
        return 0, 0.0
    position = count * alpha
    return int(position), position % 1.0




def drawn_part(points: Signal, prefix: str) -> tuple[RootSignal, RootSignal, DerivedSignal]:
    """``draw_start``, ``draw_end`` (0 and 1) and the ``drawn_points``
    derived from them and ``points``, for the node named ``prefix``."""
    start = RootSignal(0.0, SCALAR, name=f"{prefix}.draw_start")
    end = RootSignal(1.0, SCALAR, name=f"{prefix}.draw_end")
    return start, end, DerivedSignal(DrawnPart(), (points, start, end), PATH, name=f"{prefix}.drawn_points")


class DrawnPart:
    """``partial_path(points, start, end)``: the function of the
    ``drawn_points`` signal of Shape2D and Curve3D, the part of their path
    from ``draw_start`` to ``draw_end``. A class instead of a function, so
    the core (manimx._core) recognizes it and computes it in Rust."""

    __slots__ = ()

    def __call__(self, points: np.ndarray, start: float, end: float) -> np.ndarray:
        if start <= 0.0 and end >= 1.0:
            return points
        return partial_path(np.asarray(points), start, end)


__all__ = ["DrawnPart", "drawn_part", "partial_curve", "partial_path", "split_curve"]
