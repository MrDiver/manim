"""Operations on point arrays of cubic Bezier curves that animations use
while they play: the part of a path drawn so far (:func:`partial_path`).

Points are laid out as in ``RBezierPath``: four per curve (anchor,
handle, handle, anchor), a new subpath wherever a curve does not start
where the previous one ended. The core computes the same in Rust
(core/src/paths.rs) with the same operations in the same order, so both
agree; keep them in step.
"""

from __future__ import annotations

import numpy as np

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


__all__ = ["partial_curve", "partial_path", "split_curve"]
