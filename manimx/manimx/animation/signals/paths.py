"""Operations on point arrays of cubic Bezier curves that animations use
while they play: the part of a path drawn so far (:func:`partial_path`,
:class:`DrawnPart`), and points along a path (:func:`point_at`,
:class:`PointSignal`).

Points are laid out as in ``RBezierPath``: four per curve (anchor,
handle, handle, anchor), a new subpath wherever a curve does not start
where the previous one ended. The core computes the same in Rust
(core/src/paths.rs) with the same operations in the same order, so both
agree; keep them in step. Points along a path are Python only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import quaternion

from .kinds import ARRAY, PATH, SCALAR
from .signal import DerivedSignal, RootSignal, Signal

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


# --------------------------------------------------------------------------
# Points on a path
# --------------------------------------------------------------------------

LENGTH_SAMPLES = 16
"""Pieces per curve when measuring a path's length."""


def bezier_point(curve: np.ndarray, t: float) -> np.ndarray:
    """The point at parameter ``t`` of one cubic curve ``(4, D)``."""
    s = 1.0 - t
    return s**3 * curve[0] + 3 * s * s * t * curve[1] + 3 * s * t * t * curve[2] + t**3 * curve[3]


def point_at(points: np.ndarray, proportion: float, by_length: bool = True) -> np.ndarray:
    """The point a ``proportion`` (0 to 1) of the way along a path.

    With ``by_length`` the proportion is of the path's length, so equal
    steps give equally spaced points however the curves are spread. Else
    it is of its curves, as in :func:`partial_path`: each curve gets the
    same share, short or long. Jumps between subpaths do not count. An
    empty path gives the origin.
    """
    count = len(points) // POINTS_PER_CURVE
    if count == 0:
        return np.zeros(points.shape[1] if points.ndim == 2 else 2)
    curves = points.reshape(count, POINTS_PER_CURVE, -1)
    proportion = min(max(float(proportion), 0.0), 1.0)
    if not by_length:
        index, t = _curve_at(count, proportion)
        return bezier_point(curves[index], t)
    # The path as a polyline of LENGTH_SAMPLES pieces per curve.
    ts = np.linspace(0.0, 1.0, LENGTH_SAMPLES + 1)
    s = 1.0 - ts
    weights = np.stack([s**3, 3 * s * s * ts, 3 * s * ts * ts, ts**3], axis=1)  # (samples, 4)
    samples = np.einsum("sk,ckd->csd", weights, curves)  # (curves, samples, D)
    pieces = np.linalg.norm(np.diff(samples, axis=1), axis=2).reshape(-1)
    lengths = np.concatenate([[0.0], np.cumsum(pieces)])
    if lengths[-1] == 0.0:
        return curves[0, 0].copy()
    piece = int(np.clip(np.searchsorted(lengths, proportion * lengths[-1], side="right") - 1, 0, len(pieces) - 1))
    within = (proportion * lengths[-1] - lengths[piece]) / pieces[piece] if pieces[piece] > 0 else 0.0
    index, step = divmod(piece, LENGTH_SAMPLES)
    return bezier_point(curves[index], (step + min(within, 1.0)) / LENGTH_SAMPLES)


class PointAt:
    """A point on a node's path in its parent's space: the derivation of
    :meth:`Shape2D.point_at` and :meth:`Curve3D.point_at`, from the
    points, the proportion and the node's position, scale and rotation.
    Called back in Python (the core does not compute it)."""

    __slots__ = ("by_length",)

    def __init__(self, by_length: bool) -> None:
        self.by_length = by_length

    def __call__(self, points: np.ndarray, proportion: float, x: float, y: float, z: float, sx: float, sy: float, sz: float, rotation: Any) -> np.ndarray:
        local = np.zeros(3)
        point = point_at(points, proportion, self.by_length)
        local[: len(point)] = point
        moved = quaternion.rotate_vectors(rotation, local * (sx, sy, sz))
        return (moved + (x, y, z))[None]


class PointSignal:
    """A point that follows signals: ``x``, ``y`` and ``z`` derived from
    one ARRAY signal ``point`` (shape ``(1, 3)``). A node keeps to it with
    :meth:`~manimx.animation.nodes.node.PositionMixin.follow`."""

    __slots__ = ("point", "x", "y", "z")

    def __init__(self, point: Signal) -> None:
        self.point = point
        self.x, self.y, self.z = (DerivedSignal(_Component(axis), (point,), SCALAR, name=f"{point.name}.{'xyz'[axis]}") for axis in range(3))

    @property
    def axes(self) -> tuple[Signal, Signal, Signal]:
        return (self.x, self.y, self.z)

    @property
    def value(self) -> tuple[float, float, float]:
        return (self.x.value, self.y.value, self.z.value)


class _Component:
    __slots__ = ("axis",)

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def __call__(self, point: np.ndarray) -> float:
        return float(point[0, self.axis])


def point_signal(node: Any, proportion: float | Signal, by_length: bool) -> PointSignal:
    """The PointSignal of ``node.point_at`` (a Shape2D or Curve3D)."""
    if not isinstance(proportion, Signal):
        proportion = RootSignal(float(proportion), SCALAR, name=f"{node.name}.point_at.proportion")
    inputs = (node.points, proportion, *node.position.axes, *node.scale.axes, node.rotation)
    return PointSignal(DerivedSignal(PointAt(by_length), inputs, ARRAY, name=f"{node.name}.point_at"))
