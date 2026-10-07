"""CPU tessellation of :class:`Shape2D` fills into triangles with exact curves.

The approach is the one ManimGL uses (Loop & Blinn, "Resolution Independent
Curve Rendering using Programmable Graphics Hardware", 2005):

1. Every cubic Bezier is split into quadratic Beziers with a bounded error.
2. Every subpath becomes a polygon *ring*. A quadratic that bulges *into* the
   filled region (concave) contributes its control point to the ring; one that
   bulges *out of* it (convex) contributes only its chord.
3. Rings are triangulated with earcut (outer rings together with their holes).
4. Each quadratic additionally emits one *curve triangle* (start, control,
   end). Its fragment shader evaluates the implicit curve ``u^2 - v`` and
   discards the half that lies outside the shape, which makes the edge exact
   at any zoom level:

       convex:  polygon stops at the chord, curve triangle adds the bulge
       concave: polygon goes around the control point, curve triangle adds
                the part between the curve and the control point

Everything is in the shape's local 2D coordinates; transforms are applied on
the GPU, so moving a shape never re-tessellates it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mapbox_earcut
import numpy as np

from ..objects.shape2d import POINTS_PER_CURVE, TOLERANCE_FOR_POINT_EQUALITY

DEFAULT_TOLERANCE = 1e-5
"""Max distance between a cubic and its quadratic approximation (local units)."""

MAX_QUADRATICS_PER_CUBIC = 32

# Implicit-curve coordinates of a quadratic's (start, control, end) vertices.
# With these, u^2 - v is < 0 between chord and curve and > 0 towards the
# control point.
_CURVE_UV = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]])

SIDE_SOLID = 0.0
"""Triangle is entirely inside the shape."""
SIDE_CONTROL = 1.0
"""Keep the control-point side of the curve (concave curve)."""
SIDE_CHORD = -1.0
"""Keep the chord side of the curve (convex curve)."""


@dataclass(frozen=True, slots=True, eq=False)
class FillMesh:
    """Non-indexed triangle list, three consecutive vertices per triangle."""

    positions: np.ndarray  # (M, 2) local coordinates
    uv: np.ndarray  # (M, 2) implicit-curve coordinates
    side: np.ndarray  # (M,) one of the SIDE_* constants
    colors: np.ndarray  # (M, 4) RGBA

    def __len__(self) -> int:
        return len(self.positions)

    @staticmethod
    def empty() -> FillMesh:
        return FillMesh(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros((0, 4)))


@dataclass(slots=True)
class _Quadratic:
    """One quadratic Bezier with the fill color at each control point."""

    start: np.ndarray
    control: np.ndarray
    end: np.ndarray
    start_color: np.ndarray
    control_color: np.ndarray
    end_color: np.ndarray

    def reversed(self) -> _Quadratic:
        return _Quadratic(
            self.end, self.control, self.start,
            self.end_color, self.control_color, self.start_color,
        )  # fmt: skip

    def bend(self) -> float:
        """Signed bend: > 0 when the control point is left of the chord."""
        chord = self.end - self.start
        cross = _cross(chord, self.control - self.start)
        length_sq = float(chord @ chord)
        return cross / length_sq if length_sq > 0.0 else 0.0

    def sample(self, t: float) -> np.ndarray:
        s = 1.0 - t
        return s * s * self.start + 2 * s * t * self.control + t * t * self.end


def tessellate_fill(
    points: np.ndarray, colors: np.ndarray, tolerance: float = DEFAULT_TOLERANCE
) -> FillMesh:
    """Triangulate a ``VMobject``-layout point array (see :class:`Shape2D`).

    ``colors`` holds one RGBA per point. Open subpaths are closed with a
    straight line, like Cairo does when filling.
    """
    curves = points.reshape(-1, POINTS_PER_CURVE, 2)
    curve_colors = colors.reshape(-1, POINTS_PER_CURVE, 4)
    rings = [
        ring
        for start, stop in _subpath_ranges(curves)
        if len(ring := _ring(curves[start:stop], curve_colors[start:stop], tolerance)) > 0
    ]
    if not rings:
        return FillMesh.empty()

    builder = _MeshBuilder()
    for outer, holes in _classify_rings(rings):
        builder.add_polygon(outer, holes)
    return builder.build()


# --------------------------------------------------------------------------
# Step 1: subpaths and cubic -> quadratic conversion
# --------------------------------------------------------------------------


def _subpath_ranges(curves: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` curve index ranges of continuous subpaths."""
    if len(curves) == 0:
        return []
    gaps = np.linalg.norm(curves[1:, 0] - curves[:-1, -1], axis=1)
    breaks = [0, *(np.flatnonzero(gaps > TOLERANCE_FOR_POINT_EQUALITY) + 1), len(curves)]
    return list(zip(breaks[:-1], breaks[1:]))


def _ring(curves: np.ndarray, colors: np.ndarray, tolerance: float) -> list[_Quadratic]:
    """Closed list of quadratics approximating one subpath."""
    ring = [
        quadratic
        for curve, curve_colors in zip(curves, colors)
        for quadratic in _cubic_to_quadratics(curve, curve_colors, tolerance)
    ]
    first, last = ring[0], ring[-1]
    if np.linalg.norm(last.end - first.start) > TOLERANCE_FOR_POINT_EQUALITY:
        ring.append(
            _Quadratic(
                last.end, (last.end + first.start) / 2, first.start,
                last.end_color, (last.end_color + first.start_color) / 2, first.start_color,
            )
        )  # fmt: skip
    return ring if abs(_signed_area(ring)) > 1e-12 else []


def _cubic_to_quadratics(
    curve: np.ndarray, colors: np.ndarray, tolerance: float
) -> list[_Quadratic]:
    """Split one cubic into quadratics whose error stays below ``tolerance``.

    A single quadratic with control ``(3(p1 + p2) - p0 - p3) / 4`` deviates by
    at most ``sqrt(3)/36 * |p3 - 3 p2 + 3 p1 - p0|``; splitting into ``n``
    pieces divides that by ``n^3``.
    """
    p0, p1, p2, p3 = curve
    error = math.sqrt(3) / 36 * float(np.linalg.norm(p3 - 3 * p2 + 3 * p1 - p0))
    count = min(MAX_QUADRATICS_PER_CUBIC, max(1, math.ceil((error / tolerance) ** (1 / 3))))

    quadratics = []
    for index in range(count):
        t0, t1 = index / count, (index + 1) / count
        # Sub-cubic on [t0, t1] from its endpoints and end tangents.
        q0, q3 = _bezier(curve, t0), _bezier(curve, t1)
        q1 = q0 + (t1 - t0) / 3 * _bezier_derivative(curve, t0)
        q2 = q3 - (t1 - t0) / 3 * _bezier_derivative(curve, t1)
        quadratics.append(
            _Quadratic(
                q0, (3 * (q1 + q2) - q0 - q3) / 4, q3,
                _bezier(colors, t0), _bezier(colors, (t0 + t1) / 2), _bezier(colors, t1),
            )
        )  # fmt: skip
    return quadratics


def _bezier(control: np.ndarray, t: float) -> np.ndarray:
    s = 1.0 - t
    weights = np.array([s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t])
    return weights @ control


def _bezier_derivative(control: np.ndarray, t: float) -> np.ndarray:
    s = 1.0 - t
    return 3 * (s * s * (control[1] - control[0]) + 2 * s * t * (control[2] - control[1]) + t * t * (control[3] - control[2]))


# --------------------------------------------------------------------------
# Step 2: outer rings, holes and orientation
# --------------------------------------------------------------------------


def _classify_rings(
    rings: list[list[_Quadratic]],
) -> list[tuple[list[_Quadratic], list[list[_Quadratic]]]]:
    """Group rings into ``(outer, holes)`` by nesting depth (even-odd rule).

    Rings are reoriented so outer rings run counter-clockwise and holes
    clockwise; then the filled region is always on the left of a ring.
    """
    outlines = [np.array([q.sample(t) for q in ring for t in (0.0, 0.25, 0.5, 0.75)]) for ring in rings]
    containers = [
        [j for j, outline in enumerate(outlines) if j != i and _contains(outline, ring[0].sample(0.5))]
        for i, ring in enumerate(rings)
    ]  # fmt: skip
    depth = [len(c) for c in containers]

    groups: dict[int, tuple[list[_Quadratic], list[list[_Quadratic]]]] = {}
    for i, ring in enumerate(rings):
        if depth[i] % 2 == 0:
            groups[i] = (_oriented(ring, counter_clockwise=True), [])
    for i, ring in enumerate(rings):
        if depth[i] % 2 == 1:
            # The innermost container of a hole is its outer ring.
            parent = max(containers[i], key=lambda j: depth[j])
            groups[parent][1].append(_oriented(ring, counter_clockwise=False))
    return list(groups.values())


def _oriented(ring: list[_Quadratic], counter_clockwise: bool) -> list[_Quadratic]:
    if (_signed_area(ring) > 0) == counter_clockwise:
        return ring
    return [q.reversed() for q in reversed(ring)]


def _signed_area(ring: list[_Quadratic]) -> float:
    """Exact signed area enclosed by a closed ring of quadratics (CCW > 0)."""
    area = 0.0
    for q in ring:
        # Shoelace term of the chord plus the parabolic segment, which has
        # 2/3 of the area of the triangle (start, control, end).
        area += _cross(q.start, q.end) / 2 + (2 / 3) * _cross(q.control - q.start, q.end - q.start) / 2
    return area


def _contains(outline: np.ndarray, point: np.ndarray) -> bool:
    """Even-odd point-in-polygon test."""
    x, y = point
    xs, ys = outline[:, 0], outline[:, 1]
    xs_next, ys_next = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (ys_next > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = xs + (y - ys) * (xs_next - xs) / (ys_next - ys)
    return bool(np.count_nonzero(crosses & (x < x_at_y)) % 2)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


# --------------------------------------------------------------------------
# Step 3 + 4: earcut polygons and curve triangles
# --------------------------------------------------------------------------

_BEND_EPSILON = 1e-9
"""Quadratics bending less than this (relative to chord length) are lines."""


class _MeshBuilder:
    def __init__(self) -> None:
        self._positions: list[np.ndarray] = []
        self._uv: list[np.ndarray] = []
        self._side: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []

    def add_polygon(self, outer: list[_Quadratic], holes: list[list[_Quadratic]]) -> None:
        vertices: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        ring_ends: list[int] = []
        for ring in (outer, *holes):
            for q in ring:
                vertices.append(q.start)
                colors.append(q.start_color)
                bend = q.bend()
                if bend > _BEND_EPSILON:  # concave: the ring walks around the control
                    vertices.append(q.control)
                    colors.append(q.control_color)
                    self._add_curve_triangle(q, SIDE_CONTROL)
                elif bend < -_BEND_EPSILON:  # convex: the curve triangle adds the bulge
                    self._add_curve_triangle(q, SIDE_CHORD)
            ring_ends.append(len(vertices))

        vertex_array = np.array(vertices)
        indices = mapbox_earcut.triangulate_float64(vertex_array, np.array(ring_ends, dtype=np.uint32))
        count = len(indices)
        self._positions.append(vertex_array[indices])
        self._colors.append(np.array(colors)[indices])
        self._uv.append(np.zeros((count, 2)))
        self._side.append(np.full(count, SIDE_SOLID))

    def _add_curve_triangle(self, q: _Quadratic, side: float) -> None:
        self._positions.append(np.array([q.start, q.control, q.end]))
        self._colors.append(np.array([q.start_color, q.control_color, q.end_color]))
        self._uv.append(_CURVE_UV)
        self._side.append(np.full(3, side))

    def build(self) -> FillMesh:
        if not self._positions:
            return FillMesh.empty()
        return FillMesh(
            positions=np.concatenate(self._positions),
            uv=np.concatenate(self._uv),
            side=np.concatenate(self._side),
            colors=np.concatenate(self._colors),
        )
