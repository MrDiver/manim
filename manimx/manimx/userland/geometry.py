"""Shape definitions: the points of paths and the meshes of solids.

Everything that generates geometry lives here, in userland. The shape
classes of userland are built from it (Circle from arc_points(), Polyline from
corner_points(), ParametricCurve3D from parametric_points(), Box from
box_mesh(), ...), and code that builds render objects directly, such as
renderer tests, takes its points and meshes from here too. The render
side (manimx.robjects) only holds the results: point arrays in
RBezierPath objects and MeshData values.

Paths are point arrays in the RBezierPath layout: four points per cubic
Bezier curve ``(anchor, handle, handle, anchor)``; a new subpath starts
wherever a curve's start differs from the previous curve's end.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..animation.signals.paths import POINTS_PER_CURVE, partial_curve
from ..robjects import MeshData
from ..robjects.rbezier_path import TOLERANCE_FOR_POINT_EQUALITY


def arc_points(
    radius: float = 1.0,
    start_angle: float = 0.0,
    angle: float = math.tau,
    num_curves: int = 8,
) -> np.ndarray:
    """Cubic Bezier approximation of a circular arc.

    Each piece spans ``angle / num_curves``; its handles lie on the tangents
    at distance ``4/3 * tan(theta / 4)`` times the radius, which puts the
    curve's middle on the circle
    (https://spencermortensen.com/articles/bezier-circle/).
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


def corner_points(corners: Sequence[Sequence[float]], closed: bool = False) -> np.ndarray:
    """A polyline through ``corners`` (2D or 3D) as straight cubic curves,
    closed back to the first corner if asked."""
    anchors = np.asarray(corners, dtype=np.float64)
    if anchors.ndim != 2:
        raise ValueError(f"corners must be a list of 2D or 3D points, got shape {anchors.shape}")
    if closed and len(anchors) > 0:
        anchors = np.vstack([anchors, anchors[:1]])
    starts, ends = anchors[:-1], anchors[1:]
    curves = np.stack([starts, (2 * starts + ends) / 3, (starts + 2 * ends) / 3, ends], axis=1)
    return curves.reshape(-1, anchors.shape[1])


def brace_points(length: float, depth: float) -> np.ndarray:
    """A curly brace from ``(-length / 2, 0)`` to ``(length / 2, 0)`` with
    its tip at ``(0, -depth)``: quarter circles into a straight part on
    each side, and two more into the tip (six cubic curves). The circles
    shrink on a brace shorter than twice its depth."""
    half = length / 2
    r = min(depth / 2, length / 4)
    k = 4 / 3 * math.tan(math.pi / 8) * r  # the handle of a quarter circle
    d = depth / 2  # the height of the straight parts
    left = [
        [(-half, 0.0), (-half, -k), (-half + r - k, -d), (-half + r, -d)],
        [(-half + r, -d), (-half + r + (half - 2 * r) / 3, -d), (-r - (half - 2 * r) / 3, -d), (-r, -d)],
        [(-r, -d), (-r + k, -d), (0.0, -depth + k), (0.0, -depth)],
    ]
    curves = np.array(left, dtype=np.float64)
    right = curves[::-1, ::-1] * np.array([-1.0, 1.0])  # mirrored, in path order
    return np.concatenate([curves, right]).reshape(-1, 2)


def parametric_points(function: Callable[[float], Any], t_range: tuple[float, float], samples: int = 64) -> np.ndarray:
    """A smooth 3D curve through ``function(t)`` at ``samples`` evenly
    spaced ``t``.

    Handles come from the derivative estimated by central differences,
    which makes the curve C1-continuous (like a Catmull-Rom spline).
    """
    ts = np.linspace(t_range[0], t_range[1], samples + 1)
    anchors = np.array([function(t) for t in ts], dtype=np.float64)
    step = ts[1] - ts[0]
    derivative = np.gradient(anchors, step, axis=0)
    handles = derivative * step / 3
    curves = np.stack([anchors[:-1], anchors[:-1] + handles[:-1], anchors[1:] - handles[1:], anchors[1:]], axis=1)
    return curves.reshape(-1, 3)


def regular_polygon_points(n: int, radius: float = 1.0, start_angle: float | None = None) -> np.ndarray:
    """A closed regular polygon; odd ``n`` start with a vertex pointing up."""
    if start_angle is None:
        start_angle = (n % 2) * math.pi / 2
    angles = start_angle + np.arange(n) * math.tau / n
    corners = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return corner_points(corners, closed=True)


def rectangle_points(width: float = 4.0, height: float = 2.0) -> np.ndarray:
    """A closed rectangle centered at the origin, corners in the order
    upper right, upper left, lower left, lower right."""
    w, h = width / 2, height / 2
    return corner_points([(w, h), (-w, h), (-w, -h), (w, -h)], closed=True)


def ellipse_points(width: float = 2.0, height: float = 1.0) -> np.ndarray:
    """A closed ellipse ``width`` by ``height`` centered at the origin,
    starting on the right."""
    return arc_points() * np.array([width / 2, height / 2])


def sector_points(radius: float = 1.0, start_angle: float = 0.0, angle: float = math.pi / 2) -> np.ndarray:
    """A closed wedge of a circle: from the center out along
    ``start_angle``, the arc of ``angle`` and back."""
    return annular_sector_points(0.0, radius, start_angle, angle)


def annular_sector_points(inner_radius: float = 0.5, outer_radius: float = 1.0, start_angle: float = 0.0, angle: float = math.pi / 2) -> np.ndarray:
    """A closed piece of a ring: the outer arc, a straight side in, the
    inner arc back and a side out. With ``inner_radius`` 0 it is a sector."""
    outer = arc_points(outer_radius, start_angle, angle, _arc_curves(angle))
    if inner_radius > 0:
        inner = arc_points(inner_radius, start_angle + angle, -angle, _arc_curves(angle))
    else:
        inner = corner_points([(0.0, 0.0), (0.0, 0.0)])  # the center, one zero-length curve
    return np.concatenate([outer, corner_points([outer[-1], inner[0]]), inner, corner_points([inner[-1], outer[0]])])


def annulus_points(inner_radius: float = 0.5, outer_radius: float = 1.0) -> np.ndarray:
    """A ring: the outer circle counterclockwise and the inner one
    clockwise, as two subpaths (the inner one is the hole)."""
    return np.concatenate([arc_points(outer_radius), arc_points(inner_radius, 0.0, -math.tau)])


def star_points(n: int = 5, outer_radius: float = 1.0, inner_radius: float | None = None, start_angle: float = math.pi / 2) -> np.ndarray:
    """A closed star with ``n`` points on a circle of ``outer_radius`` (the
    first at ``start_angle``) and corners in between on ``inner_radius``
    (default: the regular star's, where its edges would cross)."""
    if inner_radius is None:
        inner_radius = outer_radius * math.cos(math.tau / n) / math.cos(math.pi / n) if n > 4 else outer_radius / 2
    angles = start_angle + np.arange(2 * n) * math.pi / n
    radii = np.where(np.arange(2 * n) % 2 == 0, outer_radius, inner_radius)
    return corner_points(radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)]), closed=True)


def rounded_rectangle_points(width: float = 4.0, height: float = 2.0, corner_radius: float = 0.5) -> np.ndarray:
    """A closed rectangle with quarter circles of ``corner_radius`` at
    its corners (at most half the shorter side), starting on the right."""
    r = min(corner_radius, width / 2, height / 2)
    w, h = width / 2 - r, height / 2 - r
    pieces = []
    for index, (x, y) in enumerate([(w, h), (-w, h), (-w, -h), (w, -h)]):
        corner = arc_points(r, index * math.pi / 2, math.pi / 2, 1) + [x, y]
        if pieces:
            pieces.append(corner_points([pieces[-1][-1], corner[0]]))
        pieces.append(corner)
    pieces.append(corner_points([pieces[-1][-1], pieces[0][0]]))
    return np.concatenate(pieces)


def arc_between_points(start: Sequence[float], end: Sequence[float], angle: float = math.pi / 2) -> np.ndarray:
    """The arc from ``start`` to ``end`` turning by ``angle`` (counter-
    clockwise if positive), in the plane z of ``start``. A straight line
    for an angle of 0."""
    a, b = np.asarray(start, dtype=np.float64)[:2], np.asarray(end, dtype=np.float64)[:2]
    chord = b - a
    length = float(np.linalg.norm(chord))
    if abs(angle) < 1e-9 or length == 0.0:
        return corner_points([a, b])
    radius = length / (2 * math.sin(abs(angle) / 2))
    normal = np.array([-chord[1], chord[0]]) / length  # left of the chord
    center = (a + b) / 2 + normal * math.copysign(radius * math.cos(angle / 2), angle)
    start_angle = math.atan2(*(a - center)[::-1])
    return arc_points(radius, start_angle, angle, _arc_curves(angle)) + center


def tip_points(end: Sequence[float], direction: Sequence[float], length: float, width: float) -> np.ndarray:
    """A closed triangle with its point at ``end``, pointing along
    ``direction``, ``length`` long and ``width`` wide at its base."""
    tip = np.asarray(end, dtype=np.float64)[:2]
    along = np.asarray(direction, dtype=np.float64)[:2]
    along = along / (np.linalg.norm(along) or 1.0)
    across = np.array([-along[1], along[0]]) * width / 2
    base = tip - along * length
    return corner_points([tip, base + across, base - across], closed=True)


def dashes_points(start: Sequence[float], end: Sequence[float], dash_length: float, dashed_ratio: float = 0.5) -> np.ndarray:
    """The dashes of a straight line, one subpath each: ``dash_length``
    long at most with gaps between, ``dashed_ratio`` of the length drawn,
    and a whole number of dashes centered on the line."""
    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    count = max(1, round(length * dashed_ratio / dash_length)) if dash_length > 0 else 1
    period = length / count
    pieces = [
        corner_points([a + (b - a) * ((i * period + (1 - dashed_ratio) * period / 2) / length), a + (b - a) * ((i * period + (1 + dashed_ratio) * period / 2) / length)])
        for i in range(count)
    ] if length > 0 else [corner_points([a, b])]
    return np.concatenate(pieces)


def _arc_curves(angle: float) -> int:
    """Curves for an arc of ``angle``: one per eighth of a turn, at least one."""
    return max(1, math.ceil(abs(angle) / (math.tau / 8) - 1e-9))


# --------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------


def box_mesh(width: float = 2.0, height: float = 2.0, depth: float = 2.0) -> MeshData:
    """Axis-aligned box centered at the origin with flat faces and UVs."""
    half = np.array([width, height, depth]) / 2
    positions, normals, uvs, indices = [], [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            # Cyclic order makes cross(u, v) point along +axis.
            u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
            if sign < 0:
                u_axis, v_axis = v_axis, u_axis  # keep counter-clockwise winding
            base = len(positions)
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = normal * half
                point[u_axis] = (2 * u - 1) * half[u_axis]
                point[v_axis] = (2 * v - 1) * half[v_axis]
                positions.append(point)
                normals.append(normal)
                uvs.append((u, 1 - v))
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return MeshData(np.array(positions), np.array(indices), np.array(normals), np.array(uvs))


def sphere_mesh(radius: float = 1.0, segments: int = 48, rings: int = 24) -> MeshData:
    """UV sphere around the z axis."""
    theta = np.linspace(0.0, math.tau, segments + 1)  # around z
    phi = np.linspace(0.0, math.pi, rings + 1)  # from +z to -z
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
    normals = np.stack(
        [np.sin(phi_grid) * np.cos(theta_grid), np.sin(phi_grid) * np.sin(theta_grid), np.cos(phi_grid)], axis=-1
    ).reshape(-1, 3)
    uvs = np.stack([theta_grid / math.tau, phi_grid / math.pi], axis=-1).reshape(-1, 2)
    return MeshData(normals * radius, _grid_indices(rings, segments), normals, uvs)


def torus_mesh(major_radius: float = 1.0, minor_radius: float = 0.35, segments: int = 64, sides: int = 32) -> MeshData:
    """Torus around the z axis."""
    u = np.linspace(0.0, math.tau, segments + 1)
    v = np.linspace(0.0, math.tau, sides + 1)
    v_grid, u_grid = np.meshgrid(v, u, indexing="ij")
    center = np.stack([np.cos(u_grid), np.sin(u_grid), np.zeros_like(u_grid)], axis=-1)
    normals = np.cos(v_grid)[..., None] * center + np.sin(v_grid)[..., None] * np.array([0.0, 0.0, 1.0])
    positions = major_radius * center + minor_radius * normals
    uvs = np.stack([u_grid / math.tau, v_grid / math.tau], axis=-1)
    return MeshData(positions.reshape(-1, 3), _grid_indices(sides, segments, flip=True), normals.reshape(-1, 3), uvs.reshape(-1, 2))


def surface_mesh(
    function: Callable[[float, float], Any],
    u_range: tuple[float, float] = (0.0, 1.0),
    v_range: tuple[float, float] = (0.0, 1.0),
    resolution: tuple[int, int] = (32, 32),
) -> MeshData:
    """Parametric surface through ``function(u, v)`` (a 3D point), sampled
    on a grid of ``resolution`` cells in u and v, with smooth normals and
    UVs from the normalized parameters.

    Surfaces of the same resolution share their triangles, so their
    vertices correspond one to one.
    """
    columns, rows = resolution
    if columns < 1 or rows < 1:
        raise ValueError(f"a surface needs at least one cell per direction, got {resolution}")
    u = np.linspace(u_range[0], u_range[1], columns + 1)
    v = np.linspace(v_range[0], v_range[1], rows + 1)
    positions = np.array([function(float(a), float(b)) for b in v for a in u], dtype=np.float64)
    if positions.shape != ((columns + 1) * (rows + 1), 3):
        raise ValueError("the surface function must return a 3D point")
    v_grid, u_grid = np.meshgrid(np.linspace(0.0, 1.0, rows + 1), np.linspace(0.0, 1.0, columns + 1), indexing="ij")
    uvs = np.stack([u_grid, v_grid], axis=-1).reshape(-1, 2)
    return MeshData(positions, _grid_indices(rows, columns, flip=True), uvs=uvs)


def _grid_indices(rows: int, columns: int, flip: bool = False) -> np.ndarray:
    """Two counter-clockwise triangles per cell of a (rows+1) x (columns+1) vertex grid."""
    row, column = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    a = row * (columns + 1) + column
    b, c, d = a + 1, a + columns + 1, a + columns + 2
    quads = np.stack([a, c, b, b, c, d], axis=-1) if not flip else np.stack([a, b, c, b, d, c], axis=-1)
    return quads.reshape(-1)


__all__ = [
    "annular_sector_points",
    "annulus_points",
    "arc_between_points",
    "arc_points",
    "brace_points",
    "box_mesh",
    "corner_points",
    "dashes_points",
    "ellipse_points",
    "parametric_points",
    "rectangle_points",
    "regular_polygon_points",
    "rounded_rectangle_points",
    "sector_points",
    "sphere_mesh",
    "star_points",
    "tip_points",
    "surface_mesh",
    "torus_mesh",
]


# -- aligning paths (Morph) ---------------------------------------------------------


def subpaths(points: np.ndarray) -> list[np.ndarray]:
    """``points`` split into continuous subpaths (same layout): a new one
    wherever a curve does not start where the previous one ended.
    """
    curves = points.reshape(-1, POINTS_PER_CURVE, points.shape[1])
    if len(curves) == 0:
        return []
    gaps = np.linalg.norm(curves[1:, 0] - curves[:-1, -1], axis=1)
    breaks = np.flatnonzero(gaps > TOLERANCE_FOR_POINT_EQUALITY) + 1
    return [part.reshape(-1, points.shape[1]) for part in np.split(curves, breaks)]


def insert_curves(points: np.ndarray, count: int) -> np.ndarray:
    """The same path with ``count`` more curves: every curve is cut into
    equal pieces, the extra pieces spread evenly over the path (Manim's
    ``insert_n_curves_to_point_list``).
    """
    if count <= 0:
        return points
    curves = points.reshape(-1, POINTS_PER_CURVE, points.shape[1])
    current = len(curves)
    target = current + count
    pieces_per_curve = np.bincount((np.arange(target) * current) // target, minlength=current)
    pieces = []
    for curve, pieces_here in zip(curves, pieces_per_curve, strict=True):
        cuts = np.linspace(0.0, 1.0, pieces_here + 1)
        pieces += [partial_curve(curve, lower, upper) for lower, upper in zip(cuts[:-1], cuts[1:], strict=True)]
    return np.concatenate(pieces)


def align_paths(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both paths with the same number of subpaths and of curves in each,
    so they interpolate point by point (Manim's ``align_points``). A path
    with fewer subpaths gets empty ones (four times its last point); in
    each pair of subpaths the shorter one gets more curves. Where both
    subpaths are closed, the second starts at the curve that keeps the
    pair closest (a circle into a square does not turn on the way). An
    empty path becomes a point at the other's center.
    """
    first, second = _not_empty(first, second), _not_empty(second, first)
    parts_1, parts_2 = subpaths(first), subpaths(second)
    aligned_1, aligned_2 = [], []
    for index in range(max(len(parts_1), len(parts_2))):
        part_1 = parts_1[index] if index < len(parts_1) else np.repeat(parts_1[-1][-1:], POINTS_PER_CURVE, axis=0)
        part_2 = parts_2[index] if index < len(parts_2) else np.repeat(parts_2[-1][-1:], POINTS_PER_CURVE, axis=0)
        difference = (len(part_2) - len(part_1)) // POINTS_PER_CURVE
        part_1, part_2 = insert_curves(part_1, difference), insert_curves(part_2, -difference)
        if _closed(part_1) and _closed(part_2):
            part_2 = _closest_start(part_2, part_1)
        aligned_1.append(part_1)
        aligned_2.append(part_2)
    return np.concatenate(aligned_1), np.concatenate(aligned_2)


def _closed(points: np.ndarray) -> bool:
    return len(points) > POINTS_PER_CURVE and bool(np.linalg.norm(points[0] - points[-1]) <= TOLERANCE_FOR_POINT_EQUALITY)


def _closest_start(points: np.ndarray, other: np.ndarray) -> np.ndarray:
    """The closed path ``points`` starting at the curve whose start gives
    the least squared distance between its anchors and ``other``'s (same
    curve count)."""
    curves = points.reshape(-1, POINTS_PER_CURVE, points.shape[1])
    anchors, targets = curves[:, 0], other.reshape(curves.shape)[:, 0]
    costs = [np.sum((np.roll(anchors, -shift, axis=0) - targets) ** 2) for shift in range(len(curves))]
    return np.roll(curves, -int(np.argmin(costs)), axis=0).reshape(points.shape)


def _not_empty(points: np.ndarray, other: np.ndarray) -> np.ndarray:
    if len(points):
        return points
    center = other.mean(axis=0) if len(other) else np.zeros(points.shape[1])
    return np.repeat(center[None], POINTS_PER_CURVE, axis=0)
