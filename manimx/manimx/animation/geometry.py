"""Shape definitions: the points of paths and the meshes of solids.

Everything that generates geometry lives here, on the scene side. The
shape nodes are built from it (Circle from arc_points(), Polyline from
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

from ..robjects import MeshData


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
    "arc_points",
    "box_mesh",
    "corner_points",
    "parametric_points",
    "rectangle_points",
    "regular_polygon_points",
    "sphere_mesh",
    "surface_mesh",
    "torus_mesh",
]
