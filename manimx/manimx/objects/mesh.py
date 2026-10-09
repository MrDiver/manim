"""Triangle meshes for :class:`RShape3D`, and generators for basic solids."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


def _frozen(array: np.ndarray, dtype: type, columns: int | None) -> np.ndarray:
    result = np.array(array, dtype=dtype)
    if columns is not None and (result.ndim != 2 or result.shape[1] != columns):
        raise ValueError(f"expected shape (N, {columns}), got {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class MeshData:
    """Indexed triangle mesh in local coordinates.

    ``normals`` are computed from the triangles when omitted (smooth shading
    over shared vertices). ``tangents`` (xyz plus handedness in w, as in
    glTF) are only needed for normal maps and are derived from ``uvs`` on
    demand, see :meth:`with_tangents`.
    """

    positions: np.ndarray  # (N, 3)
    indices: np.ndarray  # (M,) three per triangle
    normals: np.ndarray | None = None  # (N, 3)
    uvs: np.ndarray | None = None  # (N, 2)
    tangents: np.ndarray | None = None  # (N, 4)

    def __post_init__(self) -> None:
        positions = _frozen(self.positions, np.float32, 3)
        indices = _frozen(np.asarray(self.indices).reshape(-1), np.uint32, None)
        if len(indices) % 3 != 0:
            raise ValueError(f"indices must form triangles, got {len(indices)} indices")
        if len(indices) and int(indices.max()) >= len(positions):
            raise ValueError("an index points past the last vertex")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "indices", indices)
        normals = self.normals if self.normals is not None else _smooth_normals(positions, indices)
        object.__setattr__(self, "normals", _frozen(normals, np.float32, 3))
        if self.uvs is not None:
            object.__setattr__(self, "uvs", _frozen(self.uvs, np.float32, 2))
        if self.tangents is not None:
            object.__setattr__(self, "tangents", _frozen(self.tangents, np.float32, 4))

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    def with_tangents(self) -> MeshData:
        """This mesh with tangents computed from UVs (if it has UVs but no tangents)."""
        if self.tangents is not None or self.uvs is None:
            return self
        return MeshData(
            self.positions, self.indices, self.normals, self.uvs,
            _tangents(self.positions, self.normals, self.uvs, self.indices),
        )  # fmt: skip


def _smooth_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted average of the face normals around each vertex."""
    normals = np.zeros_like(positions, dtype=np.float64)
    triangles = positions[indices.reshape(-1, 3)].astype(np.float64)
    faces = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, indices.reshape(-1, 3)[:, corner], faces)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.tile([0.0, 0.0, 1.0], (len(normals), 1)), where=length > 0)


def _tangents(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Per-vertex tangents from UV gradients (Lengyel's method)."""
    triangles = indices.reshape(-1, 3)
    p = positions[triangles].astype(np.float64)
    t = uvs[triangles].astype(np.float64)
    edge1, edge2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    duv1, duv2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    determinant = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    scale = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=np.abs(determinant) > 1e-12)
    face_tangents = (edge1 * duv2[:, 1:2] - edge2 * duv1[:, 1:2]) * scale[:, None]
    face_bitangents = (edge2 * duv1[:, 0:1] - edge1 * duv2[:, 0:1]) * scale[:, None]

    tangents = np.zeros((len(positions), 3))
    bitangents = np.zeros((len(positions), 3))
    for corner in range(3):
        np.add.at(tangents, triangles[:, corner], face_tangents)
        np.add.at(bitangents, triangles[:, corner], face_bitangents)

    # Gram-Schmidt against the normal; w stores the bitangent's handedness.
    n = normals.astype(np.float64)
    tangents -= n * np.sum(n * tangents, axis=1, keepdims=True)
    length = np.linalg.norm(tangents, axis=1, keepdims=True)
    fallback = np.cross(n, [0.0, 0.0, 1.0]) + np.cross(n, [0.0, 1.0, 0.0]) * 1e-3
    tangents = np.where(length > 1e-12, tangents / np.maximum(length, 1e-12), fallback)
    handedness = np.where(np.sum(np.cross(n, tangents) * bitangents, axis=1) < 0.0, -1.0, 1.0)
    return np.column_stack([tangents, handedness])


# --------------------------------------------------------------------------
# Generators
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
