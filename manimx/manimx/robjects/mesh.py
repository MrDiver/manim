"""Triangle meshes for :class:`RShape3D` (the MeshData value type).

Generators for solids (box, sphere, torus, parametric surfaces) are
shape definitions and live on the scene side, in manimx.animation.geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """Per-vertex tangents from UV gradients (Eric Lengyel, "Computing
    Tangent Space Basis Vectors for an Arbitrary Mesh",
    https://terathon.com/code/tangent.html)."""
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
