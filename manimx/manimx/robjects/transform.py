"""Transform: where a node sits, as position, rotation (a unit
quaternion) and scale, and the 4x4 matrix they make
(``T @ R @ S``: scale first, then rotate, then move)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TypeAlias

import numpy as np
import quaternion

Quaternion: TypeAlias = quaternion.quaternion
Vector3: TypeAlias = tuple[float, float, float]


def _vector3(value: object, name: str) -> Vector3:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have three components, got shape {array.shape}")
    return (float(array[0]), float(array[1]), float(array[2]))


def _normalized(rotation: Quaternion) -> Quaternion:
    norm = abs(rotation)
    if norm == 0.0:
        raise ValueError("rotation quaternion must not be zero")
    return rotation / norm


def quaternion_from_axis_angle(axis: object, angle: float) -> Quaternion:
    """Unit quaternion rotating ``angle`` radians around ``axis``."""
    vector = np.asarray(axis, dtype=np.float64)
    length = np.linalg.norm(vector)
    if vector.shape != (3,) or length == 0.0:
        raise ValueError("rotation axis must be a non-zero 3D vector")
    return quaternion.from_rotation_vector(vector / length * angle)


@dataclass(frozen=True, slots=True)
class Transform:
    """Local placement of a render object in 3D space.

    Applied to a point as ``position + rotation * (scale * point)``. The
    rotation is stored as a unit quaternion so the animation engine can
    interpolate it with :meth:`interpolate` (slerp) instead of interpolating
    Euler angles or matrices.
    """

    position: Vector3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = field(default_factory=lambda: quaternion.one)
    scale: Vector3 = (1.0, 1.0, 1.0)
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "scale", _vector3(self.scale, "scale"))
        object.__setattr__(self, "rotation", _normalized(self.rotation))

    def matrix(self) -> np.ndarray:
        """Homogeneous 4x4 matrix computing ``T @ R @ S`` (read-only).

        A transform never changes, so the matrix is computed once and
        kept: snapshots of an unchanged node share their transform, and
        every render asks for its matrix again.
        """
        result = self._matrix
        if result is None:
            result = np.eye(4)
            result[:3, :3] = quaternion.as_rotation_matrix(self.rotation) * np.asarray(self.scale)
            result[:3, 3] = self.position
            result.setflags(write=False)
            object.__setattr__(self, "_matrix", result)
        return result

    def apply(self, points: object) -> np.ndarray:
        """Transform an ``(N, 3)`` array of points."""
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return apply_matrix(self.matrix(), array)

    def translated(self, offset: object) -> Transform:
        return replace(
            self,
            position=tuple(np.asarray(self.position) + _vector3(offset, "offset")),
        )

    def rotated(self, angle: float, axis: object = (0.0, 0.0, 1.0)) -> Transform:
        """Rotate by ``angle`` radians around ``axis`` after the current rotation."""
        return replace(
            self, rotation=quaternion_from_axis_angle(axis, angle) * self.rotation
        )

    def scaled(self, factor: float | object) -> Transform:
        factors = np.broadcast_to(np.asarray(factor, dtype=np.float64), (3,))
        return replace(self, scale=tuple(np.asarray(self.scale) * factors))


def rotation_towards(direction: tuple[float, float, float]) -> Quaternion:
    """Rotation turning local ``-z`` onto ``direction`` (shortest arc)."""
    target = np.asarray(direction, dtype=np.float64)
    length = np.linalg.norm(target)
    if length == 0.0:
        raise ValueError("direction must not be zero")
    target /= length
    forward = np.array([0.0, 0.0, -1.0])
    axis = np.cross(forward, target)
    cosine = float(np.dot(forward, target))
    if np.linalg.norm(axis) < 1e-9:
        # Already aligned, or exactly opposite: turn half way around x.
        return quaternion.one if cosine > 0 else quaternion.quaternion(0.0, 1.0, 0.0, 0.0)
    angle = math.atan2(float(np.linalg.norm(axis)), cosine)
    return quaternion.from_rotation_vector(axis / np.linalg.norm(axis) * angle)


def apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 matrix to an ``(N, 3)`` array of points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]
