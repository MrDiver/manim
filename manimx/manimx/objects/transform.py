from __future__ import annotations

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "scale", _vector3(self.scale, "scale"))
        object.__setattr__(self, "rotation", _normalized(self.rotation))

    def matrix(self) -> np.ndarray:
        """Homogeneous 4x4 matrix computing ``T @ R @ S``."""
        result = np.eye(4)
        result[:3, :3] = quaternion.as_rotation_matrix(self.rotation) * np.asarray(
            self.scale
        )
        result[:3, 3] = self.position
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

    def interpolate(self, other: Transform, alpha: float) -> Transform:
        """Blend towards ``other``: lerp position and scale, slerp rotation."""
        target = other.rotation
        # q and -q are the same rotation; flip to take the shortest arc.
        if np.dot(self.rotation.components, target.components) < 0.0:
            target = -target
        return Transform(
            position=tuple(
                (1.0 - alpha) * np.asarray(self.position)
                + alpha * np.asarray(other.position)
            ),
            rotation=quaternion.slerp_evaluate(self.rotation, target, alpha),
            scale=tuple(
                (1.0 - alpha) * np.asarray(self.scale) + alpha * np.asarray(other.scale)
            ),
        )


def apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 matrix to an ``(N, 3)`` array of points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]
