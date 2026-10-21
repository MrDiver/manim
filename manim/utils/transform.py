"""Transform: where something sits, as position, rotation (a unit
quaternion) and scale, and the 4x4 matrix they make.

A plain value shared by the scene side and the renderer side: it depends
on neither, and holds no logic beyond reading out its own matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import quaternion

__all__ = ["Transform"]


@dataclass(frozen=True, slots=True)
class Transform:
    """Where something sits: ``position + rotation * (scale * point)``."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: A unit quaternion.
    rotation: quaternion.quaternion = field(default_factory=lambda: quaternion.one)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def matrix(self) -> np.ndarray:
        """The 4x4 matrix ``T @ R @ S``: scale, then rotate, then move."""
        result = np.eye(4)
        result[:3, :3] = quaternion.as_rotation_matrix(self.rotation) * np.asarray(
            self.scale
        )
        result[:3, 3] = self.position
        return result
