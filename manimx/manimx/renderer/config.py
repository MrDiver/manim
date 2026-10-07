"""Render settings and camera, passed to the renderer next to the scene tree."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..objects.transform import Transform
from ..utils.color import BLACK, ManimColor, ParsableManimColor

DEFAULT_FRAME_HEIGHT = 8.0
"""Manim's default frame height in world units."""


@dataclass(frozen=True, slots=True)
class Camera:
    """A camera placed by a :class:`Transform`, looking along its local ``-z``.

    ``field_of_view=None`` gives Manim's default orthographic camera, which
    shows exactly ``frame_height`` world units vertically. A vertical field of
    view in radians switches to a perspective projection.
    """

    transform: Transform = field(
        default_factory=lambda: Transform(position=(0.0, 0.0, 11.0))
    )
    frame_height: float = DEFAULT_FRAME_HEIGHT
    field_of_view: float | None = None
    near: float = 0.1
    far: float = 100.0

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.transform.position)

    def view_matrix(self) -> np.ndarray:
        """World to view space (the inverse of the camera's placement)."""
        return np.linalg.inv(self.transform.matrix())

    def projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """View to clip space with WebGPU's depth range ``[0, 1]``."""
        near, far = self.near, self.far
        if self.field_of_view is None:
            half_height = self.frame_height / 2
            half_width = half_height * aspect_ratio
            return np.array(
                [
                    [1 / half_width, 0, 0, 0],
                    [0, 1 / half_height, 0, 0],
                    [0, 0, -1 / (far - near), -near / (far - near)],
                    [0, 0, 0, 1],
                ]
            )
        focal = 1 / math.tan(self.field_of_view / 2)
        return np.array(
            [
                [focal / aspect_ratio, 0, 0, 0],
                [0, focal, 0, 0],
                [0, 0, far / (near - far), near * far / (near - far)],
                [0, 0, -1, 0],
            ]
        )


@dataclass(frozen=True, slots=True)
class RenderConfig:
    pixel_width: int = 1920
    pixel_height: int = 1080
    background_color: ParsableManimColor = BLACK
    background_opacity: float = 1.0
    camera: Camera = field(default_factory=Camera)
    msaa_samples: int = 4
    time: float = 0.0
    """Seconds, exposed to shaders as ``frame.time``."""
    layer_depth_step: float = 2e-7
    """Depth offset between consecutive 2D layers, see ``vector.wgsl``."""

    @property
    def aspect_ratio(self) -> float:
        return self.pixel_width / self.pixel_height

    @property
    def frame_width(self) -> float:
        return self.camera.frame_height * self.aspect_ratio

    @property
    def background_rgba(self) -> tuple[float, float, float, float]:
        r, g, b = ManimColor.parse(self.background_color).to_rgb()
        return (float(r), float(g), float(b), self.background_opacity)
