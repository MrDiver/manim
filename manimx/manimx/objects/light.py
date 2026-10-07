"""Light sources. They are leaves of the scene tree like shapes, so they move
with their groups (and can be animated later), but they draw nothing.

Only :class:`Shape3D` meshes are lit. Shape2D and Curve3D keep their flat
Manim colors.

The three kinds follow glTF's KHR_lights_punctual. Lights that have a
direction shine along their local ``-z`` axis (the glTF and Blender
convention); use the ``aimed`` constructors to point them.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import quaternion

from ..utils.color import WHITE, ParsableManimColor
from .render_object import RenderObject
from .transform import Quaternion, Transform


@dataclass(slots=True, eq=False)
class Light(RenderObject):
    """``intensity`` scales the radiance directly: a white diffuse surface
    lit head-on by a directional light of intensity pi appears pure white.

    ``cast_shadows`` turns this light's shadows on (RenderConfig.shadows
    must be on too). ``size`` is how large the light appears, which sets
    how soft its shadows get away from the caster (contact-hardening
    shadows): the radius in world units for point and spot lights, the
    angular diameter in radians for directional lights. Zero gives hard
    shadows.

    A light inside a :class:`ClippingGroup`'s content is scoped to that
    group: it only lights, and only gets shadows from, objects in the same
    group (including nested groups). Lights outside any clipping group
    light everything.
    """

    color: ParsableManimColor = WHITE
    intensity: float = 1.0
    cast_shadows: bool = True
    size: float = 0.1


@dataclass(slots=True, eq=False)
class DirectionalLight(Light):
    """Parallel light (like the sun). Its position does not matter."""

    intensity: float = 3.0
    size: float = 0.03  # angular diameter in radians, about 1.7 degrees

    @classmethod
    def aimed(cls, id: str, direction: tuple[float, float, float], **kwargs: Any) -> Self:
        """A light shining in world ``direction``."""
        return cls(id, transform=Transform(rotation=_rotation_towards(direction)), **kwargs)


@dataclass(slots=True, eq=False)
class PointLight(Light):
    """Light emitted in all directions from the light's position.

    Falls off with the inverse square of the distance; ``range`` (if set)
    smoothly fades it to zero there.
    """

    intensity: float = 20.0
    range: float | None = None


@dataclass(slots=True, eq=False)
class SpotLight(Light):
    """A point light restricted to a cone around its local ``-z`` axis.

    Full intensity inside ``inner_cone_angle``, fading smoothly to zero at
    ``outer_cone_angle`` (both in radians from the axis, as in glTF).
    """

    intensity: float = 40.0
    range: float | None = None
    inner_cone_angle: float = 0.0
    outer_cone_angle: float = math.pi / 4

    def __post_init__(self, versions: Mapping[str, Hashable] | None = None) -> None:
        Light.__post_init__(self, versions)
        if not 0.0 <= self.inner_cone_angle < self.outer_cone_angle <= math.pi / 2:
            raise ValueError("cone angles need 0 <= inner < outer <= pi / 2")

    @classmethod
    def aimed(
        cls,
        id: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        **kwargs: Any,
    ) -> Self:
        """A spot light at ``position`` shining at ``target``."""
        direction = np.subtract(target, position)
        rotation = _rotation_towards(tuple(direction))
        return cls(id, transform=Transform(position=position, rotation=rotation), **kwargs)


def _rotation_towards(direction: tuple[float, float, float]) -> Quaternion:
    """Rotation turning local ``-z`` onto ``direction`` (shortest arc)."""
    target = np.asarray(direction, dtype=np.float64)
    length = np.linalg.norm(target)
    if length == 0.0:
        raise ValueError("light direction must not be zero")
    target /= length
    forward = np.array([0.0, 0.0, -1.0])
    axis = np.cross(forward, target)
    cosine = float(np.dot(forward, target))
    if np.linalg.norm(axis) < 1e-9:
        # Already aligned, or exactly opposite: turn half way around x.
        return quaternion.one if cosine > 0 else quaternion.quaternion(0.0, 1.0, 0.0, 0.0)
    angle = math.atan2(float(np.linalg.norm(axis)), cosine)
    return quaternion.from_rotation_vector(axis / np.linalg.norm(axis) * angle)
