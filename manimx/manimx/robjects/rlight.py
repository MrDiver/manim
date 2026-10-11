"""Light sources. They are leaves of the scene tree like shapes, so they move
with their groups (and can be animated later), but they draw nothing.

Only :class:`RShape3D` meshes are lit. RShape2D and RCurve3D keep their flat
Manim colors.

The three kinds follow glTF's KHR_lights_punctual. Lights that have a
direction shine along their local ``-z`` axis (the glTF and Blender
convention). Scene lights (manimx.animation.nodes.light) point them with
``direction=``, ``target=``, ``point_along()`` and ``point_at()``; a render
light only holds the resulting rotation in its transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..utils.color import WHITE, ParsableManimColor
from .rnode import RNode


@dataclass(slots=True, eq=False)
class RLight(RNode):
    """``intensity`` scales the radiance directly: a white diffuse surface
    lit head-on by a directional light of intensity pi appears pure white.

    ``cast_shadows`` turns this light's shadows on (RenderConfig.shadows
    must be on too). ``size`` is how large the light appears, which sets
    how soft its shadows get away from the caster (contact-hardening
    shadows): the radius in world units for point and spot lights, the
    angular diameter in radians for directional lights. Zero gives hard
    shadows.

    A light inside an :class:`RClippingGroup`'s content is scoped to that
    group: it only lights, and only gets shadows from, objects in the same
    group (including nested groups). Lights outside any clipping group
    light everything.
    """

    color: ParsableManimColor = WHITE
    intensity: float = 1.0
    cast_shadows: bool = True
    size: float = 0.1


@dataclass(slots=True, eq=False)
class RDirectionalLight(RLight):
    """Parallel light (like the sun). Its position does not matter."""

    intensity: float = 3.0
    size: float = 0.03  # angular diameter in radians, about 1.7 degrees


@dataclass(slots=True, eq=False)
class RPointLight(RLight):
    """Light emitted in all directions from the light's position.

    Falls off with the inverse square of the distance; ``range`` (if set)
    smoothly fades it to zero there.
    """

    intensity: float = 20.0
    range: float | None = None


@dataclass(slots=True, eq=False)
class RSpotLight(RLight):
    """A point light restricted to a cone around its local ``-z`` axis.

    Full intensity inside ``inner_cone_angle``, fading smoothly to zero at
    ``outer_cone_angle`` (both in radians from the axis, as in glTF).
    """

    intensity: float = 40.0
    range: float | None = None
    inner_cone_angle: float = 0.0
    outer_cone_angle: float = math.pi / 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.inner_cone_angle < self.outer_cone_angle <= math.pi / 2:
            raise ValueError("cone angles need 0 <= inner < outer <= pi / 2")
