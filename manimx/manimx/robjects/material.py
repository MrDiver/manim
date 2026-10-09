"""glTF-style metallic-roughness materials and their textures."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..utils.color import BLACK, WHITE, ManimColor, ParsableManimColor

_next_texture_uid = itertools.count(1)


@dataclass(frozen=True, slots=True, eq=False)
class Texture:
    """An RGBA8 image used by a material.

    ``uid`` identifies the texture for GPU caches, so the same image is
    uploaded once no matter how many materials share it.
    """

    image: np.ndarray  # (H, W, 4) uint8, first row at the top
    uid: int = field(default_factory=lambda: next(_next_texture_uid), init=False)

    def __post_init__(self) -> None:
        image = np.ascontiguousarray(self.image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"texture image must have shape (H, W, 4), got {image.shape}")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.image.shape[1], self.image.shape[0]


def linear_rgb(color: ParsableManimColor) -> tuple[float, float, float]:
    """sRGB-encoded color (how Manim colors are written) to linear RGB."""
    srgb = np.asarray(ManimColor.parse(color).to_rgb(), dtype=np.float64)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return (float(linear[0]), float(linear[1]), float(linear[2]))


@dataclass(frozen=True, slots=True, eq=False)
class PbrMaterial:
    """The glTF 2.0 metallic-roughness material model.

    Colors are given like all Manim colors (sRGB, e.g. ``"#58C4DD"``) and
    converted to linear RGB, in which lighting happens. Each factor is
    multiplied with its texture when one is set:

    * ``base_color_texture``: sRGB RGBA
    * ``metallic_roughness_texture``: linear, roughness in G, metallic in B
    * ``normal_texture``: tangent-space normals (needs mesh UVs)
    * ``occlusion_texture``: linear, ambient occlusion in R
    * ``emissive_texture``: sRGB RGB
    """

    base_color: ParsableManimColor = WHITE
    opacity: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: ParsableManimColor = BLACK
    emissive_strength: float = 1.0
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    base_color_texture: Texture | None = None
    metallic_roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    occlusion_texture: Texture | None = None
    emissive_texture: Texture | None = None

    def __post_init__(self) -> None:
        for name in ("opacity", "metallic", "roughness", "occlusion_strength"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def base_color_linear(self) -> tuple[float, float, float, float]:
        return (*linear_rgb(self.base_color), self.opacity)

    @property
    def emissive_linear(self) -> tuple[float, float, float]:
        r, g, b = linear_rgb(self.emissive)
        return (r * self.emissive_strength, g * self.emissive_strength, b * self.emissive_strength)

    @property
    def textures(self) -> tuple[Texture | None, ...]:
        """All texture slots in a fixed order (matches the shader bindings)."""
        return (
            self.base_color_texture,
            self.metallic_roughness_texture,
            self.normal_texture,
            self.occlusion_texture,
            self.emissive_texture,
        )
