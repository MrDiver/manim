"""Frame effects: settings for the core's compositor (core/src/render/composite/).

Effects are plain settings; the compositor turns them into GPU work. There
are two kinds:

* **Temporal** effects decide how often and when the renderer is called.
  :class:`RenderSamples` averages several renders per frame, each with
  the camera moved by a fraction of a pixel and the shadow taps turned,
  and spreads them over the shutter for motion blur. Removing it from
  the chain means one render per frame.
* **Image** effects run shaders on textures, either on the lit 3D layer in
  linear HDR before tone mapping (``stage="lit"``: :class:`Bloom`, custom
  shaders) or on the final linear frame (``stage="frame"``: custom shaders).

:class:`ToneMap` selects how the lit layer's HDR radiance is mapped to
displayable colors. Flat 2D colors (Shape2D, Curve3D) are never tone
mapped, so their hex colors come out exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Stage = Literal["lit", "frame"]


class Effect:
    """Base class of all effects."""


@dataclass(frozen=True, slots=True)
class RenderSamples(Effect):
    """Average ``count`` renders per frame, like the samples of Blender's
    Eevee: smoother edges, and soft shadows from fewer taps per render.

    Every render moves the camera by a different fraction of a pixel (the
    Halton sequence in bases 2 and 3, spread evenly over the pixel) and
    turns the shadow taps (``RenderConfig.shadow_index``). This is the
    accumulation buffer's antialiasing (Haeberli and Akeley 1990,
    https://doi.org/10.1145/97880.97913).

    ``shutter`` is the exposure time in seconds, centered on the frame's
    time (``RenderConfig.time``); above 0, the renders are spread over it
    and moving things blur. Film's "180 degree shutter" at 60 fps is
    ``shutter=1/120``. Because every render is a full render of the scene
    at its own time, the blur is exact for any change: movement, morphing
    points, colors, even objects appearing.
    """

    count: int = 16
    shutter: float = 0.0

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("render samples need at least one render")
        if self.shutter < 0:
            raise ValueError("shutter time must not be negative")

    def sample_times(self, time: float) -> list[tuple[float, float]]:
        """``(time, weight)`` of each render; the weights sum to one."""
        weight = 1 / self.count
        start = time - self.shutter / 2
        return [(start + self.shutter * (i + 0.5) / self.count, weight) for i in range(self.count)]

    def jitter(self, index: int) -> tuple[float, float]:
        """Render ``index``'s camera offset in pixels, in [-0.5, 0.5);
        none with one render."""
        if self.count == 1:
            return (0.0, 0.0)
        return (_halton(index + 1, 2) - 0.5, _halton(index + 1, 3) - 0.5)


def _halton(index: int, base: int) -> float:
    """Element ``index`` of the van der Corput sequence in ``base``
    (https://en.wikipedia.org/wiki/Halton_sequence)."""
    result, fraction = 0.0, 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


class ToneMapOperator(Enum):
    """Values match the TONE_MAP_* constants in core/shaders/color.wgsl."""

    NONE = 0
    """Clamp to [0, 1]."""
    PBR_NEUTRAL = 1
    """Khronos PBR Neutral: base colors stay faithful (the default)."""
    ACES = 2
    """Filmic ACES fit: strong contrast, desaturated highlights."""
    AGX = 3
    """AgX, Blender's default: smooth highlight roll-off."""
    REINHARD = 4
    """x / (1 + x): simple, low contrast."""


@dataclass(frozen=True, slots=True)
class ToneMap(Effect):
    operator: ToneMapOperator = ToneMapOperator.PBR_NEUTRAL


@dataclass(frozen=True, slots=True)
class Bloom(Effect):
    """Glow around bright lit pixels (HDR radiance above ``threshold``).

    ``intensity`` scales the glow, ``radius`` widens the blur filter and
    ``levels`` is the number of mip levels the glow spreads over (more
    levels give a wider glow).
    """

    threshold: float = 1.0
    knee: float = 0.5
    intensity: float = 0.1
    radius: float = 1.0
    levels: int = 6

    def __post_init__(self) -> None:
        if self.levels < 1:
            raise ValueError("bloom needs at least one level")


@dataclass(frozen=True, slots=True)
class ShaderEffect(Effect):
    """A custom WGSL effect applied to a texture.

    ``code`` must define::

        fn effect(uv: vec2<f32>, pixel: vec2<i32>) -> vec4<f32>

    and can use ``source`` (the input texture), ``source_sampler`` (linear,
    clamped), ``frame`` (camera, viewport and time, see frame.wgsl) and
    ``value(i)``, which returns ``values[i]``. ``stage="lit"`` runs on the
    lit 3D layer in linear HDR (premultiplied by coverage); ``stage="frame"``
    runs on the composited frame in linear light (premultiplied alpha).

    Example, a vignette::

        ShaderEffect('''
            fn effect(uv: vec2<f32>, pixel: vec2<i32>) -> vec4<f32> {
                let color = textureLoad(source, pixel, 0);
                let fade = 1.0 - value(0u) * dot(uv - 0.5, uv - 0.5);
                return vec4<f32>(color.rgb * fade, color.a);
            }
        ''', values=(1.5,))
    """

    code: str
    stage: Stage = "frame"
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.values) > 16:
            raise ValueError("a shader effect takes at most 16 values")
        if self.stage not in ("lit", "frame"):
            raise ValueError(f"unknown stage {self.stage!r}")
