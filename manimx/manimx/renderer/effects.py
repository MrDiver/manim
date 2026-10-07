"""Frame effects: settings for the :class:`FrameComposer`.

Effects are plain settings; the composer turns them into GPU work. There
are two kinds:

* **Temporal** effects decide how often and when the renderer is called.
  :class:`MotionBlur` renders several sub-frames inside the shutter
  interval and averages them. Removing it from the chain means one render
  per frame.
* **Image** effects run shaders on textures, either on the lit 3D layer in
  linear HDR before tone mapping (``stage="lit"``: :class:`Bloom`, custom
  shaders) or on the final linear frame (``stage="frame"``: custom shaders).

:class:`ToneMap` selects how the lit layer's HDR radiance is mapped to
displayable colors. Manim's flat 2D colors (Shape2D, Curve3D) are never
tone mapped, so their hex colors come out exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Stage = Literal["lit", "frame"]


class Effect:
    """Base class of all effects."""


@dataclass(frozen=True, slots=True)
class MotionBlur(Effect):
    """Average ``samples`` renders spread over the shutter interval.

    ``shutter`` is the exposure time in seconds, centered on the frame's
    time (``RenderConfig.time``). Film's "180 degree shutter" at 60 fps is
    ``shutter=1/120``. Because every sub-frame is a full render of the scene
    at its own time, the blur is exact for any change: movement, morphing
    points, colors, even objects appearing.
    """

    samples: int = 16
    shutter: float = 1 / 120

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("motion blur needs at least one sample")
        if self.shutter < 0:
            raise ValueError("shutter time must not be negative")

    def sample_times(self, time: float) -> list[tuple[float, float]]:
        """``(time, weight)`` of each sub-frame; the weights sum to one."""
        weight = 1 / self.samples
        start = time - self.shutter / 2
        return [(start + self.shutter * (i + 0.5) / self.samples, weight) for i in range(self.samples)]


class ToneMapOperator(Enum):
    """Values match the TONE_MAP_* constants in shaders/color.wgsl."""

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
