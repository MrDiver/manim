"""The Python side of the renderer: how a frame is configured.

Rendering itself happens in the core (``manimx._core.FrameRenderer``,
core/src/gpu/), driven by :mod:`manimx.drivers`. This package holds what
Python gives it: :class:`RenderConfig` and :class:`Camera` (size,
lighting, shadows, the point of view), the effect settings, the camera's
Frame record, the functions the core calls back, and the WGSL shaders
(``shaders/``). See README.md.
"""

from .config import Camera, CameraViewState, Frame, RenderConfig, ViewState
from .effects import Bloom, Effect, MotionBlur, ShaderEffect, ToneMap, ToneMapOperator

__all__ = [
    "Bloom",
    "Camera",
    "CameraViewState",
    "Effect",
    "Frame",
    "MotionBlur",
    "RenderConfig",
    "ShaderEffect",
    "ToneMap",
    "ToneMapOperator",
    "ViewState",
]
