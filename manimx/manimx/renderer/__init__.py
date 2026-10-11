"""How a frame is configured: what a renderer is given next to the
render object tree (manimx.robjects).

:class:`RenderConfig` and :class:`Camera` (size, lighting, shadows, the
point of view), :class:`ViewState` (a tree with the camera looking at
it), the effect settings, and the WGSL shaders (``shaders/``) of the
core's renderer (``manimx._core.FrameRenderer``, core/src/render/),
which :mod:`manimx.drivers` drives. See README.md.
"""

from .config import Camera, CameraViewState, Frame, RenderConfig, ViewState
from .effects import Bloom, Effect, RenderSamples, ShaderEffect, ToneMap, ToneMapOperator

__all__ = [
    "Bloom",
    "Camera",
    "CameraViewState",
    "Effect",
    "Frame",
    "RenderConfig",
    "RenderSamples",
    "ShaderEffect",
    "ToneMap",
    "ToneMapOperator",
    "ViewState",
]
