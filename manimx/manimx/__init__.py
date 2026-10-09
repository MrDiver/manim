"""Parallel package for incremental Manim experiments."""

__version__ = "0.1.0"

from .robjects import (
    CapStyle,
    RClippingGroup,
    RCurve3D,
    RDirectionalLight,
    Fill,
    RGroup,
    JointType,
    RLight,
    MeshData,
    PbrMaterial,
    RPointLight,
    Quaternion,
    RNode,
    RShape2D,
    RShape3D,
    RSpotLight,
    Stroke,
    Texture,
    Transform,
    quaternion_from_axis_angle,
)
from .renderer import RenderConfig, Renderer, WGPURenderer
from .testing import assert_snapshot, visual_test


def package_name() -> str:
    return "manimx"


__all__ = [
    "CapStyle",
    "RClippingGroup",
    "RCurve3D",
    "RDirectionalLight",
    "Fill",
    "RGroup",
    "JointType",
    "RLight",
    "MeshData",
    "PbrMaterial",
    "RPointLight",
    "Quaternion",
    "RenderConfig",
    "RNode",
    "Renderer",
    "RShape2D",
    "RShape3D",
    "RSpotLight",
    "Stroke",
    "Texture",
    "Transform",
    "WGPURenderer",
    "assert_snapshot",
    "package_name",
    "quaternion_from_axis_angle",
    "visual_test",
    "__version__",
]
