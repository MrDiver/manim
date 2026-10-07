"""Parallel package for incremental Manim experiments."""

__version__ = "0.1.0"

from .objects import (
    CapStyle,
    ClippingGroup,
    Curve3D,
    DirectionalLight,
    Fill,
    Group,
    JointType,
    Light,
    MeshData,
    PbrMaterial,
    PointLight,
    Quaternion,
    RenderObject,
    Shape2D,
    Shape3D,
    SpotLight,
    Stroke,
    Texture,
    Transform,
    box_mesh,
    quaternion_from_axis_angle,
    sphere_mesh,
    torus_mesh,
)
from .renderer import RenderConfig, Renderer, WGPURenderer
from .testing import assert_snapshot, visual_test


def package_name() -> str:
    return "manimx"


__all__ = [
    "CapStyle",
    "ClippingGroup",
    "Curve3D",
    "DirectionalLight",
    "Fill",
    "Group",
    "JointType",
    "Light",
    "MeshData",
    "PbrMaterial",
    "PointLight",
    "Quaternion",
    "RenderConfig",
    "RenderObject",
    "Renderer",
    "Shape2D",
    "Shape3D",
    "SpotLight",
    "Stroke",
    "Texture",
    "Transform",
    "WGPURenderer",
    "assert_snapshot",
    "box_mesh",
    "package_name",
    "quaternion_from_axis_angle",
    "sphere_mesh",
    "torus_mesh",
    "visual_test",
    "__version__",
]
