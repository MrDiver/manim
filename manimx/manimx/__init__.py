"""Parallel package for incremental Manim experiments."""

__version__ = "0.1.0"

from .objects import (
    CapStyle,
    ClippingGroup,
    Fill,
    Group,
    JointType,
    Quaternion,
    RenderObject,
    Shape2D,
    Shape3D,
    Stroke,
    Transform,
    quaternion_from_axis_angle,
)
from .renderer import SVGRenderer
from .testing import assert_snapshot, visual_test


def package_name() -> str:
    return "manimx"


__all__ = [
    "CapStyle",
    "ClippingGroup",
    "Fill",
    "Group",
    "JointType",
    "Quaternion",
    "RenderObject",
    "Shape2D",
    "Shape3D",
    "SVGRenderer",
    "Stroke",
    "Transform",
    "__version__",
    "assert_snapshot",
    "package_name",
    "quaternion_from_axis_angle",
    "visual_test",
]
