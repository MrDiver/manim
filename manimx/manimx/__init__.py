"""Parallel package for incremental Manim experiments."""

__version__ = "0.1.0"

# What scene code is written with: the core scene side, userland, and the
# value types nodes take (materials, textures, meshes, stroke styles). Render
# classes (RShape2D, ...) are snapshots, not authoring API: they stay in
# manimx.robjects.
from .animation import *  # noqa: F403
from .animation import __all__ as _animation_all
from .renderer import RenderConfig, Renderer, RenderService
from .robjects import CapStyle, JointType, MeshData, PbrMaterial, Texture
from .testing import assert_snapshot, visual_test
from .userland import *  # noqa: F403
from .userland import __all__ as _userland_all


def package_name() -> str:
    return "manimx"


__all__ = [
    *_animation_all,
    *_userland_all,
    "CapStyle",
    "JointType",
    "MeshData",
    "PbrMaterial",
    "RenderConfig",
    "Renderer",
    "Texture",
    "RenderService",
    "__version__",
    "assert_snapshot",
    "package_name",
    "visual_test",
]
