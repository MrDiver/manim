"""manimx: animations written in Python, rendered on the GPU.

Scenes are written with the classes this module exports (see
GETTING_STARTED.md); README.md maps the packages."""

__version__ = "0.1.0"

# What scene code is written with: the core scene side, userland, and the
# value types nodes take (materials, textures, meshes, stroke styles). Render
# classes (RShape2D, ...) are snapshots, not authoring API: they stay in
# manimx.robjects.
from .animation import *  # noqa: F403
from .animation import __all__ as _animation_all
from .drivers import render_image
from .renderer import RenderConfig
from .robjects import CapStyle, JointType, MeshData, PbrMaterial, SurfaceShader, Texture
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
    "Texture",
    "SurfaceShader",
    "__version__",
    "assert_snapshot",
    "package_name",
    "render_image",
    "visual_test",
]
