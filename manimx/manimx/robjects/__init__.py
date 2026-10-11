"""Render state and the values nodes are made of.

``PreparedScene.state_at(t)`` turns the scene nodes (manimx.animation)
into a tree of render objects, one class per node class: an ``R`` and
the node's name (``RShape2D`` is the snapshot of a ``Shape2D``). The
tree is what a scene shows at one time, for tests and tools; frames are
rendered by the core, which reads the signals itself.

| Module | What it holds |
|---|---|
| ``rnode`` | ``RNode``, the base: uid, transform, alpha |
| ``rgroup``, ``rclipping_group`` | the containers |
| ``rbezier_path``, ``rshape2d``, ``rcurve3d`` | paths of cubic Bezier curves |
| ``rshape3d``, ``rparticles``, ``rlight`` | meshes and lights |
| ``planned_frame`` | ``PlannedFrame``: a frame as the core renders it |
| ``style`` | ``Fill``, ``Stroke``, ``JointType``, ``CapStyle`` |
| ``transform`` | ``Transform`` and quaternion helpers |
| ``material``, ``mesh`` | ``PbrMaterial``, ``Texture``, ``SurfaceShader``, ``MeshData`` |

The value types (styles, materials, meshes, transforms) are also what
scene nodes take as arguments.
"""

from .material import PbrMaterial, SurfaceShader, Texture
from .mesh import MeshData
from .rclipping_group import RClippingGroup
from .rcurve3d import RCurve3D
from .rgroup import RGroup
from .rlight import RDirectionalLight, RLight, RPointLight, RSpotLight
from .rnode import RNode, new_uid
from .rparticles import RParticles
from .rshape2d import RShape2D
from .rshape3d import RShape3D
from .style import CapStyle, Fill, JointType, Stroke
from .transform import Quaternion, Transform, quaternion_from_axis_angle

__all__ = [
    "CapStyle",
    "Fill",
    "JointType",
    "MeshData",
    "PbrMaterial",
    "SurfaceShader",
    "Quaternion",
    "RClippingGroup",
    "RCurve3D",
    "RDirectionalLight",
    "RGroup",
    "RLight",
    "RNode",
    "RParticles",
    "RPointLight",
    "RShape2D",
    "RShape3D",
    "RSpotLight",
    "Stroke",
    "Texture",
    "Transform",
    "new_uid",
    "quaternion_from_axis_angle",
]
