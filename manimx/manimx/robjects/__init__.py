"""Render objects: what a scene shows at one time, as a tree of plain
values, and the values the tree holds.

The tree and a ``RenderConfig`` are everything a renderer is given::

    scene nodes, signals  --snapshot-->  RNode tree  <--reads--  any renderer
    (manimx.animation)                  (this package)          (the core's, or another)

A renderer never sees signals, animations or timelines.
``PreparedScene.state_at(t)`` and ``Node.snapshot()`` build trees from a
scene; a tree can as well be built by hand. One class per scene node
class: an ``R`` and the node's name (``RShape2D`` is the snapshot of a
``Shape2D``).

| Class | What it is |
|---|---|
| ``RNode`` | the base: ``uid``, ``transform``, ``alpha``, ``version(field)`` |
| ``RGroup``, ``RClippingGroup`` | the containers; only they have children |
| ``RBezierPath``: ``RShape2D``, ``RCurve3D`` | paths of cubic Bezier curves |
| ``RShape3D``, ``RParticles`` | lit triangle meshes |
| ``RLight``: ``RDirectionalLight``, ``RPointLight``, ``RSpotLight`` | lights |
| ``Fill``, ``Stroke`` | path styles (``JointType``, ``CapStyle`` in ``style``) |
| ``MeshData``, ``PbrMaterial``, ``Texture``, ``SurfaceShader`` | meshes and materials |
| ``Transform`` (``transform``) | where a node sits: position, rotation, scale |

The classes are implemented in the core (core/src/robjects/ and
core/src/python/robjects.rs), so a tree the core builds reaches its
renderer, or Python, without a copy. They are immutable. Reading a field
makes a new Python object each time: compare render objects by ``uid``
and values by ``version()``, never with ``is``.

Change detection: ``node.version("points")`` (and ``"fill"``,
``"stroke"``, ``"mesh"``, ``"material"``, a particle field) stays the
same while the value does: snapshots of a node whose signals did not
change share their values. A renderer keeps what it built from a value
while its version holds.

The value types (styles, materials, meshes, transforms) are also what
scene nodes take as arguments.
"""

from .._core import (
    Fill,
    MeshData,
    PbrMaterial,
    RBezierPath,
    RClippingGroup,
    RCurve3D,
    RDirectionalLight,
    RGroup,
    RLight,
    RNode,
    RParticles,
    RPointLight,
    RShape2D,
    RShape3D,
    RSpotLight,
    Stroke,
    SurfaceShader,
    Texture,
    new_uid,
)
from .style import CapStyle, JointType
from .transform import Quaternion, Transform, quaternion_from_axis_angle

__all__ = [
    "CapStyle",
    "Fill",
    "JointType",
    "MeshData",
    "PbrMaterial",
    "Quaternion",
    "RBezierPath",
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
    "SurfaceShader",
    "Texture",
    "Transform",
    "new_uid",
    "quaternion_from_axis_angle",
]
