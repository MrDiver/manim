from __future__ import annotations

from dataclasses import dataclass

from .material import PbrMaterial
from .mesh import MeshData, box_mesh
from .render_object import RenderObject
from .versioning import versioned


@dataclass(slots=True, eq=False)
class Shape3D(RenderObject):
    """A lit triangle mesh with a physically based material.

    ``mesh`` is in local coordinates; :attr:`transform` places it in the
    world. Both fields are versioned: replace them instead of mutating them
    (their arrays are read-only).
    """

    mesh: MeshData = versioned(default_factory=box_mesh)
    material: PbrMaterial = versioned(default_factory=PbrMaterial)
    cast_shadows: bool = True
    """Whether this mesh blocks light (it always receives shadows)."""
