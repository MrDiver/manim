from __future__ import annotations

from dataclasses import dataclass

from .material import PbrMaterial
from .mesh import MeshData, box_mesh
from .rnode import RNode
from .versioning import versioned


@dataclass(slots=True, eq=False)
class RShape3D(RNode):
    """A lit triangle mesh with a physically based material.

    ``mesh`` is in local coordinates; :attr:`transform` places it in the
    world. Both fields are versioned: replace them instead of mutating them
    (their arrays are read-only).
    """

    mesh: MeshData = versioned(default_factory=box_mesh)
    material: PbrMaterial = versioned(default_factory=PbrMaterial)
    cast_shadows: bool = True
    """Whether this mesh blocks light (it always receives shadows)."""
