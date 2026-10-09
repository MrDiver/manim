from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .material import PbrMaterial
from .mesh import MeshData
from .rnode import RNode
from .versioning import versioned


def _no_mesh() -> MeshData:
    """An empty mesh: nothing to draw until a snapshot sets one."""
    return MeshData(np.zeros((0, 3)), np.zeros(0, dtype=np.uint32))


@dataclass(slots=True, eq=False)
class RShape3D(RNode):
    """A lit triangle mesh with a physically based material.

    ``mesh`` is in local coordinates; :attr:`transform` places it in the
    world. Both fields are versioned: replace them instead of mutating them
    (their arrays are read-only).
    """

    mesh: MeshData = versioned(default_factory=_no_mesh)
    material: PbrMaterial = versioned(default_factory=PbrMaterial)
    cast_shadows: bool = True
    """Whether this mesh blocks light (it always receives shadows)."""
