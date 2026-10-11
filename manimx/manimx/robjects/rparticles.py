"""RParticles: the snapshot of ``Particles``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rshape3d import RShape3D


def _none(columns: int) -> np.ndarray:
    array = np.zeros((0, columns))
    array.setflags(write=False)
    return array


@dataclass(slots=True, eq=False)
class RParticles(RShape3D):
    """Many copies of one mesh, drawn as one instanced draw.

    Every particle ``i`` is the mesh placed by ``positions[i]``,
    ``rotations[i]`` (a quaternion ``w, x, y, z``) and ``scales[i]`` in the
    object's local space; :attr:`transform` places them all. ``colors[i]``
    is ``r, g, b, glow``: the color multiplies the material's base and
    emissive color (in linear light), the glow its emissive strength. A
    particle scaled to nothing on any axis is not drawn. The arrays are
    read-only.
    """

    positions: np.ndarray = field(default_factory=lambda: _none(3))
    rotations: np.ndarray = field(default_factory=lambda: _none(4))
    scales: np.ndarray = field(default_factory=lambda: _none(3))
    colors: np.ndarray = field(default_factory=lambda: _none(4))
