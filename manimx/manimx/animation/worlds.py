"""Worlds: the spaces clips play in (section 8.1 of the design doc).

A session has any number of worlds. Each clip plays in exactly one of
them, and content of different worlds never mixes directly, only through
views (a camera's image of a world, placed into another world).

* :class:`World3D` is a 3D space: clips are placed anywhere, cameras and
  lights live there, and its image depends on the camera looking at it.
* :class:`World2D` is a flat canvas in the z = 0 plane, for HUD elements
  and compositions. Clips are placed in the plane and only rotate around
  the z axis, so a 2D world never needs a perspective camera and its
  stacking order is exactly its layer order.

A world is plain data plus a render identity: its root group keeps the
same ``uid`` in every snapshot, like every other object the renderer
caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..robjects import Transform, new_uid


@dataclass(eq=False)
class World:
    """A space that clips are placed in."""

    name: str
    uid: int = field(default_factory=new_uid, init=False, repr=False)
    dimensions: ClassVar[int] = 3

    def check_transform(self, transform: Transform) -> None:
        """Raise if ``transform`` cannot place a clip in this world."""


@dataclass(eq=False)
class World3D(World):
    """A 3D space, seen through cameras."""


@dataclass(eq=False)
class World2D(World):
    """A flat canvas (z = 0) for HUD elements and compositions."""

    dimensions: ClassVar[int] = 2

    def check_transform(self, transform: Transform) -> None:
        if transform.position[2] != 0.0:
            raise ValueError(f"clips in the 2D world {self.name!r} stay in the z = 0 plane")
        # A rotation keeps the plane if and only if it keeps the z axis.
        z = transform.apply([(0.0, 0.0, 1.0)])[0] - transform.position
        if not np.allclose(z[:2], 0.0, atol=1e-9):
            raise ValueError(f"clips in the 2D world {self.name!r} only rotate around the z axis")
