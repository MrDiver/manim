from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .rbezier_path import POINTS_PER_CURVE, TOLERANCE_FOR_POINT_EQUALITY, RBezierPath, is_closed
from .style import Fill
from .versioning import versioned

__all__ = ["POINTS_PER_CURVE", "TOLERANCE_FOR_POINT_EQUALITY", "RShape2D", "is_closed"]


@dataclass(slots=True, eq=False)
class RShape2D(RBezierPath):
    """Drawable vector shape, the snapshot of an animation ``Shape2D``.

    A shape is a leaf: hierarchy is expressed with :class:`RGroup` and
    :class:`RClippingGroup`, which draw nothing. Holes and disjoint pieces
    (e.g. a glyph) are subpaths of a single shape.

    ``points`` is ``(N, 2)`` in the :class:`RBezierPath` layout (see
    :class:`RBezierPath`). Points live in the local XY plane; :attr:`transform`
    places the plane in 3D, and the stroke is a flat ribbon in that plane.

    ``points``, ``fill`` and ``stroke`` are versioned: replace them instead of
    mutating them.
    """

    DIMENSION: ClassVar[int] = 2

    fill: Fill = versioned(default_factory=Fill)

    def fill_rgba(self) -> np.ndarray:
        """Fill RGBA per control point, shape ``(N, 4)``."""
        return self.fill.rgba(len(self.points))
