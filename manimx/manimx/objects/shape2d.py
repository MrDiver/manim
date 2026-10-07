from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .path import POINTS_PER_CURVE, TOLERANCE_FOR_POINT_EQUALITY, BezierPath, is_closed
from .style import Fill
from .versioning import versioned

__all__ = ["POINTS_PER_CURVE", "TOLERANCE_FOR_POINT_EQUALITY", "Shape2D", "is_closed"]


@dataclass(slots=True, eq=False)
class Shape2D(BezierPath):
    """Drawable vector shape, the counterpart of Manim's ``VMobject``.

    Unlike ``VMobject`` a shape never has submobjects: hierarchy is expressed
    with :class:`Group` and :class:`ClippingGroup`, which draw nothing. Holes
    and disjoint pieces (e.g. a glyph) are subpaths of a single shape.

    ``points`` is ``(N, 2)`` in the ``VMobject`` layout (see
    :class:`BezierPath`). Points live in the local XY plane; :attr:`transform`
    places the plane in 3D, and the stroke is a flat ribbon in that plane.

    ``points``, ``fill`` and ``stroke`` are versioned: replace them instead of
    mutating them.
    """

    DIMENSION: ClassVar[int] = 2

    fill: Fill = versioned(default_factory=Fill)

    def fill_rgba(self) -> np.ndarray:
        """Fill RGBA per control point, shape ``(N, 4)``."""
        return self.fill.rgba(len(self.points))
