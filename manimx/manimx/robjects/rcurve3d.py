"""RCurve3D: the snapshot of a ``Curve3D``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .rbezier_path import RBezierPath


@dataclass(slots=True, eq=False)
class RCurve3D(RBezierPath):
    """A stroke-only path of cubic Bezier curves in 3D space.

    ``points`` is ``(N, 3)`` in the :class:`RBezierPath` layout, so the
    outline of an :class:`RShape2D` lifts to a curve exactly. The stroke supports everything an ``RShape2D`` stroke does
    (per-point color, opacity and width, joints, caps), but its ribbon always
    faces the camera. Its width is in world units, so under a perspective
    camera it gets thinner with distance.
    """

    DIMENSION: ClassVar[int] = 3
