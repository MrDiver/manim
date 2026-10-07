from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .render_object import RenderObject
from .style import Fill, JointType, Stroke
from .versioning import versioned

POINTS_PER_CURVE = 4
TOLERANCE_FOR_POINT_EQUALITY = 1e-6


def _empty_points() -> np.ndarray:
    return np.zeros((0, 2))


@dataclass(slots=True, eq=False)
class Shape2D(RenderObject):
    """Drawable vector shape, the counterpart of Manim's ``VMobject``.

    Unlike ``VMobject`` a shape never has submobjects: hierarchy is expressed
    with :class:`Group` and :class:`ClippingGroup`, which draw nothing. Holes
    and disjoint pieces (e.g. a glyph) are subpaths of a single shape.

    ``points`` uses the ``VMobject`` layout: ``(N, 2)`` with four points per
    cubic Bezier curve ``(anchor, handle, handle, anchor)``. A new subpath
    starts wherever a curve's start differs from the previous curve's end.
    Points live in the local XY plane; :attr:`transform` places the plane in 3D.

    ``points``, ``fill`` and ``stroke`` are versioned: replace them instead of
    mutating them (``points`` is stored read-only to enforce this).
    """

    points: np.ndarray = versioned(default_factory=_empty_points)
    fill: Fill = versioned(default_factory=Fill)
    stroke: Stroke = versioned(default_factory=Stroke)

    def __setattr__(self, name: str, value: object) -> None:
        # Validate on every assignment, including the one in __init__.
        if name == "points":
            value = _validated_points(value)
        RenderObject.__setattr__(self, name, value)

    @classmethod
    def from_corners(
        cls,
        id: str,
        corners: Sequence[Sequence[float]],
        *,
        closed: bool = False,
        **kwargs: Any,
    ) -> Shape2D:
        """Polyline through ``corners`` as straight cubic curves.

        Mirrors ``VMobject.set_points_as_corners``.
        """
        anchors = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if closed and len(anchors) > 0:
            anchors = np.vstack([anchors, anchors[:1]])
        starts, ends = anchors[:-1], anchors[1:]
        curves = np.stack(
            [starts, (2 * starts + ends) / 3, (starts + 2 * ends) / 3, ends], axis=1
        )
        return cls(id, points=curves.reshape(-1, 2), **kwargs)

    @property
    def num_curves(self) -> int:
        return len(self.points) // POINTS_PER_CURVE

    @property
    def curves(self) -> np.ndarray:
        """Control points per curve, shape ``(num_curves, 4, 2)``."""
        return self.points.reshape(-1, POINTS_PER_CURVE, 2)

    def subpaths(self) -> list[np.ndarray]:
        """Split :attr:`points` into continuous subpaths (same layout)."""
        curves = self.curves
        if len(curves) == 0:
            return []
        gaps = np.linalg.norm(curves[1:, 0] - curves[:-1, -1], axis=1)
        breaks = np.flatnonzero(gaps > TOLERANCE_FOR_POINT_EQUALITY) + 1
        return [part.reshape(-1, 2) for part in np.split(curves, breaks)]

    def stroke_rgba(self) -> np.ndarray:
        """Stroke RGBA per control point, shape ``(N, 4)``."""
        return self.stroke.rgba(len(self.points))

    def stroke_widths(self) -> np.ndarray:
        """Stroke width per control point, shape ``(N,)``."""
        return self.stroke.widths(len(self.points))

    def stroke_joints(self) -> list[JointType]:
        """Joint type at the end of each curve."""
        return self.stroke.joints(self.num_curves)

    def fill_rgba(self) -> np.ndarray:
        """Fill RGBA per control point, shape ``(N, 4)``."""
        return self.fill.rgba(len(self.points))

    def points_3d(self) -> np.ndarray:
        """Local points lifted to 3D on the ``z = 0`` plane."""
        return np.column_stack([self.points, np.zeros(len(self.points))])


def _validated_points(value: object) -> np.ndarray:
    """A read-only float64 ``(N, 2)`` copy, N a multiple of four."""
    points = np.array(value, dtype=np.float64)
    if points.size == 0:
        points = _empty_points()
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if len(points) % POINTS_PER_CURVE != 0:
        raise ValueError(
            "points need four entries (anchor, handle, handle, anchor) "
            f"per cubic Bezier curve, got {len(points)}"
        )
    points.setflags(write=False)
    return points


def is_closed(subpath: np.ndarray) -> bool:
    return bool(
        len(subpath) > 0
        and np.linalg.norm(subpath[0] - subpath[-1]) <= TOLERANCE_FOR_POINT_EQUALITY
    )
