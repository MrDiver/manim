"""Shared base of all objects made of cubic Bezier curves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np

from .render_object import RenderObject
from .style import JointType, Stroke
from .versioning import versioned

POINTS_PER_CURVE = 4
TOLERANCE_FOR_POINT_EQUALITY = 1e-6


def _no_points() -> np.ndarray:
    return np.zeros((0, 0))  # replaced by the right width in __setattr__


@dataclass(slots=True, eq=False)
class BezierPath(RenderObject):
    """Points in Manim's ``VMobject`` layout plus a stroke.

    ``points`` is ``(N, DIMENSION)`` with four points per cubic Bezier curve
    ``(anchor, handle, handle, anchor)``. A new subpath starts wherever a
    curve's start differs from the previous curve's end.

    ``points`` and ``stroke`` are versioned: replace them instead of mutating
    them (``points`` is stored read-only to enforce this).
    """

    DIMENSION: ClassVar[int] = 2

    points: np.ndarray = versioned(default_factory=_no_points)
    stroke: Stroke = versioned(default_factory=Stroke)
    cast_shadows: bool = True
    """Whether this path blocks light for lit meshes behind it."""

    def __setattr__(self, name: str, value: object) -> None:
        # Validate on every assignment, including the one in __init__.
        if name == "points":
            value = _validated_points(value, self.DIMENSION)
        RenderObject.__setattr__(self, name, value)

    @classmethod
    def from_corners(
        cls,
        corners: Sequence[Sequence[float]],
        *,
        closed: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Polyline through ``corners`` as straight cubic curves.

        Mirrors ``VMobject.set_points_as_corners``.
        """
        anchors = np.asarray(corners, dtype=np.float64).reshape(-1, cls.DIMENSION)
        if closed and len(anchors) > 0:
            anchors = np.vstack([anchors, anchors[:1]])
        starts, ends = anchors[:-1], anchors[1:]
        curves = np.stack(
            [starts, (2 * starts + ends) / 3, (starts + 2 * ends) / 3, ends], axis=1
        )
        return cls(points=curves.reshape(-1, cls.DIMENSION), **kwargs)

    @property
    def num_curves(self) -> int:
        return len(self.points) // POINTS_PER_CURVE

    @property
    def curves(self) -> np.ndarray:
        """Control points per curve, shape ``(num_curves, 4, DIMENSION)``."""
        return self.points.reshape(-1, POINTS_PER_CURVE, self.DIMENSION)

    def subpaths(self) -> list[np.ndarray]:
        """Split :attr:`points` into continuous subpaths (same layout)."""
        curves = self.curves
        if len(curves) == 0:
            return []
        gaps = np.linalg.norm(curves[1:, 0] - curves[:-1, -1], axis=1)
        breaks = np.flatnonzero(gaps > TOLERANCE_FOR_POINT_EQUALITY) + 1
        return [part.reshape(-1, self.DIMENSION) for part in np.split(curves, breaks)]

    def stroke_rgba(self) -> np.ndarray:
        """Stroke RGBA per control point, shape ``(N, 4)``."""
        return self.stroke.rgba(len(self.points))

    def stroke_widths(self) -> np.ndarray:
        """Stroke width per control point, shape ``(N,)``."""
        return self.stroke.widths(len(self.points))

    def stroke_joints(self) -> list[JointType]:
        """Joint type at the end of each curve."""
        return self.stroke.joints(self.num_curves)

    def points_3d(self) -> np.ndarray:
        """Local points in 3D (2D paths lie on the ``z = 0`` plane)."""
        if self.DIMENSION == 3:
            return self.points
        return np.column_stack([self.points, np.zeros(len(self.points))])


def _validated_points(value: object, dimension: int) -> np.ndarray:
    """A read-only float64 ``(N, dimension)`` array, N a multiple of four.

    An array that already is all of that, and owns its read-only data
    (e.g. a path signal's value), is shared instead of copied: nobody can
    change it. Snapshots rebuild render objects every frame, and copying
    unchanged points each time was measurable.
    """
    if (
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.ndim == 2
        and value.shape[1] == dimension
        and len(value) % POINTS_PER_CURVE == 0
        and value.flags.owndata
        and not value.flags.writeable
    ):
        return value
    points = np.array(value, dtype=np.float64)
    if points.size == 0:
        points = np.zeros((0, dimension))
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(f"points must have shape (N, {dimension}), got {points.shape}")
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
