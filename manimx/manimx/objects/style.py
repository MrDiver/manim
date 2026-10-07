from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np

from ..utils.color import WHITE, ManimColor, ParsableManimColor

ColorsLike: TypeAlias = ParsableManimColor | Sequence[ParsableManimColor]
ScalarsLike: TypeAlias = float | Sequence[float]

DEFAULT_STROKE_WIDTH = 4.0


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def to_rgb_array(colors: ColorsLike) -> np.ndarray:
    """Parse one color or a list/tuple of colors into a ``(K, 3)`` array.

    Follows :meth:`ManimColor.parse`: a list or tuple is always a sequence of
    colors, so pass a single RGB triple as a ``ManimColor`` or numpy array.
    """
    parsed = ManimColor.parse(colors)
    if isinstance(parsed, ManimColor):
        parsed = [parsed]
    if len(parsed) == 0:
        raise ValueError("expected at least one color")
    return _frozen(np.array([color.to_rgb() for color in parsed], dtype=np.float64))


def to_scalar_array(
    values: ScalarsLike, name: str, low: float, high: float
) -> np.ndarray:
    """Parse one value or a sequence of values into a ``(K,)`` array."""
    array = np.array(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} needs at least one value")
    if np.any(array < low) or np.any(array > high):
        raise ValueError(f"{name} must be in [{low}, {high}], got {values!r}")
    return _frozen(array)


def resample(values: np.ndarray, count: int) -> np.ndarray:
    """Spread ``K`` values evenly over ``count`` slots by linear interpolation.

    One value is broadcast and ``K == count`` is returned unchanged. Any other
    ``K`` is stretched like Manim's color gradients, so a short list of colors
    becomes a smooth gradient along the path.
    """
    if len(values) == count:
        return values
    if len(values) == 1:
        return np.repeat(values, count, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    if values.ndim == 1:
        return np.interp(target, source, values)
    return np.column_stack(
        [np.interp(target, source, channel) for channel in values.T]
    )


class JointType(Enum):
    """Port of ``manim.constants.LineJointType``."""

    AUTO = 0
    ROUND = 1
    BEVEL = 2
    MITER = 3


class CapStyle(Enum):
    """Port of ``manim.constants.CapStyleType``."""

    AUTO = 0
    ROUND = 1
    BUTT = 2
    SQUARE = 3


@dataclass(frozen=True, slots=True, eq=False)
class Fill:
    """Fill style. ``color`` and ``opacity`` take one value or a list.

    Lists are spread over the shape's control points with :func:`resample`.
    """

    color: ColorsLike = WHITE
    opacity: ScalarsLike = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", to_rgb_array(self.color))
        object.__setattr__(
            self, "opacity", to_scalar_array(self.opacity, "opacity", 0.0, 1.0)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fill):
            return NotImplemented
        return np.array_equal(self.color, other.color) and np.array_equal(
            self.opacity, other.opacity
        )

    def rgba(self, count: int) -> np.ndarray:
        """Per-control-point RGBA, shape ``(count, 4)``."""
        return np.column_stack(
            [resample(self.color, count), resample(self.opacity, count)]
        )


@dataclass(frozen=True, slots=True, eq=False)
class Stroke:
    """Stroke style; every field except ``cap`` takes one value or a list.

    ``color``, ``opacity`` and ``width`` lists are spread over the shape's
    control points with :func:`resample`, so they interpolate along the curve
    (e.g. a rainbow stroke or a tapering width). ``joint`` lists are spread
    over the curves with nearest-neighbour lookup: entry ``i`` is the joint at
    the end of curve ``i``.
    """

    color: ColorsLike = WHITE
    opacity: ScalarsLike = 1.0
    width: ScalarsLike = DEFAULT_STROKE_WIDTH
    joint: JointType | Sequence[JointType] = JointType.AUTO
    cap: CapStyle = CapStyle.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", to_rgb_array(self.color))
        object.__setattr__(
            self, "opacity", to_scalar_array(self.opacity, "opacity", 0.0, 1.0)
        )
        object.__setattr__(
            self, "width", to_scalar_array(self.width, "width", 0.0, np.inf)
        )
        joints = (
            (self.joint,) if isinstance(self.joint, JointType) else tuple(self.joint)
        )
        if len(joints) == 0:
            raise ValueError("joint needs at least one value")
        object.__setattr__(self, "joint", joints)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stroke):
            return NotImplemented
        return (
            np.array_equal(self.color, other.color)
            and np.array_equal(self.opacity, other.opacity)
            and np.array_equal(self.width, other.width)
            and self.joint == other.joint
            and self.cap == other.cap
        )

    def rgba(self, count: int) -> np.ndarray:
        """Per-control-point RGBA, shape ``(count, 4)``."""
        return np.column_stack(
            [resample(self.color, count), resample(self.opacity, count)]
        )

    def widths(self, count: int) -> np.ndarray:
        """Per-control-point width, shape ``(count,)``."""
        return resample(self.width, count)

    def joints(self, num_curves: int) -> list[JointType]:
        """Joint type at the end of each curve."""
        indices = np.rint(np.linspace(0, len(self.joint) - 1, num_curves)).astype(int)
        return [self.joint[i] for i in indices]
