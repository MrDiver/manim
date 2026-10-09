"""Fill and stroke style for the scene nodes that have them.

Shape2D and Curve3D keep every field of the render side's Fill and Stroke
as a signal (see their modules). These functions turn the current signal
values into those Fill and Stroke values for a snapshot, and parse what
authoring code passes in. They are functions, not a shared base class:
Shape2D and Curve3D stay separate classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from ..robjects import CapStyle, Fill, JointType, Stroke
from ..utils.color import ManimColor, ParsableManimColor

Colors = ParsableManimColor | Sequence[ParsableManimColor]
"""One color or a list of colors (a gradient along the path)."""
Scalars = float | Sequence[float]
"""One number or a list of numbers (spread along the path)."""


def colors(value: Colors) -> np.ndarray:
    """One color or a list of colors as a read-only ``(K, 3)`` sRGB array."""
    items = [value] if _is_one_color(value) else list(value)  # type: ignore[arg-type]
    if not items:
        raise ValueError("a color list needs at least one color")
    array = np.array([ManimColor.parse(item).to_rgb() for item in items], dtype=np.float64)
    array.setflags(write=False)
    return array


def joints(value: JointType | Sequence[JointType]) -> JointType | tuple[JointType, ...]:
    """One joint type, or a tuple of them (one per curve, spread like a
    Stroke's joint list)."""
    if isinstance(value, JointType):
        return value
    values = tuple(value)
    if not values or not all(isinstance(joint, JointType) for joint in values):
        raise TypeError(f"joint must be a JointType or a list of them, got {value!r}")
    return values


def cap(value: CapStyle) -> CapStyle:
    if not isinstance(value, CapStyle):
        raise TypeError(f"cap must be a CapStyle, got {value!r}")
    return value


def fill(color: Any, opacity: Any, alpha: float) -> Fill:
    """The Fill for signal values ``color`` (COLORS) and ``opacity``
    (SCALARS), with ``opacity`` multiplied by the enclosing ``alpha``."""
    return Fill(_color_argument(color), opacity=_unit(opacity, alpha))


def stroke(color: Any, opacity: Any, width: Any, joint: Any, cap: CapStyle, alpha: float) -> Stroke:
    """The Stroke for the stroke signal values, with ``opacity`` multiplied
    by the enclosing ``alpha`` and widths kept non-negative."""
    return Stroke(_color_argument(color), opacity=_unit(opacity, alpha), width=_non_negative(width), joint=joint, cap=cap)


def _is_one_color(value: Any) -> bool:
    if isinstance(value, (str, ManimColor)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    return isinstance(value, Sequence) and len(value) in (3, 4) and all(isinstance(c, Real) for c in value)


def _color_argument(value: Any) -> np.ndarray | list[tuple[float, ...]]:
    """What Fill and Stroke take: a ``(3,)`` float array for one color, so
    they skip parsing (the fast path), else a list of RGB float tuples
    (their parser reads a list as colors, but a 2D array as one color)."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return array
    if len(array) == 1:
        return array[0]
    return [tuple(row) for row in array.tolist()]


def _unit(value: Any, alpha: float) -> float | np.ndarray:
    """``value * alpha`` clamped to [0, 1]; plain floats for single values
    (Fill and Stroke check those without numpy)."""
    if isinstance(value, Real):
        return min(1.0, max(0.0, float(value) * alpha))
    if len(value) == 1:
        return min(1.0, max(0.0, float(value[0]) * alpha))
    return np.clip(np.asarray(value, dtype=np.float64) * alpha, 0.0, 1.0)


def _non_negative(value: Any) -> float | np.ndarray:
    if isinstance(value, Real):
        return max(0.0, float(value))
    if len(value) == 1:
        return max(0.0, float(value[0]))
    return np.maximum(np.asarray(value, dtype=np.float64), 0.0)


__all__ = ["Colors", "Scalars", "cap", "colors", "fill", "joints", "stroke"]
