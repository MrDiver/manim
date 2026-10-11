"""Fill and stroke style for the scene nodes that have them.

Shape2D and Curve3D keep every field of the render side's Fill and Stroke
as a signal (see their modules). These functions parse what authoring
code passes in. They are functions, not a shared base class: Shape2D and
Curve3D stay separate classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from ...robjects import CapStyle, JointType
from ...utils.color import ManimColor, ParsableManimColor

Colors = ParsableManimColor | Sequence[ParsableManimColor]
"""One color or a list of colors (a gradient along the path)."""
Scalars = float | Sequence[float]
"""One number or a list of numbers (spread along the path)."""


def colors(value: Colors) -> np.ndarray:
    """One color or a list of colors as a read-only ``(K, 3)`` sRGB array."""
    items = [value] if _is_one_color(value) else list(value)  # type: ignore[arg-type]
    if not items:
        raise ValueError("a color list needs at least one color")
    array = np.array([_one_color(item).to_rgb() for item in items], dtype=np.float64)
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


def _one_color(value: Any) -> ManimColor:
    # ManimColor.parse reads any sequence as a list of colors, so a color
    # given as numbers goes to the constructor (which takes a list).
    if isinstance(value, (str, int, ManimColor)):
        return ManimColor.parse(value)
    return ManimColor(list(value))


def _is_one_color(value: Any) -> bool:
    if isinstance(value, (str, ManimColor)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    return isinstance(value, Sequence) and len(value) in (3, 4) and all(isinstance(c, Real) for c in value)


__all__ = ["Colors", "Scalars", "cap", "colors", "joints"]
