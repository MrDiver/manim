"""Path styles: joint and cap styles (``JointType``, ``CapStyle``), and how
``Fill`` and ``Stroke`` read the colors and numbers users give them
(:func:`to_rgb_array`, :func:`to_scalar_array`)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TypeAlias

import numpy as np

from ..utils.color import ManimColor, ParsableManimColor

ColorsLike: TypeAlias = ParsableManimColor | Sequence[ParsableManimColor]
ScalarsLike: TypeAlias = float | Sequence[float]

STROKE_WIDTH_SCALE = 0.01
"""World units per unit of stroke width (Manim's: width 4 is 0.04 units)."""


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def to_rgb_array(colors: ColorsLike) -> np.ndarray:
    """Parse one color or a list/tuple of colors into a ``(K, 3)`` array.

    Follows :meth:`ManimColor.parse`: a list or tuple is always a sequence of
    colors, so pass a single RGB triple as a ``ManimColor`` or numpy array.
    """
    if isinstance(colors, np.ndarray) and colors.dtype.kind == "f" and colors.shape == (3,):
        # One RGB float triple (what animated color signals hold): parsing
        # returns it unchanged, so skip the parser.
        return _frozen(colors.astype(np.float64).reshape(1, 3))
    parsed = ManimColor.parse(colors)
    if isinstance(parsed, ManimColor):
        parsed = [parsed]
    if len(parsed) == 0:
        raise ValueError("expected at least one color")
    return _frozen(np.array([color.to_rgb() for color in parsed], dtype=np.float64))


def to_scalar_array(values: ScalarsLike, name: str, low: float, high: float) -> np.ndarray:
    """Parse one value or a sequence of values into a ``(K,)`` array."""
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        if values < low or values > high:
            raise ValueError(f"{name} must be in [{low}, {high}], got {values!r}")
        return _frozen(np.array((values,), dtype=np.float64))
    array = np.array(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} needs at least one value")
    if np.any(array < low) or np.any(array > high):
        raise ValueError(f"{name} must be in [{low}, {high}], got {values!r}")
    return _frozen(array)


class JointType(Enum):
    """How a stroke turns a corner between two curves."""

    AUTO = 0
    ROUND = 1
    BEVEL = 2
    MITER = 3


class CapStyle(Enum):
    """How a stroke ends."""

    AUTO = 0
    ROUND = 1
    BUTT = 2
    SQUARE = 3


__all__ = ["CapStyle", "ColorsLike", "JointType", "STROKE_WIDTH_SCALE", "ScalarsLike", "to_rgb_array", "to_scalar_array"]
