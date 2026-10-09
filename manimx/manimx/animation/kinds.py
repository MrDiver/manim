"""Signal kinds: how values of one type interpolate and combine.

Every signal has a kind. Animations never touch values directly; they ask
the kind to

* ``interpolate(start, end, alpha)`` for absolute changes ("move to"),
* ``combine(value, change)`` for relative changes ("shift by"),
* ``scale(change, alpha)`` to apply a fraction of a relative change,

and the signal asks ``equal(a, b)`` to decide whether a recomputed value is
new (which bumps its version). See section 3.1 of
docs/animation_engine_design.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import quaternion


class SignalKind(ABC):
    """Operations on the values of one signal type."""

    name: str = "value"

    def validate(self, value: Any) -> Any:
        """Normalize a value before it is stored (and reject invalid ones)."""
        return value

    @abstractmethod
    def interpolate(self, start: Any, end: Any, alpha: float) -> Any:
        """Value ``alpha`` of the way from ``start`` to ``end``."""

    def combine(self, value: Any, change: Any) -> Any:
        """``value`` changed by a relative ``change``."""
        raise TypeError(f"{self.name} signals do not support relative changes")

    def scale(self, change: Any, alpha: float) -> Any:
        """The fraction ``alpha`` of a relative ``change``."""
        raise TypeError(f"{self.name} signals do not support relative changes")

    def equal(self, a: Any, b: Any) -> bool:
        return bool(a == b)


class ScalarKind(SignalKind):
    """Floats: positions per axis, opacity, widths. Changes add."""

    name = "scalar"

    def validate(self, value: Any) -> float:
        return float(value)

    def interpolate(self, start: float, end: float, alpha: float) -> float:
        return start + (end - start) * alpha

    def combine(self, value: float, change: float) -> float:
        return value + change

    def scale(self, change: float, alpha: float) -> float:
        return change * alpha


class FactorKind(ScalarKind):
    """Positive scale factors. Relative changes multiply ("scale by 2" twice
    is 4), and a fraction of a change is geometric: half of "by 4" is
    "by 2", so a scale animation looks uniform in speed."""

    name = "factor"

    def combine(self, value: float, change: float) -> float:
        return value * change

    def scale(self, change: float, alpha: float) -> float:
        if change <= 0.0:
            raise ValueError(f"scale factors must be positive, got {change}")
        return change**alpha


class RotationKind(SignalKind):
    """Unit quaternions. Interpolation follows the shortest arc (slerp);
    relative changes compose (``change * value``, i.e. applied after)."""

    name = "rotation"

    def validate(self, value: Any) -> quaternion.quaternion:
        if not isinstance(value, quaternion.quaternion):
            raise TypeError(f"rotation must be a quaternion, got {type(value).__name__}")
        norm = abs(value)
        if norm == 0.0:
            raise ValueError("rotation quaternion must not be zero")
        return value / norm

    def interpolate(self, start: quaternion.quaternion, end: quaternion.quaternion, alpha: float) -> quaternion.quaternion:
        if np.dot(start.components, end.components) < 0.0:
            end = -end  # q and -q are the same rotation: take the short way
        return quaternion.slerp_evaluate(start, end, alpha)

    def combine(self, value: quaternion.quaternion, change: quaternion.quaternion) -> quaternion.quaternion:
        return change * value

    def scale(self, change: quaternion.quaternion, alpha: float) -> quaternion.quaternion:
        return self.interpolate(quaternion.one, change, alpha)

    def equal(self, a: quaternion.quaternion, b: quaternion.quaternion) -> bool:
        return bool(np.array_equal(a.components, b.components))


class ColorKind(SignalKind):
    """RGB(A) colors as float tuples in [0, 1], interpolated per channel
    (Manim interpolates colors in RGB). Changes add and clamp."""

    name = "color"

    def validate(self, value: Any) -> tuple[float, ...]:
        channels = tuple(float(c) for c in value)
        if len(channels) not in (3, 4):
            raise ValueError(f"a color needs 3 or 4 channels, got {len(channels)}")
        return channels

    def interpolate(self, start: tuple[float, ...], end: tuple[float, ...], alpha: float) -> tuple[float, ...]:
        if len(start) != len(end):
            raise ValueError("cannot interpolate colors with different channel counts")
        return tuple(a + (b - a) * alpha for a, b in zip(start, end))

    def combine(self, value: tuple[float, ...], change: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(min(1.0, max(0.0, a + b)) for a, b in zip(value, change))

    def scale(self, change: tuple[float, ...], alpha: float) -> tuple[float, ...]:
        return tuple(c * alpha for c in change)


def _spread(values: np.ndarray, count: int) -> np.ndarray:
    """``values`` (K entries along axis 0) spread evenly over ``count``
    entries by linear interpolation, like the renderer spreads a style
    list over a path's points (robjects.style.resample)."""
    if len(values) == count:
        return values
    if len(values) == 1:
        return np.repeat(values, count, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    if values.ndim == 1:
        return np.interp(target, source, values)
    return np.column_stack([np.interp(target, source, channel) for channel in values.T])


def _matched(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both lists spread to the longer length, so lists of different
    lengths interpolate smoothly (a two-color gradient into a five-color
    one) instead of jumping."""
    count = max(len(start), len(end))
    return _spread(start, count), _spread(end, count)


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class ColorsKind(SignalKind):
    """One RGB color or a list of them, as a read-only ``(K, 3)`` float
    array (K >= 1). This is what a Fill or Stroke color holds: a list is
    spread over the path's points as a gradient. Interpolation is per
    channel after spreading both lists to the same length; changes add and
    clamp, like COLOR."""

    name = "colors"

    def validate(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray) and value.dtype == np.float64 and value.ndim == 2 and not value.flags.writeable:
            return value  # already validated (e.g. an interpolated value)
        array = np.array(value, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
            raise ValueError(f"colors must be one RGB color or a list of them, got shape {array.shape}")
        return _frozen(array)

    def interpolate(self, start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
        start, end = _matched(start, end)
        return _frozen(start + (end - start) * alpha)

    def combine(self, value: np.ndarray, change: Any) -> np.ndarray:
        value, change = _matched(value, self.validate(change))
        return _frozen(np.clip(value + change, 0.0, 1.0))

    def scale(self, change: Any, alpha: float) -> np.ndarray:
        return _frozen(np.asarray(change, dtype=np.float64) * alpha)

    def equal(self, a: np.ndarray, b: np.ndarray) -> bool:
        return a is b or (a.shape == b.shape and bool(np.array_equal(a, b)))


class ScalarsKind(SignalKind):
    """One float or a list of them, as a read-only ``(K,)`` float array
    (K >= 1): opacity and width of a Fill or Stroke, where a list is spread
    over the path's points (e.g. a tapering width). Interpolation after
    spreading both lists to the same length; changes add, like SCALAR."""

    name = "scalars"

    def validate(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray) and value.dtype == np.float64 and value.ndim == 1 and not value.flags.writeable:
            return value
        array = np.atleast_1d(np.array(value, dtype=np.float64))
        if array.ndim != 1 or len(array) == 0:
            raise ValueError(f"scalars must be one number or a list of them, got shape {array.shape}")
        return _frozen(array)

    def interpolate(self, start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
        start, end = _matched(start, end)
        return _frozen(start + (end - start) * alpha)

    def combine(self, value: np.ndarray, change: Any) -> np.ndarray:
        value, change = _matched(value, self.validate(change))
        return _frozen(value + change)

    def scale(self, change: Any, alpha: float) -> np.ndarray:
        return _frozen(np.asarray(change, dtype=np.float64) * alpha)

    def equal(self, a: np.ndarray, b: np.ndarray) -> bool:
        return a is b or (a.shape == b.shape and bool(np.array_equal(a, b)))


class PathKind(SignalKind):
    """Point arrays (Shape2D or Curve3D points). Interpolation is point-wise
    and needs equal shapes; aligning different paths (Manim's
    ``align_points``) happens before, when a morph is compiled. Changes add
    point-wise (e.g. a translation broadcast over all points)."""

    name = "path"

    def validate(self, value: Any) -> np.ndarray:
        array = np.array(value, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"a path must be a 2D point array, got shape {array.shape}")
        array.setflags(write=False)
        return array

    def interpolate(self, start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
        if start.shape != end.shape:
            raise ValueError(f"cannot interpolate paths of shapes {start.shape} and {end.shape}; align them first")
        return self.validate(start + (end - start) * alpha)

    def combine(self, value: np.ndarray, change: Any) -> np.ndarray:
        return self.validate(value + np.asarray(change))

    def scale(self, change: Any, alpha: float) -> np.ndarray:
        return np.asarray(change) * alpha

    def equal(self, a: np.ndarray, b: np.ndarray) -> bool:
        return a is b or (a.shape == b.shape and bool(np.array_equal(a, b)))


class ArrayKind(PathKind):
    """Per-element float arrays of shape ``(N, K)`` (the positions, scales
    or colors of N particles), interpolated element-wise like paths."""

    name = "array"


class DiscreteKind(SignalKind):
    """Values without in-between states (visibility, joint types, meshes):
    an absolute change switches at the end of the animation."""

    name = "discrete"

    def interpolate(self, start: Any, end: Any, alpha: float) -> Any:
        return end if alpha >= 1.0 else start

    def equal(self, a: Any, b: Any) -> bool:
        try:
            return bool(a == b)
        except (TypeError, ValueError):  # e.g. numpy arrays
            return a is b


SCALAR = ScalarKind()
FACTOR = FactorKind()
ROTATION = RotationKind()
COLOR = ColorKind()
COLORS = ColorsKind()
SCALARS = ScalarsKind()
PATH = PathKind()
ARRAY = ArrayKind()
DISCRETE = DiscreteKind()
