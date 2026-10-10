"""Number lines, axes, planes and the graphs of functions on them.

A NumberLine's range (``minimum``, ``maximum``, ``step``) is made of
signals, and everything on it is derived from them: the ticks, the
labels (their values, places and whether they show) and, on Axes, the
graphs of functions and the points of coordinates. So animating a range
zooms: ticks, labels and plots follow at every frame::

    axes = Axes((0, 5, 1), (0, 25, 5), include_numbers=True)
    graph = axes.plot(lambda x: x**2)
    self.play(axes.animate.set_ranges(x=(0, 10, 1), y=(0, 100, 20)))

Labels are DecimalNumbers from a pool: as many as the widest range the
line is given needs (set_range grows it while the scene is composed), the
ones beyond the range hidden. Graphs and points are in the space of the
Axes' parent, so they follow the Axes' own moves too; put them next to
it. No class picks a color.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import quaternion

from ..animation import (
    ARRAY,
    DISCRETE,
    SCALAR,
    DerivedSignal,
    Group,
    Node,
    RootSignal,
    Shape2D,
    Signal,
)
from ..animation.authoring import current
from ..animation.node import ABS, ChangeList, animatable
from ..animation.paths import PointSignal
from ..animation.signals import derive
from .geometry import corner_points
from .numbers import DecimalNumber
from .shapes2d import parameter

EPSILON = 1e-9

Range = Sequence[float]
"""``(minimum, maximum)`` or ``(minimum, maximum, step)``."""


class Value(Node):
    """A number to animate that is not drawn (Manim's ValueTracker): derive
    other signals from ``value``. ``self.play(t.animate.set_value(1))``."""

    def __init__(self, value: float = 0.0, name: str | None = None) -> None:
        super().__init__(name)
        self.value = RootSignal(float(value), SCALAR, name=f"{self.name}.value")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.value

    @animatable
    def set_value(self, value: float) -> ChangeList:
        return [(self.value, ABS, float(value))]

    @animatable
    def increment_value(self, amount: float) -> ChangeList:
        return [(self.value, ABS, current(self.value) + float(amount))]

    def _build_snapshot(self) -> None:
        return None


class NumberLine(Group):
    """A line ``length`` long showing ``x_range`` (minimum, maximum, step),
    centered on its origin and turned by ``angle``: a Group of ``line``,
    ``ticks`` (one path, a tick at every multiple of the step) and, with
    ``include_numbers``, ``numbers`` (DecimalNumbers with ``decimals``
    places set with ``preamble``, ``label_buff`` beyond the side
    ``label_direction``, default below a horizontal line). Values in
    ``exclude`` get no label. ``style`` is the line's and ticks'.

    The step is DISCRETE: animating it jumps at the end, so ticks are
    always on its multiples. Labels thin out to every 2nd, 5th, 10th, ...
    tick while they would be closer than ``label_spacing`` (as during a
    zoom), so they always show round values.
    """

    def __init__(
        self,
        x_range: Range = (-5.0, 5.0, 1.0),
        length: float = 10.0,
        angle: float = 0.0,
        tick_size: float = 0.1,
        include_numbers: bool = False,
        decimals: int = 0,
        preamble: str = "",
        label_direction: Sequence[float] | None = None,
        label_buff: float = 0.25,
        label_spacing: float = 0.6,
        exclude: Sequence[float] = (),
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        minimum, maximum, step = _range(x_range)
        self.label_spacing = float(label_spacing)
        self.minimum = parameter(self, "minimum", minimum)
        self.maximum = parameter(self, "maximum", maximum)
        self.step = parameter(self, "step", step, DISCRETE)  # jumps: ticks stay on round values
        self.length, self.angle, self.tick_size = float(length), float(angle), float(tick_size)
        self.direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        normal = np.array([math.sin(self.angle), -math.cos(self.angle)])  # right of the direction: below a horizontal line
        self.label_direction = normal if label_direction is None else _unit(label_direction)
        self.label_buff = float(label_buff)
        self.line = Shape2D(corner_points([-self.direction * self.length / 2, self.direction * self.length / 2]), f"{self.name}.line", **style)
        self.ticks = Shape2D((), f"{self.name}.ticks", **style)
        self.ticks._derive_points(_Ticks(self.length, self.direction, self.tick_size), *self.range_signals)
        self.add(self.line, self.ticks)
        self.include_numbers, self.decimals, self.preamble = include_numbers, int(decimals), preamble
        self.exclude = tuple(float(value) for value in exclude)
        self.number_style = {key: value for key, value in style.items() if key in ("color", "fill_color", "fill_opacity")}
        self.numbers: list[DecimalNumber] = []
        if include_numbers:
            self._grow_numbers(minimum, maximum, step)

    @property
    def range_signals(self) -> tuple[Signal, Signal, Signal]:
        return (self.minimum, self.maximum, self.step)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from self.range_signals

    @animatable
    def set_range(self, minimum: float | None = None, maximum: float | None = None, step: float | None = None) -> ChangeList:
        """Change the range (zoom); labels and everything derived follow."""
        values = [current(signal) if value is None else float(value) for signal, value in zip(self.range_signals, (minimum, maximum, step), strict=True)]
        _range(values)  # checks it
        if self.include_numbers:
            # During the change the step is still the old one (it jumps at the end).
            self._grow_numbers(values[0], values[1], min(values[2], current(self.step)))
            self._grow_numbers(current(self.minimum), current(self.maximum), values[2])
        return [(signal, ABS, value) for signal, value in zip(self.range_signals, values, strict=True)]

    def number_to_point(self, value: float) -> np.ndarray:
        """Where ``value`` is on the line, in its own space, as authoring
        code sees the range right now."""
        return np.append(_place(value, current(self.minimum), current(self.maximum), self.length) * self.direction, 0.0)

    n2p = number_to_point

    def _grow_numbers(self, minimum: float, maximum: float, step: float) -> None:
        """Enough labels for the range (the pool never shrinks); thinned
        labels are never closer than label_spacing, which bounds them."""
        needed = min(int(math.floor((maximum - minimum) / step + EPSILON)), int(self.length / self.label_spacing)) + 2
        while len(self.numbers) < needed:
            index = len(self.numbers)
            value = DerivedSignal(_LabelValue(index, self.length, self.label_spacing), self.range_signals, SCALAR, name=f"{self.name}.number{index}.value")
            number = DecimalNumber(value, self.decimals, self.preamble, characters=7, name=f"{self.name}.number{index}", **self.number_style)
            label = _LabelPlace(index, self.length, self.label_spacing, self.direction, self.label_direction * self.label_buff, self.exclude)
            number.position.x = DerivedSignal(label.x, self.range_signals, SCALAR, name=f"{number.name}.position.x")
            number.position.y = DerivedSignal(label.y, self.range_signals, SCALAR, name=f"{number.name}.position.y")
            number.alpha = DerivedSignal(label.alpha, self.range_signals, SCALAR, name=f"{number.name}.alpha")
            self.numbers.append(number)
            self.add(number)


class Axes(Group):
    """Two NumberLines for ``x_range`` and ``y_range``, ``x_length`` and
    ``y_length`` long, crossing where both are 0 (or the nearest end of a
    range without 0): ``x_axis`` and ``y_axis``.

    :meth:`plot` draws a function's graph and :meth:`point` gives a point
    for coordinates; both follow the ranges and the Axes' own transform.
    ``axis_style`` styles both lines; ``number_style``-like keys (color)
    reach the labels too.
    """

    def __init__(
        self,
        x_range: Range = (-6.0, 6.0, 1.0),
        y_range: Range = (-3.0, 3.0, 1.0),
        x_length: float = 12.0,
        y_length: float = 6.0,
        include_numbers: bool = False,
        decimals: int = 0,
        preamble: str = "",
        tick_size: float = 0.1,
        name: str | None = None,
        **axis_style: Any,
    ) -> None:
        super().__init__(name=name)
        common = {"include_numbers": include_numbers, "decimals": decimals, "preamble": preamble, "tick_size": tick_size, "exclude": (0.0,)}
        self.x_axis = NumberLine(x_range, x_length, name=f"{self.name}.x_axis", **common, **axis_style)
        self.y_axis = NumberLine(y_range, y_length, angle=math.pi / 2, label_direction=(-1.0, 0.0), name=f"{self.name}.y_axis", **common, **axis_style)
        # Each axis sits where the other one's value is 0.
        self.x_axis.position.y = DerivedSignal(_Crossing(y_length), self.y_axis.range_signals, SCALAR, name=f"{self.x_axis.name}.position.y")
        self.y_axis.position.x = DerivedSignal(_Crossing(x_length), self.x_axis.range_signals, SCALAR, name=f"{self.y_axis.name}.position.x")
        self.add(self.x_axis, self.y_axis)

    @property
    def lengths(self) -> tuple[float, float]:
        return (self.x_axis.length, self.y_axis.length)

    def _frame_inputs(self) -> tuple[Signal, ...]:
        """The ranges and the Axes' transform: what maps coordinates to its parent's space."""
        return (
            self.x_axis.minimum, self.x_axis.maximum, self.y_axis.minimum, self.y_axis.maximum,
            *self.position.axes, *self.scale.axes, self.rotation,
        )  # fmt: skip

    @animatable
    def set_ranges(self, x: Range | None = None, y: Range | None = None) -> ChangeList:
        """Change either range (``(minimum, maximum)`` or with a step)."""
        changes: ChangeList = []
        for axis, values in ((self.x_axis, x), (self.y_axis, y)):
            if values is not None:
                changes += NumberLine.set_range._signal_changes(axis, *_range_or_keep(values))  # type: ignore[attr-defined]
        return changes

    def coords_to_point(self, x: float, y: float) -> np.ndarray:
        """Where ``(x, y)`` is, in the Axes' parent's space, as authoring
        code sees the ranges and the Axes right now."""
        values = [current(signal) for signal in self._frame_inputs()]
        return _Frame(self.lengths).place(np.array([[x, y]]), values)[0]

    c2p = coords_to_point

    def point(self, x: float | Signal, y: float | Signal) -> PointSignal:
        """The point of ``(x, y)`` (numbers or signals) as a point that
        follows the ranges and the Axes: ``dot.follow(axes.point(t, f_t))``."""
        x_signal = x if isinstance(x, Signal) else RootSignal(float(x), SCALAR, name=f"{self.name}.point.x")
        y_signal = y if isinstance(y, Signal) else RootSignal(float(y), SCALAR, name=f"{self.name}.point.y")
        return PointSignal(DerivedSignal(_FramePoint(self.lengths), (x_signal, y_signal, *self._frame_inputs()), ARRAY, name=f"{self.name}.point"))

    def plot(
        self,
        function: Callable[[float], float],
        x_range: Range | None = None,
        samples: int = 64,
        clip: bool = True,
        name: str | None = None,
        **style: Any,
    ) -> FunctionGraph:
        """The graph of ``function`` over ``x_range`` (default: the x
        range at every frame), following the ranges and the Axes; with
        ``clip``, only where it is inside the y range (split into pieces
        where it leaves)."""
        graph = FunctionGraph(function, x_range or (-1.0, 1.0), samples, name, **style)
        fixed = None if x_range is None else tuple(float(value) for value in x_range[:2])
        graph._derive_points(_Plot(function, fixed, int(samples), bool(clip), self.lengths), *self._frame_inputs())
        return graph

    def graph_point(self, x: Signal, function: Callable[[float], float]) -> PointSignal:
        """The point ``(x, function(x))`` for a signal ``x``: a dot that
        rides a graph, ``dot.follow(axes.graph_point(t.value, f))``."""
        return self.point(x, derive(function, x, name=f"{self.name}.graph_y"))


class NumberPlane(Axes):
    """Axes over a grid: ``grid`` holds a line at every tick of both
    ranges (one path), styled by ``grid_style``, drawn under the axes."""

    def __init__(
        self,
        x_range: Range = (-7.0, 7.0, 1.0),
        y_range: Range = (-4.0, 4.0, 1.0),
        x_length: float = 14.0,
        y_length: float = 8.0,
        grid_style: dict[str, Any] | None = None,
        name: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(x_range, y_range, x_length, y_length, name=name, **options)
        self.grid = Shape2D((), f"{self.name}.grid", **(grid_style or {}))
        self.grid._derive_points(_Grid(self.lengths), *self.x_axis.range_signals, *self.y_axis.range_signals)
        self.children.insert(0, self.grid)  # under the axes
        self.add()  # tells the structure listeners about it


class FunctionGraph(Shape2D):
    """The graph of ``function`` over ``x_range`` in plain coordinates
    (x and y as they are), from ``samples`` curves; ``minimum`` and
    ``maximum`` are signals. On Axes, use :meth:`Axes.plot`."""

    def __init__(self, function: Callable[[float], float], x_range: Range = (-8.0, 8.0), samples: int = 64, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.function = function
        self.minimum = parameter(self, "minimum", float(x_range[0]))
        self.maximum = parameter(self, "maximum", float(x_range[1]))
        self._derive_points(_Curve(lambda t, f=function: (t, f(t)), int(samples)), self.minimum, self.maximum)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.minimum, self.maximum)

    @animatable
    def set_x_range(self, minimum: float | None = None, maximum: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if minimum is not None:
            changes.append((self.minimum, ABS, float(minimum)))
        if maximum is not None:
            changes.append((self.maximum, ABS, float(maximum)))
        return changes


class ParametricFunction(Shape2D):
    """The 2D curve ``function(t)`` for ``t`` over ``t_range`` (``minimum``
    and ``maximum`` are signals), from ``samples`` curves."""

    def __init__(self, function: Callable[[float], Sequence[float]], t_range: Range = (0.0, 1.0), samples: int = 64, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.function = function
        self.minimum = parameter(self, "minimum", float(t_range[0]))
        self.maximum = parameter(self, "maximum", float(t_range[1]))
        self._derive_points(_Curve(function, int(samples)), self.minimum, self.maximum)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.minimum, self.maximum)

    @animatable
    def set_t_range(self, minimum: float | None = None, maximum: float | None = None) -> ChangeList:
        return FunctionGraph.set_x_range._signal_changes(self, minimum, maximum)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------


def _place(value: float, minimum: float, maximum: float, length: float) -> float:
    """How far along a line centered on its origin ``value`` is."""
    span = maximum - minimum
    return ((value - minimum) / span - 0.5) * length if span else 0.0


def _first_tick(minimum: float, step: float) -> float:
    return math.ceil(minimum / step - EPSILON) * step


class _Ticks:
    __slots__ = ("length", "direction", "size")

    def __init__(self, length: float, direction: np.ndarray, size: float) -> None:
        self.length, self.direction, self.size = length, direction, size

    def __call__(self, minimum: float, maximum: float, step: float) -> np.ndarray:
        normal = np.array([-self.direction[1], self.direction[0]]) * self.size
        pieces = []
        value = _first_tick(minimum, step)
        while value <= maximum + EPSILON * max(1.0, abs(maximum)) and len(pieces) < 10_000:
            center = _place(value, minimum, maximum, self.length) * self.direction
            pieces.append(corner_points([center - normal, center + normal]))
            value += step
        return np.concatenate(pieces) if pieces else np.zeros((0, 2))


def _label_step(minimum: float, maximum: float, step: float, length: float, spacing: float) -> float:
    """The step between labels: the step times the first of 1, 2, 5, 10,
    20, ... that keeps labels ``spacing`` apart on a line ``length`` long."""
    per_unit = length / ((maximum - minimum) or 1.0)
    factor = 1.0
    for _ in range(60):
        for multiple in (1.0, 2.0, 5.0):
            if step * factor * multiple * per_unit >= spacing - EPSILON:
                return step * factor * multiple
        factor *= 10.0
    return step * factor


class _LabelValue:
    __slots__ = ("index", "length", "spacing")

    def __init__(self, index: int, length: float, spacing: float) -> None:
        self.index, self.length, self.spacing = index, length, spacing

    def __call__(self, minimum: float, maximum: float, step: float) -> float:
        label_step = _label_step(minimum, maximum, step, self.length, self.spacing)
        value = _first_tick(minimum, label_step) + self.index * label_step
        return 0.0 if abs(value) < EPSILON * max(1.0, label_step) else value  # no -0


class _LabelPlace:
    """Where a pooled label goes and whether it shows."""

    __slots__ = ("value", "length", "direction", "offset", "exclude")

    def __init__(self, index: int, length: float, spacing: float, direction: np.ndarray, offset: np.ndarray, exclude: tuple[float, ...]) -> None:
        self.value = _LabelValue(index, length, spacing)
        self.length, self.direction, self.offset, self.exclude = length, direction, offset, exclude

    def _value(self, minimum: float, maximum: float, step: float) -> float:
        return self.value(minimum, maximum, step)

    def _point(self, minimum: float, maximum: float, step: float) -> np.ndarray:
        return _place(self._value(minimum, maximum, step), minimum, maximum, self.length) * self.direction + self.offset

    def x(self, minimum: float, maximum: float, step: float) -> float:
        return float(self._point(minimum, maximum, step)[0])

    def y(self, minimum: float, maximum: float, step: float) -> float:
        return float(self._point(minimum, maximum, step)[1])

    def alpha(self, minimum: float, maximum: float, step: float) -> float:
        value = self._value(minimum, maximum, step)
        if value > maximum + EPSILON * max(1.0, abs(maximum)):
            return 0.0
        return 0.0 if any(abs(value - excluded) < EPSILON * max(1.0, step) for excluded in self.exclude) else 1.0


class _Crossing:
    """Where an axis sits on the other one: at its value 0, or the nearest end."""

    __slots__ = ("length",)

    def __init__(self, length: float) -> None:
        self.length = length

    def __call__(self, minimum: float, maximum: float, step: float) -> float:
        return _place(min(max(0.0, minimum), maximum), minimum, maximum, self.length)


class _Frame:
    """Coordinates to an Axes' parent space: the ranges onto the lengths,
    then the Axes' scale, rotation and position."""

    __slots__ = ("lengths",)

    def __init__(self, lengths: tuple[float, float]) -> None:
        self.lengths = lengths

    def place(self, coordinates: np.ndarray, values: Sequence[Any]) -> np.ndarray:
        x_min, x_max, y_min, y_max = (float(value) for value in values[:4])
        position, scale, rotation = np.asarray(values[4:7], dtype=np.float64), np.asarray(values[7:10], dtype=np.float64), values[10]
        local = np.zeros((len(coordinates), 3))
        local[:, 0] = ((coordinates[:, 0] - x_min) / ((x_max - x_min) or 1.0) - 0.5) * self.lengths[0]
        local[:, 1] = ((coordinates[:, 1] - y_min) / ((y_max - y_min) or 1.0) - 0.5) * self.lengths[1]
        return quaternion.rotate_vectors(rotation, local * scale) + position


class _FramePoint(_Frame):
    __slots__ = ()

    def __call__(self, x: float, y: float, *values: Any) -> np.ndarray:
        return self.place(np.array([[x, y]], dtype=np.float64), values)


class _Plot(_Frame):
    """A function's graph on Axes, sampled over its range or the x range."""

    __slots__ = ("function", "fixed", "samples", "clip")

    def __init__(self, function: Callable[[float], float], fixed: tuple[float, float] | None, samples: int, clip: bool, lengths: tuple[float, float]) -> None:
        super().__init__(lengths)
        self.function, self.fixed, self.samples, self.clip = function, fixed, samples, clip

    def __call__(self, *values: Any) -> np.ndarray:
        start, end = self.fixed if self.fixed is not None else (float(values[0]), float(values[1]))
        xs = np.linspace(start, end, self.samples + 1)
        coordinates = np.column_stack([xs, [self.function(x) for x in xs]])
        runs = _inside(coordinates, float(values[2]), float(values[3])) if self.clip else [coordinates]
        pieces = [_smooth(self.place(run, values)[:, :2]) for run in runs if len(run) > 1]
        return np.concatenate(pieces) if pieces else np.zeros((0, 2))


def _inside(coordinates: np.ndarray, low: float, high: float) -> list[np.ndarray]:
    """The runs of samples with y in ``[low, high]``, each ended where the
    straight piece to the next sample crosses the bound."""
    ys = coordinates[:, 1]
    inside = (ys >= low) & (ys <= high) & np.isfinite(ys)
    runs: list[np.ndarray] = []
    run: list[np.ndarray] = []
    for index in range(len(coordinates)):
        if inside[index]:
            if not run and index > 0 and np.isfinite(ys[index - 1]):
                run.append(_crossing(coordinates[index - 1], coordinates[index], low, high))
            run.append(coordinates[index])
        elif run:
            if np.isfinite(ys[index]):
                run.append(_crossing(coordinates[index - 1], coordinates[index], low, high))
            runs.append(np.array(run))
            run = []
    if run:
        runs.append(np.array(run))
    return runs


def _crossing(a: np.ndarray, b: np.ndarray, low: float, high: float) -> np.ndarray:
    """Where the segment from ``a`` to ``b`` meets ``y = low`` or ``y = high``."""
    bound = high if max(a[1], b[1]) > high else low
    t = (bound - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
    return a + (b - a) * min(max(t, 0.0), 1.0)


class _Grid:
    """A line at every tick of both ranges, across the whole plane."""

    __slots__ = ("lengths",)

    def __init__(self, lengths: tuple[float, float]) -> None:
        self.lengths = lengths

    def __call__(self, x_min: float, x_max: float, x_step: float, y_min: float, y_max: float, y_step: float) -> np.ndarray:
        # Ticks as long as the plane is across: the grid lines.
        width, height = self.lengths
        columns = _Ticks(width, np.array([1.0, 0.0]), height / 2)(x_min, x_max, x_step)
        rows = _Ticks(height, np.array([0.0, 1.0]), width / 2)(y_min, y_max, y_step)
        return np.concatenate([columns, rows])


class _Curve:
    """A parametric 2D curve sampled over ``[minimum, maximum]``."""

    __slots__ = ("function", "samples")

    def __init__(self, function: Callable[[float], Sequence[float]], samples: int) -> None:
        self.function, self.samples = function, samples

    def __call__(self, minimum: float, maximum: float) -> np.ndarray:
        ts = np.linspace(minimum, maximum, self.samples + 1)
        return _smooth(np.array([np.asarray(self.function(t), dtype=np.float64)[:2] for t in ts]))


def _smooth(anchors: np.ndarray) -> np.ndarray:
    """Cubic curves through ``anchors`` with handles along the finite
    difference tangents (as parametric_points)."""
    if len(anchors) < 2:
        return np.zeros((0, 2))
    tangents = np.gradient(anchors, axis=0) / 3
    curves = np.stack([anchors[:-1], anchors[:-1] + tangents[:-1], anchors[1:] - tangents[1:], anchors[1:]], axis=1)
    return curves.reshape(-1, 2)


def _range(values: Range) -> tuple[float, float, float]:
    minimum, maximum = float(values[0]), float(values[1])
    step = float(values[2]) if len(values) > 2 else 1.0
    if not maximum > minimum:
        raise ValueError(f"a range needs maximum > minimum, got {minimum} and {maximum}")
    if not step > 0:
        raise ValueError(f"a range needs a positive step, got {step}")
    return minimum, maximum, step


def _range_or_keep(values: Range) -> tuple[float, float, float | None]:
    minimum, maximum = float(values[0]), float(values[1])
    return minimum, maximum, float(values[2]) if len(values) > 2 else None


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)[:2]
    return array / (np.linalg.norm(array) or 1.0)


__all__ = ["Axes", "FunctionGraph", "NumberLine", "NumberPlane", "ParametricFunction", "Value"]
