"""Number lines, axes, planes and the graphs of functions on them.

A NumberLine's range (``minimum``, ``maximum``, ``step``) is made of
signals, and everything on it is derived from them: the ticks, the
labels (their values, places and whether they show) and, on Axes, the
graphs of functions and the points of coordinates. So animating a range
zooms: ticks, labels and plots follow at every frame::

    axes = Axes((0, 5, 1), (0, 25, 5), include_numbers=True)
    graph = axes.plot(lambda x: x**2)
    self.play(axes.animate.set_ranges(x=(0, 10), y=(0, 100)))

Adaptive ticks (the default) keep the marks readable at any zoom, like a
map: ticks, grid lines and labels lie on nested levels of the step,

    ..., step / 10, step / 2, step, 5 step, 10 step, 50 step, ...

each a multiple of the one before. A level shows while its marks are at
least ``threshold`` apart on the line, fading in between that and the
spacing at which the next finer level starts; coarser levels show fully.
Zooming in, finer marks fade in between the others; zooming out, they
fade out. Nothing jumps, and what shows depends on the range alone, so
it is the same however the range got there. The thresholds come from the
range a line is made with: there, exactly the multiples of the step
show. ``adaptive=False`` puts marks on the step's multiples only.

Labels are DecimalNumbers from a pool, the ones beyond the range hidden.
Graphs and points are in the space of the Axes' parent, so they follow
the Axes' own moves too; put them next to it. No class picks a color.
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
from ..animation.nodes.node import ABS, ChangeList, animatable
from ..animation.signals.paths import PointSignal
from ..animation.signals.signal import derive
from .geometry import corner_points, tip_points
from .lines import TIP_LENGTH, _tip_style
from .numbers import DecimalNumber
from .shapes2d import parameter

EPSILON = 1e-9

Range = Sequence[float]
"""``(minimum, maximum)`` or ``(minimum, maximum, step)``."""


class Value(Node):
    """A number to animate that is not drawn (Manim's ValueTracker): derive
    other signals from ``value``. ``self.play(t.animate.set_value(1))``."""

    drawn = False

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
    ``ticks`` and ``minor_ticks`` (one path each; the finest level of
    adaptive ticks, fading, see the module docs), with ``include_tip`` a
    ``tip`` at the maximum end and, with ``include_numbers``, ``numbers``
    (DecimalNumbers with ``decimals`` places set with ``preamble``,
    ``label_buff`` beyond the side ``label_direction``, default below a
    horizontal line). Values in ``exclude`` get no label; neither do
    values under the tip. ``style`` is the line's, ticks' and tip's.

    Labels follow the same levels as the ticks, but only levels whose
    labels are at least ``label_spacing`` apart and whose values have at
    most ``decimals`` places. The step is DISCRETE: animating it jumps at
    the end.
    """

    def __init__(
        self,
        x_range: Range = (-5.0, 5.0, 1.0),
        length: float = 10.0,
        angle: float = 0.0,
        tick_size: float = 0.1,
        include_tip: bool = False,
        tip_length: float = TIP_LENGTH,
        include_numbers: bool = False,
        decimals: int = 0,
        preamble: str = "",
        label_direction: Sequence[float] | None = None,
        label_buff: float = 0.25,
        label_spacing: float = 0.6,
        exclude: Sequence[float] = (),
        adaptive: bool = True,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        minimum, maximum, step = _range(x_range)
        self.minimum = parameter(self, "minimum", minimum)
        self.maximum = parameter(self, "maximum", maximum)
        self.step = parameter(self, "step", step, DISCRETE)  # jumps: ticks stay on round values
        self.length, self.angle, self.tick_size = float(length), float(angle), float(tick_size)
        self.direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        normal = np.array([math.sin(self.angle), -math.cos(self.angle)])  # right of the direction: below a horizontal line
        self.label_direction = normal if label_direction is None else _unit(label_direction)
        self.label_buff = float(label_buff)
        self.decimals, self.preamble, self.include_numbers = int(decimals), preamble, include_numbers
        # Where marks end: under the tip they would be hidden.
        self.tip_length = float(tip_length) if include_tip else 0.0
        end = self.length / 2 - self.tip_length
        self.line = Shape2D(corner_points([-self.direction * self.length / 2, self.direction * end]), f"{self.name}.line", **style)
        self.add(self.line)
        if include_tip:
            self.tip = Shape2D(tip_points(self.direction * self.length / 2, self.direction, self.tip_length, self.tip_length), f"{self.name}.tip", **_tip_style(style))
            self.add(self.tip)
        spacing = step * self.length / (maximum - minimum)  # between the step's ticks, as made
        self.levels = _Levels(self.length, spacing / 2, spacing / 4, adaptive)
        label_spacing = float(label_spacing)
        self.label_levels = _Levels(self.length, max(spacing, label_spacing) / 2, max(spacing / 4, label_spacing / 2), adaptive, self.decimals)
        marks = _Marks(self.length, self.direction, self.tip_length)
        self.ticks = Shape2D((), f"{self.name}.ticks", **style)
        self.ticks._derive_points(_TickPoints(marks, self.levels, self.tick_size, minor=False), *self.range_signals)
        self.minor_ticks = Shape2D((), f"{self.name}.minor_ticks", **style)
        self.minor_ticks._derive_points(_TickPoints(marks, self.levels, self.tick_size, minor=True), *self.range_signals)
        self.minor_ticks.alpha = DerivedSignal(self.levels.fade, self.range_signals, SCALAR, name=f"{self.name}.minor_ticks.alpha")
        self.add(self.minor_ticks, self.ticks)
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
        """Enough labels for the range (the pool never shrinks). Adaptive
        labels are never closer than their threshold, which bounds them."""
        needed = self.label_levels.most_marks(minimum, maximum, step) + 1
        marks = _Marks(self.length, self.direction, self.tip_length)
        offset = self.label_direction * self.label_buff
        while len(self.numbers) < needed:
            index = len(self.numbers)
            label = _Label(index, marks, self.label_levels, offset, self.exclude)
            value = DerivedSignal(label.value, self.range_signals, SCALAR, name=f"{self.name}.number{index}.value")
            number = DecimalNumber(value, self.decimals, self.preamble, characters=7, name=f"{self.name}.number{index}", **self.number_style)
            number.position.x = DerivedSignal(label.x, self.range_signals, SCALAR, name=f"{number.name}.position.x")
            number.position.y = DerivedSignal(label.y, self.range_signals, SCALAR, name=f"{number.name}.position.y")
            number.alpha = DerivedSignal(label.alpha, self.range_signals, SCALAR, name=f"{number.name}.alpha")
            self.numbers.append(number)
            self.add(number)


class Axes(Group):
    """Two NumberLines for ``x_range`` and ``y_range``, ``x_length`` and
    ``y_length`` long, crossing where both are 0 (or the nearest end of a
    range without 0): ``x_axis`` and ``y_axis``, with tips at their
    maximum ends (``include_tips``) and adaptive ticks (``adaptive``, see
    the module docs).

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
        include_tips: bool = True,
        adaptive: bool = True,
        name: str | None = None,
        **axis_style: Any,
    ) -> None:
        super().__init__(name=name)
        common = {
            "include_numbers": include_numbers, "decimals": decimals, "preamble": preamble, "tick_size": tick_size,
            "include_tip": include_tips, "adaptive": adaptive, "exclude": (0.0,),
        }  # fmt: skip
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
    """Axes over a grid: ``grid`` holds a line across the plane at every
    fully shown tick of both ranges (one path), ``minor_x_lines`` and
    ``minor_y_lines`` those at the ticks of each range's finest, fading
    level (see the module docs), all styled by ``grid_style`` and drawn
    under the axes. Its axes have no tips unless asked."""

    def __init__(
        self,
        x_range: Range = (-7.0, 7.0, 1.0),
        y_range: Range = (-4.0, 4.0, 1.0),
        x_length: float = 14.0,
        y_length: float = 8.0,
        grid_style: dict[str, Any] | None = None,
        include_tips: bool = False,
        name: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(x_range, y_range, x_length, y_length, include_tips=include_tips, name=name, **options)
        ranges = (*self.x_axis.range_signals, *self.y_axis.range_signals)
        self.grid = Shape2D((), f"{self.name}.grid", **(grid_style or {}))
        self.grid._derive_points(_Grid(self.lengths, self.x_axis.levels, self.y_axis.levels), *ranges)
        minors = []
        for axis, line in enumerate((self.x_axis, self.y_axis)):
            minor = Shape2D((), f"{self.name}.minor_{'xy'[axis]}_lines", **(grid_style or {}))
            minor._derive_points(_GridLines(self.lengths, axis, line.levels, minor=True), *line.range_signals)
            minor.alpha = DerivedSignal(line.levels.fade, line.range_signals, SCALAR, name=f"{minor.name}.alpha")
            minors.append(minor)
        self.minor_x_lines, self.minor_y_lines = minors
        self.children[:0] = [*minors, self.grid]  # under the axes
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


class _Levels:
    """The nested levels of a line's marks (see the module docs): for a
    range, the finest level that shows, ``fine`` (a step), how many of its
    steps make the next coarser level, ``ratio``, and how far it has faded
    in, ``fade``. Marks at multiples of ``fine * ratio`` show fully, the
    others with ``fade``.

    Level ``n`` is ``step * 10 ** (n // 2)``, times 5 for odd ``n``: level
    0 is the step. A level shows while its marks are at least a threshold
    apart, fading in up to the spacing at which the level below would
    start: levels finer than the step need ``finer`` (zooming in, they
    appear only beyond the range a line was made with), the step and
    coarser ones ``coarser``, smaller (zooming out, the step's marks stay
    until they are dense; when they are gone, the 5 steps' marks are
    about as far apart as the step's were). With ``decimals``, levels
    whose values need more places never show. Without ``adaptive``, there
    is one level, the step, shown fully.
    """

    __slots__ = ("length", "finer", "coarser", "adaptive", "decimals", "_last")

    def __init__(self, length: float, finer: float, coarser: float, adaptive: bool, decimals: int | None = None) -> None:
        self.length, self.finer, self.coarser, self.adaptive, self.decimals = length, finer, coarser, adaptive, decimals
        self._last: tuple[tuple[float, float, float], tuple[float, int, float]] | None = None
        """The last range and its levels: every label of a line asks for
        the same range in turn."""

    def __call__(self, minimum: float, maximum: float, step: float) -> tuple[float, int, float]:
        key = (minimum, maximum, step)
        if self._last is None or self._last[0] != key:
            self._last = (key, self._levels(minimum, maximum, step))
        return self._last[1]

    def _levels(self, minimum: float, maximum: float, step: float) -> tuple[float, int, float]:
        if not self.adaptive:
            return step, 1, 1.0
        per_value = self.length / ((maximum - minimum) or 1.0)
        # The first level far enough apart, from below.
        level = min(0, 2 * math.floor(math.log10(self.finer / (step * per_value))) - 2)
        while _level_step(step, level) * per_value < self._threshold(level) * (1 - EPSILON):
            level += 1
        finer = 2.0 if level % 2 == 0 else 5.0  # how much finer the level below is
        spaced = _level_step(step, level) * per_value / self._threshold(level)
        fade = _smoothstep(min(math.log(max(spaced, 1.0)) / math.log(finer), 1.0))
        while self.decimals is not None and not _has_places(_level_step(step, level), self.decimals):
            level, fade = level + 1, 1.0  # too fine for the labels: the coarser level, fully
        return _level_step(step, level), 5 if level % 2 == 0 else 2, fade

    def _threshold(self, level: int) -> float:
        return self.finer if level < 0 else self.coarser

    def fade(self, minimum: float, maximum: float, step: float) -> float:
        return self(minimum, maximum, step)[2]

    def most_marks(self, minimum: float, maximum: float, step: float) -> int:
        """How many marks a range can have at most, over any zoom (adaptive)
        or for this range (otherwise)."""
        if self.adaptive:
            return int(self.length / min(self.finer, self.coarser)) + 2
        return int(math.floor((maximum - minimum) / step + EPSILON)) + 2


def _level_step(step: float, level: int) -> float:
    return step * 10.0 ** (level // 2) * (5.0 if level % 2 else 1.0)


def _has_places(value: float, decimals: int) -> bool:
    scaled = value * 10.0**decimals
    return abs(scaled - round(scaled)) < 1e-6 * max(1.0, abs(scaled))


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _multiples(minimum: float, maximum: float, step: float) -> range:
    """The ``k`` with ``k * step`` in ``[minimum, maximum]``."""
    first = math.ceil(minimum / step - EPSILON)
    last = math.floor(maximum / step + EPSILON)
    return range(first, max(first, min(last + 1, first + 10_000)))


def _multiple_array(minimum: float, maximum: float, step: float) -> np.ndarray:
    """:func:`_multiples` as an integer array."""
    multiples = _multiples(minimum, maximum, step)
    return np.arange(multiples.start, multiples.stop, dtype=np.int64)


def _place_array(values: np.ndarray, minimum: float, maximum: float, length: float) -> np.ndarray:
    span = maximum - minimum
    return ((values - minimum) / span - 0.5) * length if span else np.zeros(len(values))


def _segments(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Straight lines from ``starts`` to ``ends`` (``(N, 2)``), one subpath
    each, as corner_points makes them."""
    thirds = np.linspace(0.0, 1.0, 4)[None, :, None]
    return (starts[:, None, :] + (ends - starts)[:, None, :] * thirds).reshape(-1, 2)


class _Marks:
    """Where marks go on a line, and whether a value's mark is hidden
    under its tip."""

    __slots__ = ("length", "direction", "end")

    def __init__(self, length: float, direction: np.ndarray, tip_length: float) -> None:
        self.length, self.direction = length, direction
        self.end = length / 2 - tip_length - EPSILON  # beyond it: under the tip

    def along(self, value: float, minimum: float, maximum: float) -> float:
        return _place(value, minimum, maximum, self.length)

    def shown(self, value: float, minimum: float, maximum: float) -> bool:
        return self.end >= self.length / 2 - EPSILON or self.along(value, minimum, maximum) <= self.end


class _TickPoints:
    """A line's ticks of the levels that show fully or (``minor``) its
    finest, fading level."""

    __slots__ = ("marks", "levels", "size", "minor")

    def __init__(self, marks: _Marks, levels: _Levels, size: float, minor: bool) -> None:
        self.marks, self.levels, self.size, self.minor = marks, levels, size, minor

    def __call__(self, minimum: float, maximum: float, step: float) -> np.ndarray:
        fine, ratio, _ = self.levels(minimum, maximum, step)
        k = _multiple_array(minimum, maximum, fine)
        k = k[((k % ratio) != 0) == self.minor]
        along = _place_array(k * fine, minimum, maximum, self.marks.length)
        along = along[along <= self.marks.end] if self.marks.end < self.marks.length / 2 - EPSILON else along
        direction = self.marks.direction
        normal = np.array([-direction[1], direction[0]]) * self.size
        centers = along[:, None] * direction
        return _segments(centers - normal, centers + normal)


class _Label:
    """What a pooled label shows: the ``index``-th multiple of the finest
    level in the range, where, and how visible."""

    __slots__ = ("index", "marks", "levels", "offset", "exclude")

    def __init__(self, index: int, marks: _Marks, levels: _Levels, offset: np.ndarray, exclude: tuple[float, ...]) -> None:
        self.index, self.marks, self.levels, self.offset, self.exclude = index, marks, levels, offset, exclude

    def _mark(self, minimum: float, maximum: float, step: float) -> tuple[float, float]:
        """The label's value and its alpha."""
        fine, ratio, fade = self.levels(minimum, maximum, step)
        multiples = _multiples(minimum, maximum, fine)
        k = multiples.start + self.index
        value = k * fine
        value = 0.0 if abs(value) < EPSILON * fine else value  # no -0
        if k not in multiples or not self.marks.shown(value, minimum, maximum):
            return value, 0.0
        if any(abs(value - excluded) < EPSILON * max(1.0, fine) for excluded in self.exclude):
            return value, 0.0
        return value, 1.0 if k % ratio == 0 else fade

    def value(self, minimum: float, maximum: float, step: float) -> float:
        return self._mark(minimum, maximum, step)[0]

    def _point(self, minimum: float, maximum: float, step: float) -> np.ndarray:
        return self.marks.along(self.value(minimum, maximum, step), minimum, maximum) * self.marks.direction + self.offset

    def x(self, minimum: float, maximum: float, step: float) -> float:
        return float(self._point(minimum, maximum, step)[0])

    def y(self, minimum: float, maximum: float, step: float) -> float:
        return float(self._point(minimum, maximum, step)[1])

    def alpha(self, minimum: float, maximum: float, step: float) -> float:
        return self._mark(minimum, maximum, step)[1]


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


class _GridLines:
    """A plane's lines across it at the marks of one range (``axis`` 0:
    x, the vertical lines) that show fully or (``minor``) at its finest,
    fading level; none on the plane's edges."""

    __slots__ = ("lengths", "axis", "levels", "minor")

    def __init__(self, lengths: tuple[float, float], axis: int, levels: _Levels, minor: bool) -> None:
        self.lengths, self.axis, self.levels, self.minor = lengths, axis, levels, minor

    def __call__(self, minimum: float, maximum: float, step: float) -> np.ndarray:
        length, across = self.lengths[self.axis], self.lengths[1 - self.axis]
        fine, ratio, _ = self.levels(minimum, maximum, step)
        k = _multiple_array(minimum, maximum, fine)
        values = k[((k % ratio) != 0) == self.minor] * fine
        edge = EPSILON * (maximum - minimum)
        at = _place_array(values[(values > minimum + edge) & (values < maximum - edge)], minimum, maximum, length)
        starts, ends = np.zeros((len(at), 2)), np.zeros((len(at), 2))
        starts[:, self.axis] = ends[:, self.axis] = at
        starts[:, 1 - self.axis], ends[:, 1 - self.axis] = -across / 2, across / 2
        return _segments(starts, ends)


class _Grid:
    """The fully shown lines of both ranges, one path."""

    __slots__ = ("columns", "rows")

    def __init__(self, lengths: tuple[float, float], x_levels: _Levels, y_levels: _Levels) -> None:
        self.columns = _GridLines(lengths, 0, x_levels, minor=False)
        self.rows = _GridLines(lengths, 1, y_levels, minor=False)

    def __call__(self, x_min: float, x_max: float, x_step: float, y_min: float, y_max: float, y_step: float) -> np.ndarray:
        return np.concatenate([self.columns(x_min, x_max, x_step), self.rows(y_min, y_max, y_step)])


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
