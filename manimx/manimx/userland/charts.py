"""Bar charts on Axes.

A :class:`BarChart`'s values are signals. Each bar is a Shape2D whose
points are derived from its value and the Axes' ranges and transform,
like a plot (graphs.py), so ``set_values`` grows and shrinks the bars and
zooming the y range rescales them. No class picks a color: bars take
``bar_colors`` (one color, or one per bar, repeated).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..animation import SCALAR, Group, RootSignal, Shape2D, Signal
from ..animation.authoring import current
from ..animation.nodes.node import ABS, ChangeList, animatable
from ..utils.color import ParsableManimColor
from .constants import DOWN, UP
from .geometry import corner_points
from .graphs import Axes, Range, _Frame
from .numbers import DecimalNumber
from .typst import Typst


class BarChart(Group):
    """One bar per value of ``values``, side by side on ``axes``: bar ``i``
    stands on the x range's ``[i, i + 1]``, ``bar_width`` of it wide.

    ``bar_names`` are Typst markup under the bars; ``y_range`` defaults to
    a round range holding the values and 0. With ``include_values`` a
    DecimalNumber with ``decimals`` places rides on every bar's top.
    ``preamble`` sets the names, numbers and the axes' labels;
    ``axis_style`` styles the axes, ``bar_style`` (Shape2D style
    arguments) the bars, which are filled and have no stroke unless it
    says otherwise.
    """

    def __init__(
        self,
        values: Sequence[float],
        bar_names: Sequence[str] | None = None,
        y_range: Range | None = None,
        x_length: float = 10.0,
        y_length: float = 6.0,
        bar_width: float = 0.6,
        bar_colors: ParsableManimColor | Sequence[ParsableManimColor] | None = None,
        include_values: bool = False,
        decimals: int = 0,
        preamble: str = "",
        axis_style: dict[str, Any] | None = None,
        name: str | None = None,
        **bar_style: Any,
    ) -> None:
        super().__init__(name=name)
        if not values:
            raise ValueError("a BarChart needs at least one value")
        count = len(values)
        self.values = [RootSignal(float(value), SCALAR, name=f"{self.name}.value{index}") for index, value in enumerate(values)]
        self.axes = Axes(
            (0.0, float(count), 1.0), _nice_range(values) if y_range is None else y_range, x_length, y_length,
            include_numbers=True, preamble=preamble, include_tips=False, name=f"{self.name}.axes", **(axis_style or {}),
        )  # fmt: skip
        self.axes.x_axis.ticks.alpha.set(0.0)
        self.axes.x_axis.minor_ticks.alpha = RootSignal(0.0, SCALAR, name=f"{self.axes.x_axis.name}.minor_ticks.alpha")
        for number in self.axes.x_axis.numbers:
            number.alpha = RootSignal(0.0, SCALAR, name=f"{number.name}.alpha")
        self.add(self.axes)
        colors = _colors(bar_colors, count)
        style = {"fill_opacity": 1.0, "stroke_width": 0.0} | bar_style
        self.bars: list[Shape2D] = []
        for index, value in enumerate(self.values):
            color = {"color": colors[index]} if colors[index] is not None else {}
            bar = Shape2D((), f"{self.name}.bar{index}", **(color | style))
            bar._derive_points(_Bar(index + 0.5, float(bar_width), self.axes.lengths), value, *self.axes._frame_inputs())
            self.bars.append(bar)
        self.add(*self.bars)
        self.names: list[Typst] = []
        for index, label in enumerate(bar_names or ()):
            text = Typst(label, preamble, name=f"{self.name}.name{index}")
            text.move_to(self.axes.coords_to_point(index + 0.5, current(self.axes.y_axis.minimum)) + DOWN * 0.35)
            self.names.append(text)
        self.add(*self.names)
        self.numbers: list[DecimalNumber] = []
        if include_values:
            for index, value in enumerate(self.values):
                number = DecimalNumber(value, decimals, preamble, name=f"{self.name}.number{index}")
                number.follow(self.axes.point(index + 0.5, value), UP * 0.3)
                self.numbers.append(number)
            self.add(*self.numbers)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from self.values

    @animatable
    def set_values(self, values: Sequence[float]) -> ChangeList:
        """Change every bar's value (as many values as bars)."""
        if len(values) != len(self.values):
            raise ValueError(f"{self.name} has {len(self.values)} bars, got {len(values)} values")
        return [(signal, ABS, float(value)) for signal, value in zip(self.values, values, strict=True)]


class _Bar(_Frame):
    """A bar's rectangle from the x axis' height to its value, kept inside
    the y range."""

    __slots__ = ("x", "width")

    def __init__(self, x: float, width: float, lengths: tuple[float, float]) -> None:
        super().__init__(lengths)
        self.x, self.width = x, width

    def __call__(self, value: float, *frame: Any) -> np.ndarray:
        low, high = float(frame[2]), float(frame[3])
        base = min(max(0.0, low), high)
        top = min(max(float(value), low), high)
        left, right = self.x - self.width / 2, self.x + self.width / 2
        corners = self.place(np.array([[left, base], [right, base], [right, top], [left, top]]), frame)[:, :2]
        return corner_points(corners, closed=True)


def _nice_range(values: Sequence[float]) -> tuple[float, float, float]:
    """A range from 0 (or below the smallest value) to above the largest,
    with a step of 1, 2 or 5 times a power of ten and about five steps."""
    low, high = min(0.0, *values), max(0.0, *values)
    span = (high - low) or 1.0
    power = 10.0 ** math.floor(math.log10(span / 5))
    step = next(factor * power for factor in (1.0, 2.0, 5.0, 10.0) if span / (factor * power) <= 6)
    return (math.floor(low / step) * step, math.ceil(high / step) * step, step)


def _colors(colors: ParsableManimColor | Sequence[ParsableManimColor] | None, count: int) -> list[Any]:
    if colors is None:
        return [None] * count
    if isinstance(colors, str) or not isinstance(colors, Sequence) or (len(colors) in (3, 4) and all(isinstance(c, (int, float)) for c in colors)):
        return [colors] * count
    return [colors[index % len(colors)] for index in range(count)]


__all__ = ["BarChart"]
