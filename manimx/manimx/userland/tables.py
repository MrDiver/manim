"""Tables and matrices: nodes laid out on a grid of rows and columns.

Both measure their entries once, when they are made: a column is as
wide as its widest entry and a row as high as its highest, plus the
buffers, and every entry is centered in its cell. Entries are any
nodes; text and numbers given as such become Typst (:class:`Table`) or
Typst math (:class:`Matrix`). Rows and columns count from 0, like
Python lists. No class picks a color: ``style`` reaches the entries made
here, ``line_style`` a table's lines.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from ..animation import Group, Node, RootSignal, Shape2D, Signal
from ..animation.authoring import current
from ..animation.nodes.node import ABS, ChangeList, animatable
from .constants import ORIGIN
from .geometry import union_points
from .lines import Line
from .markers import current_bounds
from .numbers import DecimalNumber
from .svg import read_svg
from .typst import PAGE, Typst, TypstMath, compile_typst

Entry = Node | str | float | None
"""What a cell holds: a node, text or a number (made into Typst), or
nothing."""


class _Grid:
    """Where the cells of ``cells`` go: column widths and row heights from
    the entries' sizes plus the buffers, centered on the origin."""

    def __init__(self, cells: Sequence[Sequence[Node | None]], h_buff: float, v_buff: float) -> None:
        rows, columns = len(cells), max((len(row) for row in cells), default=0)
        widths, heights = np.zeros(columns), np.zeros(rows)
        self.boxes: dict[int, np.ndarray] = {}
        for row, entries in enumerate(cells):
            for column, entry in enumerate(entries):
                if entry is None:
                    continue
                box = current_bounds(entry)
                self.boxes[entry.uid] = box
                widths[column] = max(widths[column], box[1, 0] - box[0, 0])
                heights[row] = max(heights[row], box[1, 1] - box[0, 1])
        self.widths, self.heights = widths + h_buff, heights + v_buff
        self.left = -self.widths.sum() / 2
        self.top = self.heights.sum() / 2
        self.lefts = self.left + np.concatenate([[0.0], np.cumsum(self.widths)])
        self.tops = self.top - np.concatenate([[0.0], np.cumsum(self.heights)])
        # Text in a row shares a baseline: at the middle of the row less
        # the text's usual height of its middle above the baseline.
        self.baselines: dict[int, float] = {}
        self.drop: dict[int, float] = {}
        for row, entries in enumerate(cells):
            middles = [float((self.boxes[entry.uid][0, 1] + self.boxes[entry.uid][1, 1]) / 2) - baseline for entry in entries if (baseline := _baseline(entry)) is not None]
            if middles:
                self.drop[row] = float(np.median(middles))

    def center(self, row: int, column: int) -> np.ndarray:
        return np.array([(self.lefts[column] + self.lefts[column + 1]) / 2, (self.tops[row] + self.tops[row + 1]) / 2, 0.0])

    def place(self, entry: Node, row: int, column: int) -> None:
        """Move ``entry`` so its bounding box is centered in its cell; text
        (an SVG with a baseline) sits on the row's baseline instead."""
        box = self.boxes[entry.uid]
        offset = self.center(row, column) - (box[0] + box[1]) / 2 * [1.0, 1.0, 0.0]
        baseline = _baseline(entry)
        if baseline is not None:
            offset[1] = self.center(row, column)[1] - self.drop[row] - baseline
        entry.shift(offset)


def _baseline(entry: Node) -> float | None:
    """The baseline of text in its parent's space, where it is now."""
    baseline = getattr(entry, "baseline", None)
    return None if baseline is None else float(current(entry.position.y)) + baseline * float(current(entry.scale.y))  # type: ignore[attr-defined]


def _in_parent(group: Group, point: np.ndarray) -> np.ndarray:
    """``point`` of ``group``'s own space in its parent's space, where the
    group is now (rotation is not followed)."""
    scale = np.array([current(axis) for axis in group.scale.axes])
    return np.array([current(axis) for axis in group.position.axes]) + point * scale


class Table(Group):
    """``rows`` of entries with optional ``row_labels`` (a column on the
    left), ``column_labels`` (a row on top) and ``top_left`` (the corner
    between them), separated by lines.

    Text entries are Typst markup set with ``preamble``, numbers are
    shown as written; ``style`` reaches them. ``h_buff`` and ``v_buff``
    are the room around the widest and highest entries. Lines run between
    all rows and columns, and around the table with ``outer_lines``;
    ``line_style`` styles them (a Line's style arguments).

    :meth:`cell` is an entry of the whole grid (labels included, from the
    top left, counting from 0); :meth:`row` and :meth:`column` are lists
    of them; :meth:`region` is the box of some cells, for a highlight.
    """

    def __init__(
        self,
        rows: Sequence[Sequence[Entry]],
        row_labels: Sequence[Entry] | None = None,
        column_labels: Sequence[Entry] | None = None,
        top_left: Entry = None,
        preamble: str = "",
        h_buff: float = 0.6,
        v_buff: float = 0.3,
        outer_lines: bool = False,
        line_style: Mapping[str, Any] | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        make = lambda entry, place: _entry(entry, lambda text: Typst(text, preamble, name=f"{self.name}.{place}", **style))  # noqa: E731
        grid: list[list[Node | None]] = [[make(entry, f"{r}.{c}") for c, entry in enumerate(row)] for r, row in enumerate(rows)]
        if row_labels is not None:
            if len(row_labels) != len(grid):
                raise ValueError(f"{len(row_labels)} row labels for {len(grid)} rows")
            for index, (row, label) in enumerate(zip(grid, row_labels, strict=True)):
                row.insert(0, make(label, f"row_label{index}"))
        if column_labels is not None:
            header = [make(label, f"column_label{index}") for index, label in enumerate(column_labels)]
            grid.insert(0, ([make(top_left, "top_left")] if row_labels is not None else []) + header)
        columns = max((len(row) for row in grid), default=0)
        self.cells: list[list[Node | None]] = [row + [None] * (columns - len(row)) for row in grid]
        """Every entry of the grid, labels included (None where empty)."""
        self.grid = _Grid(self.cells, h_buff, v_buff)
        line_style = dict(line_style or {})
        g = self.grid
        inner_rows = range(0 if outer_lines else 1, len(g.tops) if outer_lines else len(g.tops) - 1)
        inner_columns = range(0 if outer_lines else 1, len(g.lefts) if outer_lines else len(g.lefts) - 1)
        self.horizontal_lines = Group(
            *(Line((g.lefts[0], g.tops[i], 0.0), (g.lefts[-1], g.tops[i], 0.0), name=f"{self.name}.horizontal{i}", **line_style) for i in inner_rows),
            name=f"{self.name}.horizontal_lines",
        )
        self.vertical_lines = Group(
            *(Line((g.lefts[j], g.tops[0], 0.0), (g.lefts[j], g.tops[-1], 0.0), name=f"{self.name}.vertical{j}", **line_style) for j in inner_columns),
            name=f"{self.name}.vertical_lines",
        )
        self.add(self.horizontal_lines, self.vertical_lines)
        for r, row in enumerate(self.cells):
            for c, entry in enumerate(row):
                if entry is not None:
                    g.place(entry, r, c)
                    self.add(entry)
        self.row_labels = [row[0] for row in self.cells[1 if column_labels is not None else 0 :]] if row_labels is not None else []
        self.column_labels = self.cells[0][1 if row_labels is not None else 0 :] if column_labels is not None else []

    @property
    def entries(self) -> list[list[Node | None]]:
        """The entries without the labels."""
        top = 1 if self.column_labels else 0
        left = 1 if self.row_labels else 0
        return [row[left:] for row in self.cells[top:]]

    def cell(self, row: int, column: int) -> Node | None:
        return self.cells[row][column]

    def row(self, index: int) -> list[Node]:
        return [entry for entry in self.cells[index] if entry is not None]

    def column(self, index: int) -> list[Node]:
        return [row[index] for row in self.cells if row[index] is not None]  # type: ignore[misc]

    def region(self, row: int, column: int, last_row: int | None = None, last_column: int | None = None) -> tuple[np.ndarray, float, float]:
        """``(center, width, height)`` of the cells from ``(row, column)``
        to ``(last_row, last_column)`` (default: the one cell), where the
        table is now, in its parent's space (rotation is not followed):
        ``Rectangle(width, height).move_to(center)`` covers them."""
        last_row = row if last_row is None else last_row
        last_column = column if last_column is None else last_column
        g = self.grid
        left, right = g.lefts[column], g.lefts[last_column + 1]
        top, bottom = g.tops[row], g.tops[last_row + 1]
        scale = np.array([current(axis) for axis in self.scale.axes])
        return _in_parent(self, np.array([(left + right) / 2, (top + bottom) / 2, 0.0])), float((right - left) * scale[0]), float((top - bottom) * scale[1])


BRACKETS = {"[": "]", "(": ")", "{": "}", "|": "|", "‖": "‖", "": ""}
"""The brackets a Matrix can have (left: right); "" for none."""


class Matrix(Group):
    """``rows`` of entries between brackets: ``Matrix([[1, 2], [3, 4]])``.

    Text and numbers become Typst math set with ``preamble`` (``"x^2"``,
    ``"pi"``), other entries are nodes; ``style`` reaches the entries made
    here and the brackets. ``bracket`` is the left one of "[", "(", "{",
    "|", "‖" or "" for none: Typst's delimiter stretched to the entries'
    height, compiled once per bracket, height and preamble.
    ``entries`` are the rows of entries; :meth:`row` and :meth:`column`
    are lists of them; ``brackets`` the left and right one.
    """

    def __init__(
        self,
        rows: Sequence[Sequence[Entry]],
        preamble: str = "",
        h_buff: float = 0.8,
        v_buff: float = 0.4,
        bracket: str = "[",
        name: str | None = None,
        **style: Any,
    ) -> None:
        if bracket not in BRACKETS:
            raise ValueError(f"bracket must be one of {BRACKETS}, not {bracket!r}")
        super().__init__(name=name)
        self.entries: list[list[Node | None]] = [
            [self._entry(entry, preamble, f"{self.name}.{r}.{c}", style) for c, entry in enumerate(row)] for r, row in enumerate(rows)
        ]
        self.grid = _Grid(self.entries, h_buff, v_buff)
        for r, row in enumerate(self.entries):
            for c, entry in enumerate(row):
                if entry is not None:
                    self.grid.place(entry, r, c)
                    self.add(entry)
        self.brackets: list[Shape2D] = []
        if bracket:
            (left, right), bracket_style = typst_brackets(bracket, round(float(self.grid.top * 2), 3), preamble)
            inner_left, inner_right = self.grid.lefts[0] + h_buff / 4, self.grid.lefts[-1] - h_buff / 4
            for side, points, x in (("left", left, inner_left), ("right", right, inner_right)):
                self.brackets.append(Shape2D(points + [x, 0.0], f"{self.name}.{side}_bracket", **(bracket_style | style)))
            self.add(*self.brackets)

    def _entry(self, entry: Entry, preamble: str, name: str, style: dict[str, Any]) -> Node | None:
        return _entry(entry, lambda text: TypstMath(text, preamble, name=name, **style))

    def row(self, index: int) -> list[Node]:
        return [entry for entry in self.entries[index] if entry is not None]

    def column(self, index: int) -> list[Node]:
        return [row[index] for row in self.entries if index < len(row) and row[index] is not None]  # type: ignore[misc]


class DecimalMatrix(Matrix):
    """A Matrix of numbers that follow signals: every entry is a
    :class:`DecimalNumber` with ``decimals`` places of a number (then
    ``set_values`` animates them) or of a SCALAR signal it shows. Cells are measured on the first values; numbers grow
    from their centers."""

    def __init__(
        self,
        values: Sequence[Sequence[float]],
        decimals: int = 1,
        preamble: str = "",
        h_buff: float = 0.8,
        v_buff: float = 0.4,
        bracket: str = "[",
        name: str | None = None,
        **style: Any,
    ) -> None:
        self.decimals = int(decimals)
        super().__init__(values, preamble, h_buff, v_buff, bracket, name, **style)  # type: ignore[arg-type]

    def _entry(self, entry: Entry, preamble: str, name: str, style: dict[str, Any]) -> Node | None:
        number_style = {key: value for key, value in style.items() if key in ("color", "fill_color", "fill_opacity")}
        value = entry if isinstance(entry, Signal) else float(entry)  # type: ignore[arg-type]
        return DecimalNumber(value, self.decimals, preamble, ORIGIN, name=name, **number_style)

    def values(self) -> Iterator[RootSignal]:
        for row in self.entries:
            for entry in row:
                if isinstance(entry, DecimalNumber) and isinstance(entry.value, RootSignal):
                    yield entry.value

    @animatable
    def set_values(self, values: Sequence[Sequence[float]]) -> ChangeList:
        """Change every entry to the matching one of ``values`` (same shape)."""
        flat = [float(value) for row in values for value in row]
        signals = list(self.values())
        if len(flat) != len(signals):
            raise ValueError(f"{self.name} has {len(signals)} entries, got {len(flat)} values")
        return [(signal, ABS, value) for signal, value in zip(signals, flat, strict=True)]


class IntegerMatrix(DecimalMatrix):
    """A :class:`DecimalMatrix` of whole numbers."""

    def __init__(
        self,
        values: Sequence[Sequence[float]],
        preamble: str = "",
        h_buff: float = 0.8,
        v_buff: float = 0.4,
        bracket: str = "[",
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(values, 0, preamble, h_buff, v_buff, bracket, name, **style)


def _entry(entry: Entry, typeset: Any) -> Node | None:
    if entry is None or isinstance(entry, Node):
        return entry
    if isinstance(entry, Signal):
        raise TypeError("give a signal's number as a DecimalNumber node")
    return typeset(_number_text(entry) if isinstance(entry, (int, float)) else str(entry))


def _number_text(value: float) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "−")


@lru_cache(maxsize=256)
def typst_brackets(bracket: str, height: float, preamble: str = "") -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """The points of the left ``bracket`` and its right partner as Typst
    stretches them to ``height`` (``lr(size: ...)``), centered on y = 0,
    the left one ending at x = 0 on its right and the right one starting
    at x = 0; and the style the document gives them. Typst builds a tall
    delimiter from overlapping pieces; each bracket is their union, one
    outline, so it draws and fills as one shape. Compiled once per
    bracket, height and preamble."""
    right = BRACKETS[bracket]
    escape = lambda delimiter: "\\" + delimiter if delimiter in "[](){}" else delimiter  # noqa: E731
    source = f"{PAGE}{preamble}\n$ lr(size: #{height * 72:.3f}pt, {escape(bracket)} #h(1in) {escape(right)}) $"
    paths = read_svg(compile_typst(source), centered=False)
    everything = np.vstack([path.points for path in paths])
    middle = everything[:, 0].mean()
    center = (everything[:, 1].min() + everything[:, 1].max()) / 2
    left, right_points = (union_points([path.points for path in paths if (path.points[:, 0].mean() < middle) == is_left]) - [0.0, center] for is_left in (True, False))
    return (left - [left[:, 0].max(), 0.0], right_points - [right_points[:, 0].min(), 0.0]), paths[0].style()


__all__ = ["BRACKETS", "DecimalMatrix", "Entry", "IntegerMatrix", "Matrix", "Table", "typst_brackets"]
