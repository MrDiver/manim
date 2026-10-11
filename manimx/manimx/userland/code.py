"""Source code as Shape2D glyphs on a grid of rows and columns.

:class:`Code` sets source code with Typst's ``raw`` element, which
highlights it (Typst ships syntaxes for most languages), and reads the
result like :class:`~manimx.userland.typst.Typst`: one Shape2D per glyph.
Code is set in a monospace font, so every glyph has a cell: the row and
column of its character in the source. The cell comes from where the
SVG places the glyph (``SVGPath.origin``), so the glyph's character is
known without matching the PDF's text.

On that grid, code can be focused (``code.focus(code.line(3))`` dims
everything else), boxed (``code.region(...)`` is the rectangle around
some lines or characters), clipped to a window of rows and scrolled
(``code.clip(rows=6)``, ``code.scroll_to(10)``), and :class:`MorphCode`
turns one version of a program into the next: characters of the tokens
both versions share move to their new cells, the rest fade out and in.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..animation import ClippingGroup, Group, Node, Shape2D
from ..animation.authoring import current as authoring_current
from ..animation.nodes.node import ABS, ChangeList, animatable
from ..utils.color import ManimColor, ParsableManimColor
from .animations import Morph, _world_alpha
from .shapes2d import Rectangle
from .svg import read_svg
from .typst import PAGE, compile_typst

TAB_SIZE = 4
"""Columns a tab stands for (tabs become spaces before Typst sees them)."""


@dataclass(frozen=True, slots=True)
class Grid:
    """The cells of code set with one preamble and theme, in the code's
    own space: column ``c`` of row ``r`` has its pen position at
    ``(left + c * advance, baseline - r * pitch)``."""

    left: float
    baseline: float
    advance: float
    """Width of a column."""
    pitch: float
    """Distance from one row's baseline to the next."""
    cap_height: float
    """Height of a digit above the baseline (for boxes around rows)."""

    def pen(self, row: int, column: int) -> np.ndarray:
        return np.array([self.left + column * self.advance, self.baseline - row * self.pitch])

    def cell(self, row: int, column: int) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` of a cell: a row is one pitch
        high, centered on its digits."""
        x, y = self.pen(row, column)
        middle = y + self.cap_height / 2
        return (x, middle - self.pitch / 2, x + self.advance, middle + self.pitch / 2)


class Code(Group):
    """``source`` highlighted as ``language`` (a Typst ``raw`` language
    name such as ``"python"`` or ``"haskell"``; None: plain text).

    A Group with one Group per line of ``source`` (``code.line(n)``,
    counted from 1 like an editor does) holding a Shape2D per glyph, named
    by its character. The code's origin is the top left corner of its
    first cell, so versions of a program placed at the same point line up.

    ``preamble`` comes before the code (``#set text(...)``; raw text is
    sized with ``#show raw: set text(size: ...)``). Colors: ``colors``
    maps TextMate scopes to colors (``{"keyword": ..., "comment": ...}``,
    the most specific scope wins), ``theme`` is a .tmTheme file; with
    neither, Typst's own theme colors the code. Text no scope colors has
    the preamble's text fill. ``style`` (any Shape2D style argument)
    overrides the colors on every glyph.
    """

    def __init__(
        self,
        source: str,
        language: str | None = None,
        preamble: str = "",
        colors: Mapping[str, ParsableManimColor] | None = None,
        theme: str | Path | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        self.source = source.expandtabs(TAB_SIZE)
        self.rows = self.source.split("\n")
        setup = preamble + "\n" + _theme_rule(colors, theme)
        self.grid = _grid(setup)
        self.glyphs: list[Shape2D] = []
        """Every glyph, in reading order."""
        self.cells: dict[int, tuple[int, int]] = {}
        """``(row, column)`` of each glyph, by uid (rows and columns from 0)."""
        self.lines = [Group(name=f"{self.name}.line{number}") for number in range(1, len(self.rows) + 1)]
        for (row, column), points, path_style in _set(self.source, language, setup, self.grid):
            character = self.rows[row][column]
            glyph = Shape2D(points, character, **(path_style | style))
            self.cells[glyph.uid] = (row, column)
            self.glyphs.append(glyph)
            self.lines[row].add(glyph)
        self.add(*self.lines)
        self._scroll_home: float | None = None

    # -- finding parts ------------------------------------------------------

    def line(self, number: int) -> Group:
        """The Group of line ``number`` (from 1)."""
        if not 1 <= number <= len(self.lines):
            raise IndexError(f"{self.name} has lines 1 to {len(self.lines)}, no line {number}")
        return self.lines[number - 1]

    def select(self, text: str, occurrence: int = 0, line: int | None = None) -> list[Shape2D]:
        """The glyphs of the ``occurrence``-th appearance (from 0) of
        ``text`` within a line of the source, or within line ``line``."""
        rows = range(len(self.rows)) if line is None else [self.lines.index(self.line(line))]
        found = [(row, match.start()) for row in rows for match in re.finditer(re.escape(text), self.rows[row])]
        if not text or occurrence >= len(found):
            where = f"line {line} of {self.name}" if line is not None else self.name
            raise LookupError(f"{text!r} appears {len(found)} times in {where}, no occurrence {occurrence}")
        row, column = found[occurrence]
        wanted = {(row, column + offset) for offset in range(len(text))}
        return [glyph for glyph in self.glyphs if self.cells[glyph.uid] in wanted]

    def _glyphs_of(self, parts: Iterable[Node | Iterable[Node]]) -> list[Shape2D]:
        wanted: set[int] = set()
        for part in parts:
            for node in [part] if isinstance(part, Node) else part:
                wanted.update(member.uid for member in node.family())
        return [glyph for glyph in self.glyphs if glyph.uid in wanted]

    # -- focus and boxes ----------------------------------------------------

    @animatable
    def focus(self, *parts: Node | Iterable[Node], dim: float = 0.25) -> ChangeList:
        """Show ``parts`` (lines, glyphs, lists of glyphs) fully and the
        rest of the code at ``dim`` opacity."""
        chosen = {glyph.uid for glyph in self._glyphs_of(parts)}
        return [(glyph.alpha, ABS, 1.0 if glyph.uid in chosen else dim) for glyph in self.glyphs]

    @animatable
    def unfocus(self) -> ChangeList:
        """Show all of the code fully again."""
        return [(glyph.alpha, ABS, 1.0) for glyph in self.glyphs]

    def region(self, *parts: Node | Iterable[Node], padding: float = 0.1) -> tuple[np.ndarray, float, float]:
        """``(center, width, height)`` of the box around the cells of
        ``parts``, ``padding`` larger on every side, where the code is now
        (in its parent's space; rotation is not followed). A Rectangle
        made with that size and moved to that center frames them, and
        ``rectangle.animate.move_to(center).set_size(width, height)``
        moves it on to other parts."""
        cells = [self.cells[glyph.uid] for glyph in self._glyphs_of(parts)]
        if not cells:
            raise ValueError("region() needs at least one glyph")
        boxes = np.array([self.grid.cell(row, column) for row, column in cells])
        low = np.array([boxes[:, 0].min(), boxes[:, 1].min()]) - padding
        high = np.array([boxes[:, 2].max(), boxes[:, 3].max()]) + padding
        scale = np.array([authoring_current(axis) for axis in self.scale.axes])
        center = self.get_center() + np.append((low + high) / 2, 0.0) * scale
        size = (high - low) * scale[:2]
        return center, float(size[0]), float(size[1])

    # -- windows ------------------------------------------------------------

    def clip(self, rows: int, first: int = 1, width: float | None = None, padding: float = 0.1, name: str | None = None) -> ClippingGroup:
        """A ClippingGroup showing ``rows`` lines of the code from line
        ``first`` on, as wide as the longest line (or ``width``) and
        ``padding`` wider on each side, where the code is now. Add the
        window to the scene instead of the code, and :meth:`scroll_to` a
        line to move the code under it."""
        top, bottom = self.grid.cell(first - 1, 0)[3], self.grid.cell(first + rows - 2, 0)[1]
        columns = max((len(row) for row in self.rows), default=0)
        left = self.grid.left - padding
        right = self.grid.left + (width if width is not None else columns * self.grid.advance) + padding
        scale = np.array([authoring_current(axis) for axis in self.scale.axes])
        mask = Rectangle((right - left) * scale[0], (top - bottom) * scale[1], name=f"{self.name}.window", fill_opacity=1.0, stroke_width=0.0)
        mask.move_to(self.get_center() + np.array([(left + right) / 2 * scale[0], (top + bottom) / 2 * scale[1], 0.0]))
        self._scroll_home = float(authoring_current(self.position.y)) - (first - 1) * self.grid.pitch * scale[1]
        return ClippingGroup(mask, self, name or f"{self.name}.clip")

    @animatable
    def scroll_to(self, line: int) -> ChangeList:
        """Move the code so that line ``line`` is the first one its window
        (:meth:`clip`) shows."""
        if self._scroll_home is None:
            raise RuntimeError(f"{self.name} has no window to scroll in; make one with clip() first")
        scale = float(authoring_current(self.scale.y))
        return [(self.position.y, ABS, self._scroll_home + (line - 1) * self.grid.pitch * scale)]


@dataclass(eq=False)
class MorphCode(Morph[Code]):
    """Turn one version of a program into the next, the way slides of
    code change: characters of the tokens both versions share (found by a
    diff of the token sequences, so order is kept) move to their new
    cells and take on their new color; the source's other characters fade
    out in the first ``fade`` of the morph, the target's fade in in the
    last ``fade``. Like :class:`Morph`, it changes neither side and
    leaves ``target`` in the scene."""

    fade: float = 0.4

    def _pair(self, sources: list[Node], targets: list[Node]) -> None:
        source, target = self.source, self.target
        if not isinstance(source, Code) or not isinstance(target, Code):
            raise TypeError("MorphCode turns a Code into a Code")
        matched = _matching_cells(source.source, target.source)
        by_cell = {cell: glyph for glyph in target.glyphs for cell in [target.cells[glyph.uid]]}
        taken: set[int] = set()
        for glyph in source.glyphs:
            partner = by_cell.get(matched.get(source.cells[glyph.uid], (-1, -1)))
            if partner is not None:
                taken.add(partner.uid)
                self._inbetween(glyph, partner)
            else:
                self._inbetween(glyph, glyph, (_world_alpha(glyph), 0.0), window=(0.0, self.fade))
        for glyph in target.glyphs:
            if glyph.uid not in taken:
                self._inbetween(glyph, glyph, (0.0, _world_alpha(glyph)), window=(1.0 - self.fade, 1.0))


# --------------------------------------------------------------------------
# Setting code with Typst
# --------------------------------------------------------------------------


def _typst_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _raw(source: str, language: str | None) -> str:
    language_argument = f", lang: {_typst_string(language)}" if language else ""
    return f"#raw(block: true{language_argument}, {_typst_string(source)})"


def _theme_rule(colors: Mapping[str, ParsableManimColor] | None, theme: str | Path | None) -> str:
    """The ``#set raw(theme: ...)`` rule for ``colors`` or a theme file."""
    if colors is not None and theme is not None:
        raise ValueError("give colors or a theme, not both")
    if theme is not None:
        text = Path(theme).read_text()
    elif colors is not None:
        text = _tm_theme(colors)
    else:
        return ""
    return f"#set raw(theme: bytes({_typst_string(text)}))"


def _tm_theme(colors: Mapping[str, ParsableManimColor]) -> str:
    """A TextMate theme (the format Typst reads) giving each scope its color.
    https://macromates.com/manual/en/themes"""
    rules = "".join(
        f"<dict><key>scope</key><string>{scope}</string><key>settings</key><dict><key>foreground</key>"
        f"<string>{ManimColor.parse(color).to_hex()}</string></dict></dict>"
        for scope, color in colors.items()
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict><key>settings</key><array><dict><key>settings</key><dict/></dict>{rules}</array></dict></plist>'


@lru_cache(maxsize=64)
def _grid(setup: str) -> Grid:
    """The grid of code set with ``setup``, measured on a probe: "00" over
    "0" gives the advance, the pitch and the first pen position."""
    paths = read_svg(compile_typst(PAGE + setup + "\n" + _raw("00\n0", None)), centered=False)
    origins = {path.glyph: path.origin for path in paths if path.origin is not None}
    first, second, below = (np.array(origins[index]) for index in (0, 1, 2))
    zero = np.concatenate([path.points for path in paths if path.glyph == 0])
    return Grid(
        left=float(first[0]),
        baseline=float(first[1]),
        advance=float(second[0] - first[0]),
        pitch=float(first[1] - below[1]),
        cap_height=float(zero[:, 1].max() - first[1]),
    )


def _set(source: str, language: str | None, setup: str, grid: Grid) -> list[tuple[tuple[int, int], np.ndarray, dict[str, Any]]]:
    """``((row, column), points, style)`` of every glyph of ``source``, in
    reading order. The cell is where the glyph's pen position falls on
    ``grid``; one glyph covering several characters (a ligature) would
    take the first one's cell, so fonts with ligatures should turn them
    off (``#set text(ligatures: false)``)."""
    paths = read_svg(compile_typst(PAGE + setup + "\n" + _raw(source, language)), centered=False)
    by_glyph: dict[int, list[Any]] = {}
    for path in paths:
        if path.glyph is not None and path.origin is not None:
            by_glyph.setdefault(path.glyph, []).append(path)
    rows = source.split("\n")
    glyphs = []
    for parts in by_glyph.values():
        x, y = parts[0].origin
        cell = (round((grid.baseline - y) / grid.pitch), round((x - grid.left) / grid.advance))
        if not (0 <= cell[0] < len(rows) and 0 <= cell[1] < len(rows[cell[0]])):
            raise ValueError(f"a glyph of the code falls outside its text at row {cell[0]}, column {cell[1]}; is the font monospace?")
        glyphs.append((cell, np.concatenate([part.points for part in parts]), parts[0].style()))
    glyphs.sort(key=lambda glyph: glyph[0])
    return glyphs


# --------------------------------------------------------------------------
# Matching two versions of a program
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"\w+|\S")
"""Words and single other characters."""


def _tokens(source: str) -> list[tuple[str, int, int]]:
    """``(text, row, column)`` of every token of ``source``."""
    return [(match.group(), row, match.start()) for row, line in enumerate(source.split("\n")) for match in _TOKEN.finditer(line)]


def _matching_cells(source: str, target: str) -> dict[tuple[int, int], tuple[int, int]]:
    """The target cell of every source character that a token both
    versions share, found by a diff of the token sequences, keeps.
    https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher"""
    before, after = _tokens(source), _tokens(target)
    matcher = difflib.SequenceMatcher(None, [token for token, _, _ in before], [token for token, _, _ in after], autojunk=False)
    cells: dict[tuple[int, int], tuple[int, int]] = {}
    for start, end, size in matcher.get_matching_blocks():
        for (text, row, column), (_, new_row, new_column) in zip(before[start : start + size], after[end : end + size], strict=True):
            for offset in range(len(text)):
                cells[(row, column + offset)] = (new_row, new_column + offset)
    return cells


__all__ = ["Code", "Grid", "MorphCode", "TAB_SIZE"]
