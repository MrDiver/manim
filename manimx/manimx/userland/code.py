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

The grid also makes text animations of editors and boards simple:
:class:`CycleCharacters` turns code (or plain text) into other text by
counting every cell through an alphabet, with glyphs compiled once each
(:func:`glyph_points`), and :class:`Flipboard` folds each step like the
flap of a split-flap display; :class:`TypeCode` and
:class:`Untype` type it in and take it back key by key behind a cursor
(``code.cursor()``), which :class:`Blink` blinks.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..animation import (
    SCALAR,
    Animation,
    Change,
    ClippingGroup,
    Group,
    Node,
    PositionMixin,
    RootSignal,
    Shape2D,
    Signal,
    Sound,
    Target,
)
from ..animation.authoring import current as authoring_current
from ..animation.nodes.node import ABS, ChangeList, animatable, require
from ..animation.signals.contributions import Contribution
from ..animation.signals.rates import linear
from ..animation.signals.signal_tracks import RateFunction
from ..utils.color import ManimColor, ParsableManimColor
from .animations import Morph, _lagged, _PlannedAnimation, _world_alpha
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
        x, middle = self.caret(row, column)
        return (x, middle - self.pitch / 2, x + self.advance, middle + self.pitch / 2)

    def caret(self, row: int, column: int) -> np.ndarray:
        """The middle of the cell's left edge: where a text cursor before
        the cell stands."""
        x, y = self.pen(row, column)
        return np.array([x, y + self.cap_height / 2])


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
        self.setup = setup
        """The Typst rules the code is set with (preamble and theme)."""
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
        size = (high - low) * self._scale()[:2]
        return self.in_parent((low + high) / 2), float(size[0]), float(size[1])

    def in_parent(self, point: Sequence[float]) -> np.ndarray:
        """``point`` of the code's own space (such as a cell corner of
        :attr:`grid`) in its parent's space, where the code is now
        (rotation is not followed)."""
        return self.get_center() + np.append(np.asarray(point, dtype=np.float64)[:2], 0.0) * self._scale()

    def _scale(self) -> np.ndarray:
        return np.array([authoring_current(axis) for axis in self.scale.axes])

    def cursor(self, width: float = 0.12, height: float = 0.8, **style: Any) -> Rectangle:
        """A text cursor for :class:`TypeCode`: a bar ``width`` columns
        wide and ``height`` rows high, on the left edge of the first cell.
        ``style`` as for any Shape2D (it has no color of its own)."""
        scale = self._scale()
        bar = Rectangle(width * self.grid.advance * scale[0], height * self.grid.pitch * scale[1], name=f"{self.name}.cursor", **style)
        return bar.move_to(self.in_parent(self.grid.caret(0, 0)))

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
        scale = self._scale()
        mask = Rectangle((right - left) * scale[0], (top - bottom) * scale[1], name=f"{self.name}.window", fill_opacity=1.0, stroke_width=0.0)
        mask.move_to(self.in_parent(((left + right) / 2, (top + bottom) / 2)))
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


ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?'\"-+*/=<>()[]{}_#@&%$|\\^~`"
"""The order :class:`CycleCharacters` counts characters in, like the flaps of
a split-flap display: a space (an empty flap) first."""


@dataclass(eq=False)
class CycleCharacters(_PlannedAnimation):
    """Turn ``source`` into ``target`` cell by cell, each cell counting
    from its character to the target's through ``alphabet``, up or down,
    one character after another. All cells count at the same speed, so
    short ways arrive first; ``lag_ratio`` starts the columns one after
    another instead. A character not in the alphabet changes straight to
    its target. Colors go from the source's to the target's. Plain text
    counts too: it is ``Code`` without a language.

    The cells are ``target``'s: place both at the same point. Like
    :class:`Morph`, it changes neither side; ``target`` takes ``source``'s
    place in the scene when it ends.
    """

    source: Code
    target: Code
    alphabet: str = ALPHABET
    lag_ratio: float = 0.0
    run_time: float = 1.5
    rate: RateFunction = linear
    sounds: Sequence[str | Path | Sound] = ()
    """Sounds to play as the cells change, one picked at random (by
    ``seed``) for each step; steps of several cells at once sound once."""
    volume: float = 1.0
    seed: int = 0

    fold = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, Code) or not isinstance(self.target, Code):
            raise TypeError(f"{type(self).__name__} turns a Code into a Code")
        self._plan()
        before = {self.source.cells[glyph.uid]: glyph for glyph in self.source.glyphs}
        after = {self.target.cells[glyph.uid]: glyph for glyph in self.target.glyphs}
        cells = sorted(set(before) | set(after))
        ways = {cell: _flaps(_character(self.source, cell), _character(self.target, cell), self.alphabet) for cell in cells}
        span = max((len(way) - 1 for way in ways.values()), default=0)
        columns = sorted({column for _, column in cells})
        windows = dict(zip(columns, _lagged(len(columns), self.lag_ratio), strict=True)) if columns else {}
        self._ticks = _ticks(ways, span, windows, 0.5 if self.fold else 0.0)
        self.board = _Board(self.target, name=f"{self.target.name}.cells")
        grid = self.target.grid
        for cell in cells:
            ends = [glyph for glyph in (before.get(cell), after.get(cell)) if glyph is not None]
            first, last = (before.get(cell) or ends[0]), (after.get(cell) or ends[-1])
            slot = Shape2D((), f"{self.board.name}.{cell[0]}.{cell[1]}", **_glyph_style(last))
            slot._derive_points(_Flap(ways[cell], span, windows[cell[1]], grid.pen(*cell), grid.cap_height / 2, self.target.setup, self.fold), self.board.progress)
            for name in ("fill_color", "stroke_color"):
                self._between(getattr(slot, name), authoring_current(getattr(first, name)), authoring_current(getattr(last, name)))
            # A space has no glyph to fade: a cell counting from or to one keeps the other side's alpha.
            alpha = (_world_alpha(first), _world_alpha(last))
            slot.alpha.set(alpha[0])
            self._between(slot.alpha, *alpha)
            self.board.add(slot)
        self._between(self.board.progress, 0.0, 1.0)
        self._between(self.source.alpha, authoring_current(self.source.alpha), 0.0, 0.0, 0.0)
        self._between(self.target.alpha, 0.0, authoring_current(self.target.alpha), 1.0, 1.0)

    def nodes(self) -> Sequence[Node]:
        return (self.source, self.target, self.board)

    def removes(self) -> Sequence[Node]:
        return (self.source, self.board)

    def audio(self) -> Sequence[tuple[Sound, float, float]]:
        return _random_sounds(self.sounds, self.volume, self.seed, self._ticks)


@dataclass(eq=False)
class Flipboard(CycleCharacters):
    """:class:`CycleCharacters` as a split-flap display shows it: each
    character folds away about its row's middle and the next one folds
    in, as a flap falls."""

    fold = True


class _Board(Group):
    """The cells of a CycleCharacters while it plays, where ``code`` is, and the
    progress they flip by."""

    def __init__(self, code: Code, name: str) -> None:
        super().__init__(name=name)
        self.progress = RootSignal(0.0, SCALAR, name=f"{name}.progress")
        self.move_to(code.get_center())
        self.set_scale(code._scale())

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.progress


def _character(code: Code, cell: tuple[int, int]) -> str:
    row, column = cell
    return code.rows[row][column] if row < len(code.rows) and column < len(code.rows[row]) else " "


def _flaps(start: str, end: str, alphabet: str) -> str:
    """The characters a cell shows on its way from ``start`` to ``end``."""
    if start == end:
        return start
    if start not in alphabet or end not in alphabet:
        return start + end
    first, last = alphabet.index(start), alphabet.index(end)
    step = 1 if last > first else -1
    return "".join(alphabet[index] for index in range(first, last + step, step))


def _glyph_style(glyph: Shape2D) -> dict[str, Any]:
    names = ("fill_opacity", "stroke_width", "stroke_opacity")
    return {name: authoring_current(getattr(glyph, name)) for name in names}


@lru_cache(maxsize=4096)
def glyph_points(character: str, setup: str) -> np.ndarray:
    """``character`` set as code with ``setup``, its pen position at the
    origin (empty for a space); each compiled once."""
    if not character.strip():
        return np.zeros((0, 2))
    paths = [path for path in read_svg(compile_typst(PAGE + setup + "\n" + _raw(character, None)), centered=False) if path.origin is not None]
    if not paths:
        return np.zeros((0, 2))
    return np.concatenate([path.points for path in paths]) - np.array(paths[0].origin)


class _Flap:
    """The points of a CycleCharacters cell at a progress (its
    derivation): the character it has counted to, folded while it changes
    for a Flipboard."""

    __slots__ = ("way", "span", "window", "pen", "middle", "setup", "fold")

    def __init__(self, way: str, span: int, window: tuple[float, float], pen: np.ndarray, middle: float, setup: str, fold: bool) -> None:
        self.way, self.span, self.window, self.pen, self.middle, self.setup, self.fold = way, span, window, pen, middle, setup, fold

    def __call__(self, progress: float) -> np.ndarray:
        lower, upper = self.window
        local = min(1.0, max(0.0, (progress - lower) / (upper - lower))) if upper > lower else float(progress >= upper)
        steps = len(self.way) - 1
        reached = local * self.span
        if reached >= steps:
            return glyph_points(self.way[-1], self.setup) + self.pen
        flap = int(reached)
        turned = reached - flap
        if not self.fold:
            return glyph_points(self.way[flap], self.setup) + self.pen
        character = self.way[flap] if turned < 0.5 else self.way[flap + 1]
        height = max(abs(math.cos(math.pi * turned)), 0.02)
        return glyph_points(character, self.setup) * [1.0, height] + [0.0, self.middle * (1.0 - height)] + self.pen


@dataclass(eq=False)
class TypeCode(_PlannedAnimation):
    """Type ``code`` in, key by key: each character appears when its key
    is pressed, in reading order, spaces and line breaks taking a key
    press too. ``cursor`` (a node with a position, such as
    ``code.cursor(color=...)``) stands after the last character typed.
    ``speed`` is in key presses per second and sets the run time unless
    one is given; ``jitter`` (0 to 1) varies the time between presses, as
    a hand does, the same way for the same ``seed``. ``sounds`` (files or
    Sounds) are key clicks: each press plays one picked at random by
    ``seed``, at ``volume``.
    """

    code: Code
    cursor: Node | None = None
    speed: float = 20.0
    jitter: float = 0.0
    seed: int = 0
    run_time: float | None = None  # type: ignore[assignment]
    rate: RateFunction = linear
    sounds: Sequence[str | Path | Sound] = ()
    volume: float = 1.0

    backwards = False

    def __post_init__(self) -> None:
        if self.cursor is not None:
            require(self.cursor, PositionMixin, type(self).__name__)
        self._plan()
        keys = [(row, column) for row, line in enumerate(self.code.rows) for column in range(len(line) + 1)][:-1]
        if self.run_time is None:
            self.run_time = max(len(keys), 1) / self.speed
        pauses = 1.0 + self.jitter * (2.0 * np.random.default_rng(self.seed).random(len(keys)) - 1.0)
        presses = np.concatenate([[0.0], np.cumsum(pauses)])[:-1] / max(float(pauses.sum()), 1e-12)
        if self.backwards:
            presses = 1.0 - presses  # the last key is taken back first
        self._presses = np.sort(presses)
        self._carets = [self.code.in_parent(self.code.grid.caret(*key)) for key in keys]
        self._carets.append(self.code.in_parent(self.code.grid.caret(*keys[-1]) + [self.code.grid.advance, 0.0]) if keys else self.code.get_center())
        press_of = {key: index for index, key in enumerate(keys)}
        for glyph in self.code.glyphs:
            press = float(presses[press_of[self.code.cells[glyph.uid]]])
            shown = authoring_current(glyph.alpha)
            self._between(glyph.alpha, *((shown, 0.0) if self.backwards else (0.0, shown)), press, press)

    def targets(self) -> Sequence[Target]:
        moves = [Target(axis, Change.ABSOLUTE) for axis in self.cursor.position.axes] if self.cursor is not None else []
        return [*super().targets(), *moves]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        if signal.id in self._planned:
            return super().contribution(signal, alpha, start_value)
        typed = int(np.searchsorted(self._presses, alpha, side="right"))
        if self.backwards:
            typed = len(self._presses) - typed
        return float(self._carets[typed][self.cursor.position.axes.index(signal)])

    def planned_contribution(self, signal: RootSignal) -> Contribution | None:
        return self._planned[signal.id][1] if signal.id in self._planned else None

    def nodes(self) -> Sequence[Node]:
        return (self.code,) if self.cursor is None else (self.code, self.cursor)

    def audio(self) -> Sequence[tuple[Sound, float, float]]:
        return _random_sounds(self.sounds, self.volume, self.seed, list(self._presses))


@dataclass(eq=False)
class Untype(TypeCode):
    """Take ``code`` back key by key from its end, as backspace does, the
    cursor going along, and remove it (the cursor stays)."""

    backwards = True

    def removes(self) -> Sequence[Node]:
        return (self.code,)


@dataclass(eq=False)
class Blink(Animation):
    """Blink a node (a cursor) ``times`` times: it is hidden for the
    second half of each blink, and shown at the end."""

    node: Node
    times: int = 3
    run_time: float = 2.0
    rate: RateFunction = linear

    def targets(self) -> Sequence[Target]:
        return [Target(self.node.alpha, Change.ABSOLUTE)]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        return 1.0 if alpha >= 1.0 or math.floor(alpha * self.times * 2) % 2 == 0 else 0.0

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


def _ticks(ways: Mapping[tuple[int, int], str], span: int, windows: Mapping[int, tuple[float, float]], offset: float) -> list[float]:
    """When cells change character, as fractions of the run time (with a
    linear rate): step ``k`` of a cell at ``k - offset`` of ``span`` steps
    into its column's window. Steps closer than 1% of the run time count
    once, so a board changing all at once clicks once per step."""
    times = sorted(
        lower + (step - offset) / span * (upper - lower)
        for (_, column), way in ways.items()
        for lower, upper in [windows[column]]
        for step in range(1, len(way))
    ) if span else []  # fmt: skip
    ticks: list[float] = []
    for time in times:
        if not ticks or time - ticks[-1] >= 0.01:
            ticks.append(time)
    return ticks


@lru_cache(maxsize=64)
def _load_sound(path: str) -> Sound:
    return Sound.load(path)


def _random_sounds(sounds: Sequence[str | Path | Sound], volume: float, seed: int, times: Sequence[float]) -> list[tuple[Sound, float, float]]:
    """One of ``sounds`` at each of ``times``, picked at random by ``seed``."""
    if not sounds:
        return []
    loaded = [sound if isinstance(sound, Sound) else _load_sound(str(sound)) for sound in sounds]
    picks = np.random.default_rng(seed).integers(0, len(loaded), len(times))
    return [(loaded[int(pick)], float(volume), float(time)) for pick, time in zip(picks, times, strict=True)]


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


__all__ = ["ALPHABET", "TAB_SIZE", "Blink", "Code", "CycleCharacters", "Flipboard", "Grid", "MorphCode", "TypeCode", "Untype", "glyph_points"]
