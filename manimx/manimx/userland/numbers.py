"""Numbers that follow a signal, laid out from cached glyphs.

:class:`DecimalNumber` shows the value of a SCALAR signal. Typst compiles
each character it can show (the digits, the decimal point and the minus
sign) once per preamble, as a glyph with its advance; the number itself
is laid out here, each character at the sum of the advances before it.
So a number that changes at every frame never runs Typst.

Each character is a Shape2D slot whose points are derived from the
value: the slot shows the glyph at its place in the formatted number, or
nothing if the number is shorter.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..animation import SCALAR, Group, RootSignal, Shape2D, Signal
from ..animation.nodes.node import ABS, ChangeList, animatable
from .constants import ORIGIN
from .svg import UNITS_PER_INCH, _inches, read_svg
from .typst import PAGE, compile_typst

CHARACTERS = "0123456789.−"
"""What a DecimalNumber shows: digits, the decimal point and a minus sign."""

BASELINE_EDGES = '#set text(top-edge: "baseline", bottom-edge: "baseline")\n'
"""Makes a glyph's page start at its pen position on the baseline and as
wide as its advance."""


@dataclass(frozen=True, slots=True)
class Glyph:
    """One character as Typst sets it: ``points`` in scene units with the
    pen position on the baseline as origin, its ``advance`` and the style
    the document gives it."""

    points: np.ndarray
    advance: float
    style: dict[str, Any]


@lru_cache(maxsize=1024)
def glyph(character: str, preamble: str = "") -> Glyph:
    """``character`` set with ``preamble`` (``#set text(...)`` rules), compiled once."""
    import xml.etree.ElementTree as ElementTree

    svg = compile_typst(PAGE + preamble + "\n" + BASELINE_EDGES + character)
    paths = read_svg(svg, centered=False)
    advance = (_inches(ElementTree.fromstring(svg).get("width")) or 0.0) * UNITS_PER_INCH
    points = np.concatenate([path.points for path in paths]) if paths else np.zeros((0, 2))
    return Glyph(points, advance, paths[0].style() if paths else {})


class DecimalNumber(Group):
    """``value`` with ``decimals`` places, set in Typst with ``preamble``
    and following the value at every frame.

    ``value`` is a number or a SCALAR signal to show, such as a Circle's
    ``radius``; showing one, the number is bound to it (``set_value``
    then changes that signal). ``anchor`` says which point of the number
    is its origin: LEFT (the left end, vertically centered on the digits),
    ORIGIN (the middle) or RIGHT, so a number grows away from it.
    ``characters`` is the room it has (default: as many as the first
    value needs, one more for a sign it may get); a longer value raises.
    Colors come from the preamble's text fill, ``style`` overrides them.
    """

    def __init__(
        self,
        value: float | Signal = 0.0,
        decimals: int = 2,
        preamble: str = "",
        anchor: Sequence[float] = ORIGIN,
        characters: int | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        self.value = value if isinstance(value, Signal) else RootSignal(float(value), SCALAR, name=f"{self.name}.value")
        self.layout = NumberLayout(decimals, preamble, float(anchor[0]))
        first = self.layout.text(float(self.value.value))
        room = characters if characters is not None else len(first) + (0 if first.startswith("−") else 1)
        glyph_style = glyph("0", preamble).style | style
        self.slots = [Shape2D((), name=f"{self.name}.{index}", **glyph_style) for index in range(room)]
        for index, slot in enumerate(self.slots):
            slot._derive_points(_Slot(self.layout, index, room), self.value)
        self.add(*self.slots)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        if isinstance(self.value, RootSignal):
            yield self.value

    @animatable
    def set_value(self, value: float) -> ChangeList:
        if not isinstance(self.value, RootSignal):
            raise TypeError(f"{self.name} shows {self.value.name}, which is derived; change its inputs instead")
        return [(self.value, ABS, float(value))]


class NumberLayout:
    """Where each character of a formatted number goes."""

    __slots__ = ("decimals", "preamble", "anchor", "_center")

    def __init__(self, decimals: int, preamble: str, anchor: float) -> None:
        self.decimals, self.preamble, self.anchor = int(decimals), preamble, anchor
        zero = glyph("0", preamble).points
        self._center = (zero[:, 1].min() + zero[:, 1].max()) / 2 if len(zero) else 0.0

    def text(self, value: float) -> str:
        text = f"{value:.{self.decimals}f}"
        if float(text) == 0.0:
            text = text.lstrip("-")  # no "-0.00"
        return text.replace("-", "−")

    @lru_cache(maxsize=64)  # noqa: B019 - one layout per DecimalNumber, kept as long as it
    def placed(self, value: float) -> tuple[np.ndarray, ...]:
        """The points of every character of ``value``, in place."""
        glyphs = [glyph(character, self.preamble) for character in self.text(value)]
        pens = np.cumsum([0.0] + [g.advance for g in glyphs])
        width = pens[-1]
        shift = np.array([-width * (self.anchor + 1) / 2, -self._center])
        return tuple(g.points + [pen, 0.0] + shift for g, pen in zip(glyphs, pens[:-1], strict=True))


class _Slot:
    """The points of character ``index`` of the number (a slot's derivation)."""

    __slots__ = ("layout", "index", "room")

    def __init__(self, layout: NumberLayout, index: int, room: int) -> None:
        self.layout, self.index, self.room = layout, index, room

    def __call__(self, value: float) -> np.ndarray:
        placed = self.layout.placed(float(value))
        if len(placed) > self.room:
            raise ValueError(f"{self.layout.text(value)!r} needs {len(placed)} characters; the DecimalNumber has room for {self.room} (give characters=)")
        return placed[self.index] if self.index < len(placed) else np.zeros((0, 2))


__all__ = ["CHARACTERS", "DecimalNumber", "Glyph", "NumberLayout", "glyph"]
