"""Typst text and formulas as Shape2D nodes.

:class:`Typst` compiles Typst markup to SVG with the ``typst`` package and
reads it with :mod:`manimx.userland.svg`: a Group with one Shape2D per
glyph, rule and shape, so Write, Create and Morph work on it like on any
other group. Sizes follow Typst's (a scene unit is one inch, see svg.py).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .svg import SVG

PAGE = "#set page(width: auto, height: auto, margin: 0pt, fill: none)\n"
"""Makes the page as large as its content, with no background."""


class Typst(SVG):
    """Typst markup as a Group of Shape2D: ``Typst("Hello $x^2$")``.

    ``preamble`` comes before the markup (``#set text(...)`` rules and
    imports). Colors come from the document (Typst's default is black);
    ``style`` (any Shape2D style argument, ``color`` for one) overrides
    them on every shape. ``width`` and ``height`` scale as in :class:`SVG`.
    """

    def __init__(
        self,
        markup: str,
        preamble: str = "",
        width: float | None = None,
        height: float | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        self.markup = markup
        super().__init__(compile_typst(PAGE + preamble + "\n" + markup), width=width, height=height, name=name, **style)


class TypstMath(Typst):
    """A display formula: ``TypstMath("sum_(k=1)^n k")`` is ``$ ... $``."""

    def __init__(self, formula: str, preamble: str = "", width: float | None = None, height: float | None = None, name: str | None = None, **style: Any) -> None:
        super().__init__(f"$ {formula} $", preamble, width, height, name, **style)


@lru_cache(maxsize=256)
def compile_typst(source: str) -> str:
    """The SVG of the first page of the Typst document ``source``."""
    import typst  # imported here: compiling is the only use

    output = typst.compile(source.encode(), format="svg")
    page = output[0] if isinstance(output, list) else output
    return page.decode()


__all__ = ["PAGE", "Typst", "TypstMath", "compile_typst"]
