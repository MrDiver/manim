"""Formulas: Typst text and an SVG file, written, morphed and faded.

Shows ``Typst`` and ``TypstMath`` (markup compiled to Shape2D glyphs),
``Write`` on text, ``Morph`` from one formula into another, and ``SVG``
reading a file from the project's assets.
"""

from __future__ import annotations

from pathlib import Path

from manimx.animation import Scene
from manimx.userland import (
    DOWN,
    SVG,
    UP,
    Create,
    FadeOut,
    Morph,
    Typst,
    TypstMath,
    Write,
)

INK = "#ece6e2"
ACCENT = "#e07a5f"
STYLE = f'#set text(size: 36pt, fill: rgb("{INK}"))'
BADGE = Path(__file__).parents[1] / "assets" / "badge.svg"


class Formulas(Scene):
    def construct(self) -> None:
        title = Typst(f'Typst in #text(fill: rgb("{ACCENT}"))[manimx]', STYLE).shift(UP * 3)
        self.play(Write(title), run_time=1.5)

        first = TypstMath("(a + b)^2", STYLE)
        second = TypstMath("a^2 + 2 a b + b^2", STYLE)
        third = TypstMath("sum_(k=0)^2 binom(2, k) a^(2-k) b^k", STYLE)
        self.play(Write(first))
        self.wait(0.5)
        self.play(Morph(first, second), run_time=1.5)
        self.wait(0.5)
        self.play(Morph(second, third), run_time=1.5)
        self.wait(0.5)

        badge = SVG(BADGE, height=1.6).shift(DOWN * 2.6)
        self.play(Create(badge), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(title), FadeOut(third), FadeOut(badge))
