"""Text: Typst markup and formulas as shapes, written and morphed.

Typst and TypstMath compile with Typst and become one Shape2D per
glyph, so Write draws them and Morph turns one formula into another.
SVG reads a vector file the same way.
"""

from __future__ import annotations

from pathlib import Path

from manimx.animation import Scene
from manimx.userland import (
    DOWN,
    SVG,
    UP,
    Create,
    FadeIn,
    FadeOut,
    Morph,
    Typst,
    TypstMath,
    Write,
)

from .style import INK, RED, caption

STYLE = f'#set text(size: 34pt, fill: rgb("{INK}"))'
LOGO = Path(__file__).parents[1] / "assets" / "logo.svg"


class Text(Scene):
    def construct(self) -> None:
        title = caption("Text", "Typst markup and formulas, Write, Morph, and SVG files")
        self.play(FadeIn(title), run_time=0.6)

        line = Typst(f'Shapes from #text(fill: rgb("{RED}"))[Typst]: every glyph a Shape2D', STYLE).shift(UP * 1.6)
        self.play(Write(line), run_time=2.0)

        first = TypstMath("e^(i pi) + 1 = 0", STYLE).shift(DOWN * 0.2)
        second = TypstMath("e^(i theta) = cos theta + i sin theta", STYLE).shift(DOWN * 0.2)
        third = TypstMath("integral_0^infinity e^(-x^2) dif x = sqrt(pi) / 2", STYLE).shift(DOWN * 0.2)
        self.play(Write(first), run_time=1.2)
        self.play(Morph(first, second), run_time=1.5)
        self.wait(0.3)
        self.play(Morph(second, third), run_time=1.5)

        logo = SVG(LOGO, height=1.3).shift(DOWN * 2.4)
        self.play(Create(logo), run_time=1.5)
        self.wait(0.5)
        self.play(*(FadeOut(node) for node in (line, third, logo, title)), run_time=0.6)
