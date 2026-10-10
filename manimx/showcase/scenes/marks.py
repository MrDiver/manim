"""Marks: rectangles, underlines, crosses and braces around other nodes.

Every mark follows bounds(node), a signal of the node's bounding box:
when the formula moves and grows, the rectangle, the underline and the
brace with its label go with it.
"""

from __future__ import annotations

from manimx.animation import Scene
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    BackgroundRectangle,
    BraceLabel,
    Create,
    Cross,
    FadeIn,
    FadeOut,
    Star,
    SurroundingRectangle,
    Typst,
    TypstMath,
    Underline,
    Write,
)

from .style import BLUE, INK, MUTED, NAVY, RED, SAND, caption

TEXT = f'#set text(size: 30pt, fill: rgb("{INK}"))'
SMALL = f'#set text(size: 18pt, fill: rgb("{MUTED}"))'


class Marks(Scene):
    def construct(self) -> None:
        title = caption("Marks", "Rectangles, underlines, crosses and braces that follow what they mark")
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.8)

        formula = TypstMath("a^2 + b^2 = c^2", TEXT).move_to(LEFT * 3 + UP * 0.6)
        behind = BackgroundRectangle(formula, buff=0.3, corner_radius=0.15, color=NAVY)
        around = SurroundingRectangle(formula, buff=0.3, corner_radius=0.15, color=SAND, stroke_width=3.0)
        braced = BraceLabel(formula, Typst("Pythagoras", SMALL), DOWN, buff=0.45, color=BLUE, stroke_width=[1.0, 5.0, 1.0])
        self.play(FadeIn(behind), Write(formula), run_time=1.2)
        self.play(Create(around), FadeIn(braced), run_time=1.0)
        self.play(formula.animate.move_to(RIGHT * 0.5 + UP * 0.9).scale_by(1.4), run_time=1.6)

        star = Star(5, 0.8, color=RED, fill_opacity=0.9, stroke_width=0.0).move_to(RIGHT * 4.6 + DOWN * 1.6)
        crossed = Cross(star, buff=0.15, color=INK, stroke_width=6.0)
        line = Typst("Underline", TEXT).move_to(LEFT * 3.5 + DOWN * 2.2)
        under = Underline(line, buff=0.12, color=SAND, stroke_width=4.0)
        self.play(FadeIn(star), Write(line), run_time=1.0)
        self.play(Create(crossed), Create(under), run_time=1.0)
        self.play(star.animate.shift(LEFT * 1.2).rotate(0.6), line.animate.shift(RIGHT * 0.8), run_time=1.2)
        self.wait(0.5)
        everything = (formula, behind, around, braced, star, crossed, line, under, title)
        self.play(*(FadeOut(node) for node in everything), run_time=0.6)
