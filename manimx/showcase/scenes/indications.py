"""Indications: animations that grow, point at, wiggle and wave.

GrowFromPoint, GrowFromEdge, GrowArrow and SpinInFromNothing bring
shapes in; Succession and LaggedStart order animations; Flash,
Circumscribe, Wiggle and ApplyWave point at things; Homotopy bends text
through a function; FadeTransform cross-fades one shape into another;
AddTextLetterByLetter and RemoveTextLetterByLetter show text glyph by
glyph.
"""

from __future__ import annotations

import math

from manimx.animation import LaggedStart, Scene, Succession
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AddTextLetterByLetter,
    ApplyWave,
    Arrow,
    Circle,
    Circumscribe,
    FadeIn,
    FadeOut,
    FadeTransform,
    Flash,
    GrowArrow,
    GrowFromEdge,
    GrowFromPoint,
    Homotopy,
    RemoveTextLetterByLetter,
    SpinInFromNothing,
    Square,
    Star,
    Typst,
    Wiggle,
)

from .style import BLUE, GREEN, INK, RED, SAND, VIOLET, caption

TEXT = f'#set text(size: 40pt, fill: rgb("{INK}"))'


class Indications(Scene):
    def construct(self) -> None:
        title = caption("Indications", "Grow, Succession and LaggedStart, Flash, Circumscribe, Wiggle, ApplyWave, Homotopy")
        self.play(FadeIn(title), run_time=0.6)

        square = Square(1.4, color=RED, fill_opacity=0.9).move_to(LEFT * 4.5 + UP * 0.6)
        circle = Circle(0.75, color=SAND, fill_opacity=0.9).move_to(LEFT * 1.5 + UP * 0.6)
        star = Star(5, 0.85, color=GREEN, fill_opacity=0.9).move_to(RIGHT * 1.5 + UP * 0.6)
        arrow = Arrow(RIGHT * 3.4 + UP * 0.6, RIGHT * 5.6 + UP * 0.6, buff=0.0, color=BLUE, stroke_width=6.0)
        self.play(
            LaggedStart(
                GrowFromPoint(square, LEFT * 6.5 + DOWN * 2.0),
                GrowFromEdge(circle, DOWN),
                SpinInFromNothing(star, angle=math.pi),
                GrowArrow(arrow),
                lag_ratio=0.4,
            ),
            run_time=2.4,
        )
        self.play(Succession(square.animate.shift(UP * 0.4), square.animate.shift(DOWN * 0.4)), run_time=0.8)

        self.play(Flash(star.get_center(), line_length=0.35, num_lines=14, flash_radius=1.0, style={"color": SAND, "stroke_width": 5.0}), run_time=0.8)
        self.play(Circumscribe(circle, shape="circle", buff=0.15, style={"color": VIOLET, "stroke_width": 5.0}), run_time=1.2)
        self.play(Wiggle(square), run_time=1.4)

        words = Typst("waves run through text", TEXT).move_to(DOWN * 1.9)
        self.play(AddTextLetterByLetter(words, time_per_char=0.05))
        self.play(ApplyWave(words, amplitude=0.25), run_time=1.6)
        self.play(Homotopy(lambda x, y, z, t: (x, y + 0.25 * math.sin(3.0 * x - 4.0 * t) * math.sin(math.pi * t), z), words, run_time=2.0))

        diamond = Square(1.4, color=VIOLET, fill_opacity=0.9).rotate(math.pi / 4).move_to(LEFT * 1.5 + UP * 0.6)
        self.play(FadeTransform(circle, diamond), run_time=1.0)
        self.play(RemoveTextLetterByLetter(words, time_per_char=0.03))
        self.play(*(FadeOut(node) for node in (square, diamond, star, arrow, title)), run_time=0.6)
