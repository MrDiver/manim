"""Intro: the Manim logo assembles itself on a light card.

Shows the basics of an animation scene: shapes, lagged fades with a shift,
several moves in one ``play`` (each axis animates independently), groups,
``Indicate``, and fading everything out again.
"""

from __future__ import annotations

import numpy as np

from manimx.animation import DOWN, LEFT, PI, RIGHT, UP, Circle, FadeIn, FadeOut, Group, Indicate, Rectangle, Scene, Square, Shape2D, Triangle
from manimx.examples.manim_logo import BACKGROUND, LOGO_BLACK, LOGO_BLUE, LOGO_GREEN, LOGO_RED, M_GLYPH_POINTS, manim_logo

CENTER = -np.array(manim_logo().transform.position)  # the logo's center in Manim's layout


class Intro(Scene):
    def construct(self) -> None:
        card = Rectangle(15, 8.5, color=BACKGROUND, fill_opacity=1, stroke_width=0)
        circle = Circle(color=LOGO_GREEN, fill_opacity=1)
        square = Square(color=LOGO_BLUE, fill_opacity=1)
        triangle = Triangle(color=LOGO_RED, fill_opacity=1)
        m = Shape2D(np.load(M_GLYPH_POINTS), color=LOGO_BLACK, fill_opacity=1, stroke_width=0)
        m.scale_by(7)

        self.play(FadeIn(card), run_time=0.6)

        # Start spread out in a row, then gather into the logo.
        for shape, x in ((triangle, 3.0), (square, 0.0), (circle, -3.0)):
            shape.move_to((x, -0.5, 0.0))
        self.play(*(FadeIn(shape, shift=UP) for shape in (triangle, square, circle)), lag_ratio=0.4, run_time=1.2)
        self.play(
            triangle.animate.move_to(RIGHT - CENTER),
            square.animate.move_to(UP - CENTER).rotate(PI / 2),
            circle.animate.move_to(LEFT - CENTER),
            run_time=1.5,
        )

        m.move_to(np.array([-2.25, 1.5, 0.0]) - CENTER)
        self.play(FadeIn(m, shift=DOWN * 0.5))

        logo = Group(triangle, square, circle, m)
        self.add(logo)
        self.play(Indicate(logo, scale_factor=1.1))
        self.wait(0.5)
        self.play(FadeOut(logo), FadeOut(card), run_time=0.8)
