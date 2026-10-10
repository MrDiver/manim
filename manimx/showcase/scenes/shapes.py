"""Shapes: the 2D shapes and the animations that bring them in and out.

Circle, Square, Triangle, RegularPolygon, Rectangle, Polyline and Dot;
gradients along a path and tapering strokes; Create, DrawBorderThenFill,
GrowFromCenter, FadeIn with a shift, Indicate, Rotate, ShrinkToCenter,
Uncreate and FadeOut.
"""

from __future__ import annotations

from manimx.animation import Scene
from manimx.userland import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Circle,
    Create,
    Dot,
    DrawBorderThenFill,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    Polyline,
    Rectangle,
    RegularPolygon,
    Rotate,
    ShrinkToCenter,
    Square,
    Triangle,
    Uncreate,
)

from .style import BLUE, GREEN, INK, PALETTE, RED, SAND, VIOLET, caption


class Shapes(Scene):
    def construct(self) -> None:
        title = caption("Shapes", "2D shapes, gradients, tapering strokes and their entrances")
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.8)

        circle = Circle(0.9, color=RED, fill_opacity=0.9).move_to(LEFT * 5 + UP * 0.8)
        square = Square(1.6, color=SAND, fill_opacity=0.9).move_to(LEFT * 2.5 + UP * 0.8)
        triangle = Triangle(1.0, color=GREEN, fill_opacity=0.9).move_to(UP * 0.8)
        hexagon = RegularPolygon(6, 0.95, color=BLUE, fill_opacity=0.9).move_to(RIGHT * 2.5 + UP * 0.8)
        dots = [Dot(0.12, color=color).move_to(RIGHT * (4.4 + 0.4 * i) + UP * 0.8) for i, color in enumerate(PALETTE)]

        # A gradient along the path, and a stroke that tapers.
        banner = Rectangle(6.0, 0.8, fill_color=list(PALETTE), fill_opacity=1.0, stroke_width=0.0).move_to(DOWN * 1.4 + LEFT * 2.5)
        zigzag = Polyline([(-2.0, -0.5), (-1.0, 0.5), (0.0, -0.5), (1.0, 0.5), (2.0, -0.5)], color=INK, stroke_width=[1.0, 12.0, 1.0])
        zigzag.move_to(DOWN * 1.4 + RIGHT * 3.0)
        frame = Rectangle(13.0, 5.2, color=VIOLET, stroke_width=2.0).shift(DOWN * 0.3)

        self.play(Create(circle), DrawBorderThenFill(square), GrowFromCenter(triangle), run_time=1.5)
        self.play(FadeIn(hexagon, shift=UP * 0.5), *(FadeIn(dot, shift=DOWN * 0.3) for dot in dots), run_time=1.0)
        self.play(GrowFromCenter(banner), Create(zigzag), Create(frame), run_time=1.5)
        self.play(Indicate(square), Rotate(hexagon, PI / 3), Rotate(triangle, -PI, run_time=1.5), run_time=1.0)
        self.wait(0.5)
        self.play(ShrinkToCenter(circle), Uncreate(zigzag), *(FadeOut(node) for node in (square, triangle, hexagon, banner, frame, *dots)), run_time=1.2)
        self.play(FadeOut(title), run_time=0.5)
