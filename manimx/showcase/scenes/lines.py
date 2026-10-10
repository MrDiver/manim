"""Lines: lines, arrows, angle marks and the shapes made of arcs.

Line ends are signals: an arrow connects two dots and follows them, an
Angle follows its two lines and a DecimalNumber shows it in degrees
(derived from Angle.radians). Arc, Annulus, Star, RoundedRectangle and
Sector animate their parameters.
"""

from __future__ import annotations

import math

from manimx.animation import Group, Scene, derive
from manimx.userland import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Angle,
    Annulus,
    Arc,
    Arrow,
    Create,
    CurvedArrow,
    DashedLine,
    DecimalNumber,
    Dot,
    DoubleArrow,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line,
    RoundedRectangle,
    Sector,
    Star,
)

from .style import BLUE, GREEN, INK, MUTED, RED, SAND, VIOLET, caption

NUMBER = f'#set text(size: 22pt, fill: rgb("{INK}"))'


class Lines(Scene):
    def construct(self) -> None:
        title = caption("Lines", "Lines, arrows and angles that follow their signals; shapes of arcs")
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.8)

        # An arrow between two dots, following them.
        left, right = Dot(0.1, color=RED).move_to(LEFT * 5.5 + UP * 1.2), Dot(0.1, color=RED).move_to(LEFT * 2.5 + UP * 1.2)
        arrow = Arrow(color=SAND, buff=0.2).connect(start=left, end=right)
        dashed = DashedLine(LEFT * 5.5 + DOWN * 0.2, LEFT * 2.5 + DOWN * 0.2, dash_length=0.15, color=MUTED, stroke_width=3.0)
        double = DoubleArrow(LEFT * 5.5 + DOWN * 1.0, LEFT * 2.5 + DOWN * 1.0, color=BLUE, buff=0.0)
        curved = CurvedArrow(LEFT * 5.5 + DOWN * 2.0, LEFT * 2.5 + DOWN * 2.0, angle=PI / 3, color=GREEN)
        self.play(FadeIn(left), FadeIn(right), Create(arrow), Create(dashed), Create(double), Create(curved), run_time=1.2)
        self.play(left.animate.shift(DOWN * 0.6), right.animate.shift(UP * 0.5 + RIGHT * 0.4), run_time=1.2)

        # An angle between two lines, in degrees, as one turns.
        corner = DOWN * 1.0 + LEFT * 0.4
        base = Line(corner, corner + RIGHT * 2.6, color=INK, stroke_width=3.0)
        arm = Line(corner, corner + RIGHT * 1.8 + UP * 1.8, color=INK, stroke_width=3.0)
        angle = Angle(base, arm, radius=0.6, color=SAND, stroke_width=3.0)
        degrees = derive(math.degrees, angle.radians, name="degrees")
        label = DecimalNumber(degrees, decimals=0, preamble=NUMBER, anchor=LEFT, characters=3).move_to(corner + RIGHT * 0.45 + DOWN * 0.35)
        self.play(Create(base), Create(arm), Create(angle), FadeIn(label), run_time=1.0)
        self.play(arm.animate.set_end(corner + LEFT * 1.0 + UP * 2.4), run_time=2.0)
        self.play(arm.animate.set_end(corner + RIGHT * 2.4 + UP * 0.6), run_time=1.5)

        # Shapes made of arcs, animating their parameters.
        arc = Arc(0.6, 0.0, PI / 3, color=RED, stroke_width=6.0).move_to(RIGHT * 3.4 + UP * 1.3)
        ring = Annulus(0.3, 0.6, color=BLUE, fill_opacity=0.8, stroke_width=0.0).move_to(RIGHT * 5.2 + UP * 1.3)
        star = Star(5, 0.65, color=SAND, fill_opacity=0.9, stroke_width=0.0).move_to(RIGHT * 3.4 + DOWN * 0.6)
        rounded = RoundedRectangle(1.3, 1.0, 0.1, color=VIOLET, fill_opacity=0.8, stroke_width=0.0).move_to(RIGHT * 5.2 + DOWN * 0.6)
        wedge = Sector(0.7, PI / 2, PI / 2, color=GREEN, fill_opacity=0.8, stroke_width=0.0).move_to(RIGHT * 4.3 + DOWN * 2.3)
        shapes = Group(arc, ring, star, rounded, wedge)
        self.play(*(GrowFromCenter(shape) for shape in shapes.children), run_time=1.0)
        self.play(
            arc.animate.set_angles(angle=2 * PI * 0.9),
            ring.animate.set_radii(inner_radius=0.5),
            star.animate.set_radii(inner_radius=0.2),
            rounded.animate.set_corner_radius(0.5),
            wedge.animate.set_angles(angle=3 * PI / 2),
            run_time=2.0,
        )
        self.wait(0.5)
        everything = (left, right, arrow, dashed, double, curved, base, arm, angle, label, *shapes.children, title)
        self.play(*(FadeOut(node) for node in everything), run_time=0.6)
