"""Measures: numbers and braces that follow a shape's parameters.

A DecimalNumber shows the Circle's radius while it grows, a Morph turns
the circle into a Rectangle, and Braces on two of its sides, each with
a number, follow its width and height as they change one after the
other. Nothing is updated by hand: the numbers, braces and the label's
position are derived from the shapes' parameter signals.
"""

from __future__ import annotations

from manimx.animation import Group, Scene, derive
from manimx.animation.nodes.node import Offset
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    Brace,
    Circle,
    Create,
    DecimalNumber,
    FadeIn,
    FadeOut,
    Morph,
    Rectangle,
    Typst,
)

from .style import BLUE, INK, SAND, caption

NUMBER = f'#set text(size: 30pt, fill: rgb("{INK}"))'
BRACE = {"color": SAND, "stroke_width": [1.0, 6.0, 1.0]}


class Measures(Scene):
    def construct(self) -> None:
        title = caption("Measures", "Numbers and braces that follow a shape's parameters")
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.8)

        circle = Circle(0.5, color=BLUE, fill_opacity=0.35)
        # "r = 0.50" with its left end 0.3 right of the circle, wherever the radius is.
        name = Typst("$r =$", NUMBER)
        name_width = _width(name)
        radius = DecimalNumber(circle.radius, preamble=NUMBER, anchor=LEFT).shift(RIGHT * (name_width + 0.15))
        label = Group(name.shift(RIGHT * name_width / 2), radius)
        label.position.x = derive(Offset(0.3), circle.radius, name="label.position.x")

        self.play(Create(circle), FadeIn(label), run_time=1.2)
        self.play(circle.animate.set_radius(1.0), run_time=2.0)
        self.wait(0.4)

        square = Rectangle(2.0, 2.0, color=BLUE, fill_opacity=0.35)
        self.play(Morph(circle, square), FadeOut(label, run_time=0.6), run_time=1.5)

        # A brace under the bottom side and one right of the right side, each with its length.
        bottom, right = Brace(square.edge(DOWN), **BRACE), Brace(square.edge(RIGHT), **BRACE)
        width = bottom.attach(DecimalNumber(square.width, preamble=NUMBER))
        height = right.attach(DecimalNumber(square.height, preamble=NUMBER))
        self.play(Create(bottom), Create(right), FadeIn(width), FadeIn(height), run_time=1.0)

        self.play(square.animate.set_size(width=1.0), run_time=1.5)
        self.play(square.animate.set_size(height=1.0), run_time=1.5)
        self.wait(0.6)
        self.play(*(FadeOut(node) for node in (square, bottom, right, width, height, title)), run_time=0.6)


def _width(group: Group) -> float:
    """The width of a Typst group's glyphs."""
    xs = [x for shape in group.children for x in shape.points.value[:, 0]]
    return max(xs) - min(xs)
