"""Tables and charts: Table, Matrix, DecimalMatrix, BarChart, Variable
and BulletedList.

Entries are laid out on a grid measured once; region() gives the box of
some cells, for a highlight. A DecimalMatrix shows numbers that follow
signals, here derived from one angle; a BarChart's bars and labels follow
its values; a Variable shows a signal next to its name.
"""

from __future__ import annotations

import math

from manimx.animation import Scene, derive
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    BarChart,
    BulletedList,
    DecimalMatrix,
    FadeIn,
    FadeOut,
    Matrix,
    RoundedRectangle,
    Table,
    Value,
    Variable,
    Write,
)

from .style import BLUE, GREEN, INK, MUTED, PALETTE, RED, SAND, caption

TEXT = f'#set text(size: 20pt, fill: rgb("{INK}"))'
MATH = f'#set text(size: 24pt, fill: rgb("{INK}"))'
LINES = {"color": MUTED, "stroke_width": 1.5}


class Tables(Scene):
    def construct(self) -> None:
        title = caption("Tables and charts", "Table, Matrix, DecimalMatrix, Variable, BarChart and BulletedList")
        self.play(FadeIn(title), run_time=0.6)

        planets = Table(
            [["0", "0.39", "88"], ["1", "1.00", "365"], ["2", "1.52", "687"], ["95", "9.54", "10 759"]],
            row_labels=["Mercury", "Earth", "Mars", "Saturn"],
            column_labels=["*moons*", "*AU*", "*days*"],
            top_left="",
            preamble=TEXT,
            line_style=LINES,
        ).shift(LEFT * 3.4 + UP * 0.2)
        self.play(FadeIn(planets), run_time=0.8)
        center, width, height = planets.region(2, 0, 2, 3)
        highlight = RoundedRectangle(width, height, corner_radius=0.06, stroke_color=SAND, stroke_width=3.0).move_to(center)
        self.play(FadeIn(highlight), run_time=0.5)
        center, width, height = planets.region(4, 0, 4, 3)
        self.play(highlight.animate.move_to(center).set_size(width, height), run_time=0.8)
        center, width, height = planets.region(0, 1, 4, 1)
        self.play(highlight.animate.move_to(center).set_size(width, height), run_time=0.8)

        angle = Value(0.0)
        cosine = derive(math.cos, angle.value)
        sine = derive(math.sin, angle.value)
        minus_sine = derive(lambda a: -math.sin(a), angle.value)
        rotation = DecimalMatrix([[cosine, minus_sine], [sine, cosine]], decimals=2, preamble=MATH, bracket="(", color=INK, stroke_width=2.5).shift(RIGHT * 3.6 + UP * 0.8)
        symbolic = Matrix([["cos theta", "-sin theta"], ["sin theta", "cos theta"]], MATH, bracket="(", color=INK, stroke_width=2.5).shift(RIGHT * 3.6 + UP * 0.8)
        theta = Variable(angle.value, "$theta$", decimals=2, preamble=MATH, color=GREEN).shift(RIGHT * 3.6 + DOWN * 1.0)
        self.play(Write(symbolic), run_time=1.2)
        self.wait(0.4)
        self.play(FadeOut(symbolic), FadeIn(rotation), FadeIn(theta), run_time=0.6)
        self.play(angle.animate.set_value(math.pi / 3), run_time=2.0)
        self.play(angle.animate.set_value(math.pi), run_time=1.6)
        self.wait(0.4)
        self.play(*(FadeOut(node) for node in (planets, highlight, rotation, theta)), run_time=0.6)

        chart = BarChart(
            [3, 5, 2, 6, 4], ["mon", "tue", "wed", "thu", "fri"], y_range=(0, 8, 2), x_length=6.0, y_length=4.0,
            bar_colors=list(PALETTE), include_values=True, preamble=TEXT, axis_style={"color": MUTED, "stroke_width": 2.0},
        ).shift(LEFT * 3.0 + DOWN * 0.3)  # fmt: skip
        self.play(FadeIn(chart), run_time=0.8)
        self.play(chart.animate.set_values([6, 2, 7, 3, 5]), run_time=1.5)
        self.play(chart.animate.set_values([1, 4, 4, 7, 2]), run_time=1.5)

        notes = BulletedList(
            "bars follow their values",
            f'labels ride #text(fill: rgb("{RED}"))[on top]',
            f'zoom with #text(fill: rgb("{BLUE}"))[`axes.set_ranges`]',
            preamble=TEXT,
        ).shift(RIGHT * 3.6 + DOWN * 0.3)
        for item in notes.items:
            item.set_alpha(0.0)
        self.add(notes)
        for item in notes.items:
            self.play(item.animate.set_alpha(1.0).shift(RIGHT * 0.3), run_time=0.5)
        self.play(chart.axes.animate.set_ranges(y=(0, 16, 4)), run_time=1.2)
        self.wait(0.6)
        self.play(*(FadeOut(node) for node in (chart, notes, title)), run_time=0.6)
