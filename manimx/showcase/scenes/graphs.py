"""Graphs: axes whose ranges are signals, and plots that follow them.

The ranges animate like any value: ticks, labels and the graph of x²
rescale at every frame (a zoom). A Value drives a dot along the graph
and a number shows its y.
"""

from __future__ import annotations

from manimx.animation import Scene, derive
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    Axes,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    NumberPlane,
    Value,
)

from .style import INK, MUTED, NAVY, RED, SAND, caption

NUMBERS = f'#set text(size: 16pt, fill: rgb("{MUTED}"))'
READOUT = f'#set text(size: 26pt, fill: rgb("{INK}"))'


class Graphs(Scene):
    def construct(self) -> None:
        title = caption("Graphs", "Ranges are signals: zoom the axes and the plot follows")
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.8)

        plane = NumberPlane((-1, 6, 1), (-5, 30, 5), x_length=10.0, y_length=5.2, grid_style={"color": NAVY, "stroke_width": 1.5}, color=MUTED, stroke_width=2.0)
        plane.move_to(DOWN * 0.5 + LEFT * 0.8)
        axes = Axes((-1, 6, 1), (-5, 30, 5), x_length=10.0, y_length=5.2, include_numbers=True, preamble=NUMBERS, color=INK, stroke_width=2.5)
        axes.move_to(DOWN * 0.5 + LEFT * 0.8)
        graph = axes.plot(lambda x: x * x, color=SAND, stroke_width=5.0)
        self.play(Create(plane.grid), Create(axes), run_time=1.5)
        self.play(Create(graph), run_time=1.2)

        x = Value(0.0)
        rider = Dot(0.12, color=RED).follow(axes.graph_point(x.value, lambda value: value * value))
        # The graph's y at the dot: derived from the same Value.
        readout = DecimalNumber(derive(lambda value: value * value, x.value), decimals=2, preamble=READOUT, anchor=LEFT, characters=6).move_to(RIGHT * 5.0 + DOWN * 0.5)
        self.play(FadeIn(rider), FadeIn(readout), run_time=0.5)
        self.play(x.animate.set_value(4.5), run_time=2.0)

        # Zoom out: the ranges grow, everything on the axes follows.
        self.play(axes.animate.set_ranges(x=(-2, 12, 2), y=(-20, 120, 20)), plane.animate.set_ranges(x=(-2, 12, 2), y=(-20, 120, 20)), run_time=2.0)
        self.play(x.animate.set_value(10.0), run_time=1.5)
        self.play(axes.animate.set_ranges(x=(-1, 6, 1), y=(-5, 30, 5)), plane.animate.set_ranges(x=(-1, 6, 1), y=(-5, 30, 5)), x.animate.set_value(2.0), run_time=2.0)
        self.wait(0.5)
        self.play(*(FadeOut(node) for node in (plane, axes, graph, rider, readout, title)), run_time=0.6)

