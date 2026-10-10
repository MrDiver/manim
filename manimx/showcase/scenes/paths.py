"""Paths: points along a shape's border.

point_at(proportion) is a point that follows the shape: its path, its
position, scale and rotation. By length (the default), equal steps are
equally far apart along the border; by curves, each Bezier curve gets
the same share, so points crowd where the curves are short. Both shapes
here have a dense wave on top and three long sides. MoveAlongPath moves
a node through a path the same two ways.
"""

from __future__ import annotations

import numpy as np
from manimx.animation import Scene, Shape2D
from manimx.userland import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    MoveAlongPath,
    Typst,
)
from manimx.userland.geometry import corner_points

from .style import BLUE, MUTED, RED, SAND, caption

DOTS = 32
LABEL = f'#set text(size: 18pt, fill: rgb("{MUTED}"))'


def _outline() -> np.ndarray:
    """Three long sides and a wave of 48 short ones on top."""
    xs = np.linspace(2.4, -2.4, 49)
    wave = [(x, 1.2 + 0.18 * np.sin(x * 9.0)) for x in xs]
    return corner_points([(-2.4, -1.2), (2.4, -1.2), *wave], closed=True)


class Paths(Scene):
    def construct(self) -> None:
        title = caption("Paths", "Points along a border: by length or by Bezier curves")
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.8)

        sides = []
        for side, by_length, name in ((LEFT, True, "by length"), (RIGHT, False, "by curves")):
            shape = Shape2D(_outline(), color=BLUE, stroke_width=3.0).move_to(side * 3.3 + DOWN * 0.3)
            dots = [Dot(0.07, color=SAND).follow(shape.point_at(i / DOTS, by_length=by_length)) for i in range(DOTS)]
            label = Typst(name, LABEL).move_to(side * 3.3 + DOWN * 2.3)
            sides.append((shape, dots, label, by_length))

        self.play(*(Create(shape) for shape, *_ in sides), *(FadeIn(label) for *_, label, _ in sides), run_time=1.2)
        self.play(*(FadeIn(dot) for _, dots, *_ in sides for dot in dots), run_time=0.6)
        self.wait(0.5)
        # The dots are points of the shapes: they keep to them as the shapes turn and grow.
        self.play(*(shape.animate.rotate(PI / 8).scale_by(1.15) for shape, *_ in sides), run_time=1.5)
        self.play(*(shape.animate.rotate(-PI / 8).scale_by(1 / 1.15) for shape, *_ in sides), run_time=1.0)

        # Through the whole path in the same time: steady by length, slow on the wave by curves.
        runners = [Dot(0.14, color=RED).move_to(shape.points.value[0]) for shape, *_ in sides]
        self.play(*(FadeIn(runner) for runner in runners), run_time=0.4)
        self.play(*(MoveAlongPath(runner, shape, by_length=by_length, rate=lambda t: t) for runner, (shape, _, _, by_length) in zip(runners, sides, strict=True)), run_time=5.0)
        self.wait(0.4)
        everything = [node for shape, dots, label, _ in sides for node in (shape, *dots, label)]
        self.play(*(FadeOut(node) for node in (*everything, *runners, title)), run_time=0.6)


__all__ = ["Paths"]
