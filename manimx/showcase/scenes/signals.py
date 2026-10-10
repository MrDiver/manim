"""Signals: how values change and depend on each other.

Every value is a signal. node.animate chains describe one end state;
moves along different axes in one play run independently; derive() and
follow() replace updaters (a value computed from others, at any time).
"""

from __future__ import annotations

import numpy as np
from manimx.animation import Scene, derive
from manimx.animation.kinds import COLORS
from manimx.userland import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    FadeOut,
    Square,
    Typst,
)

from .style import INK, MUTED, RED, SAND, caption


class Signals(Scene):
    def construct(self) -> None:
        title = caption("Signals", "animate chains, independent axes, derive() and follow()")
        self.play(FadeIn(title), run_time=0.6)

        # A chain describes the end state, step by step.
        square = Square(1.2, color=SAND, fill_opacity=0.9).move_to(LEFT * 4.5)
        self.play(FadeIn(square), run_time=0.5)
        self.play(square.animate.move_to(LEFT * 2).rotate(PI / 4).scale_by(0.6).shift(UP), run_time=1.5)

        # Two moves on different axes at once: each axis animates on its own.
        ball = Dot(0.25, color=RED).move_to(RIGHT * 1 + DOWN * 1.5)
        ring = Circle(0.45, color=INK, stroke_width=3).follow(ball)  # always around the ball
        label = Typst("follow()", f'#set text(size: 14pt, fill: rgb("{MUTED}"))').follow(ball, offset=(0.0, 0.75, 0.0))
        self.play(FadeIn(ball), FadeIn(ring), FadeIn(label), run_time=0.5)
        self.play(ball.animate.shift(RIGHT * 4), run_time=2.0)
        self.play(ball.animate.shift(UP * 3), ball.animate.shift(LEFT * 2), run_time=1.5)

        # derive(): the marker's color comes from the ball's height.
        marker = Circle(0.35, fill_opacity=1.0, stroke_width=0.0).move_to(RIGHT * 5.5 + DOWN * 2.5)
        low, high = np.array([0.37, 0.64, 0.82]), np.array([0.88, 0.48, 0.37])
        marker.fill_color = derive(lambda y: np.clip((y + 1.5) / 3.0, 0.0, 1.0) * (high - low)[None] + low[None], ball.position.y, kind=COLORS)
        self.play(FadeIn(marker), run_time=0.5)
        self.play(ball.animate.shift(DOWN * 3), run_time=1.5)
        self.play(ball.animate.shift(UP * 2.5), run_time=1.0)
        self.wait(0.3)
        self.play(*(FadeOut(node) for node in (square, ball, ring, label, marker, title)), run_time=0.6)
