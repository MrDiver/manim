"""Morphing: one shape (or group) turning into another.

Morph aligns the paths of both sides once and interpolates points,
colors, opacities and widths; extra shapes on one side grow from or
shrink into the other side's center.
"""

from __future__ import annotations

import numpy as np
from manimx.animation import Group, Scene
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    Circle,
    FadeIn,
    FadeOut,
    Morph,
    Polyline,
    RegularPolygon,
    Square,
    Triangle,
)

from .style import BLUE, GREEN, PALETTE, RED, SAND, VIOLET, caption


def star(points: int = 5, outer: float = 1.3, inner: float = 0.55) -> Polyline:
    angles = np.pi / 2 + np.arange(2 * points) * np.pi / points
    radii = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    corners = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return Polyline(corners, closed=True, color=VIOLET, fill_opacity=0.9)


class Morphing(Scene):
    def construct(self) -> None:
        title = caption("Morphing", "Morph between shapes, and between groups of different sizes")
        self.play(FadeIn(title), run_time=0.6)

        circle = Circle(1.2, color=RED, fill_opacity=0.9).shift(LEFT * 3.5)
        square = Square(2.2, color=SAND, fill_opacity=0.9).shift(LEFT * 3.5)
        shape = star().shift(LEFT * 3.5)
        self.play(FadeIn(circle), run_time=0.5)
        self.play(Morph(circle, square), run_time=1.2)
        self.play(Morph(square, shape), run_time=1.2)

        three = Group(*(Circle(0.45, color=color, fill_opacity=0.9).shift(RIGHT * (2.3 + 1.1 * i)) for i, color in enumerate(PALETTE[:3])))
        five = Group(
            *(RegularPolygon(3 + i, 0.4, color=color, fill_opacity=0.9).shift(RIGHT * (1.8 + 0.85 * i) + DOWN * (0.6 if i % 2 else -0.6)) for i, color in enumerate(PALETTE))
        )
        one = Triangle(1.4, color=GREEN, fill_opacity=0.9).shift(RIGHT * 3.5)
        self.play(FadeIn(three), run_time=0.5)
        self.play(Morph(three, five), run_time=1.5)
        self.play(Morph(five, one), run_time=1.5)
        round_again = Circle(1.2, color=BLUE, fill_opacity=0.9).shift(LEFT * 3.5)
        self.play(Morph(shape, round_again), run_time=1.2)
        self.wait(0.4)
        self.play(FadeOut(round_again), FadeOut(one), FadeOut(title), run_time=0.6)
