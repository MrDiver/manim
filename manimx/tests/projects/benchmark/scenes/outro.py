"""Outro: an animation scene.

Render it alone with ``manimx render Outro``, or the whole timeline with
``manimx render``.
"""

from __future__ import annotations

from manimx.animation import Scene
from manimx.userland import DOWN, PI, UP, Circle, FadeIn, FadeOut, Square


class Outro(Scene):
    def construct(self) -> None:
        square = Square(2.0, color="#5ea3d0", fill_opacity=0.8)
        circle = Circle(1.0, color="#e07a5f", fill_opacity=0.8).shift(DOWN * 2)

        self.play(FadeIn(square, shift=UP * 0.5))
        self.play(square.animate.rotate(PI / 4).shift(UP), FadeIn(circle), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(square), FadeOut(circle))
