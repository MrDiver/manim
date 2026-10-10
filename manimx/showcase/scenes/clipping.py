"""Clipping: content seen only through a mask.

A ClippingGroup draws its content where its mask is. The mask is an
ordinary node: here a circle that moves and grows like a spotlight over
a striped pattern, and a text mask that reveals a gradient.
"""

from __future__ import annotations

from manimx.animation import ClippingGroup, Group, Scene
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    Circle,
    FadeIn,
    FadeOut,
    Rectangle,
    Typst,
)

from .style import INK, NAVY, PALETTE, caption


class Clipping(Scene):
    def construct(self) -> None:
        title = caption("Clipping", "ClippingGroup: a mask node and the content it shows")
        self.play(FadeIn(title), run_time=0.6)

        stripes = Group(
            *(Rectangle(0.5, 4.0, color=PALETTE[i % len(PALETTE)], fill_opacity=1.0, stroke_width=0.0).shift(RIGHT * (0.5 * i - 3.0)) for i in range(13))
        ).shift(LEFT * 3.0 + DOWN * 0.6)
        lens = Circle(0.9, fill_opacity=1.0).move_to(LEFT * 5.0 + DOWN * 0.6)
        spotlight = ClippingGroup(lens, stripes, "spotlight")
        outline = Rectangle(6.5, 4.0, color=NAVY, stroke_width=2.0).move_to(LEFT * 3.0 + DOWN * 0.6)
        self.play(FadeIn(outline), FadeIn(spotlight), run_time=0.6)
        self.play(lens.animate.shift(RIGHT * 4.0), run_time=1.5)
        self.play(lens.animate.scale_by(1.8).shift(LEFT * 2.0), run_time=1.2)

        word = Typst("MASK", f'#set text(size: 110pt, weight: "black", fill: rgb("{INK}"))')
        gradient = Rectangle(5.0, 2.0, fill_color=list(PALETTE), fill_opacity=1.0, stroke_width=0.0)
        reveal = ClippingGroup(word, gradient, "reveal").move_to(RIGHT * 3.6 + DOWN * 0.6)
        self.play(FadeIn(reveal), run_time=0.8)
        self.play(gradient.animate.shift(RIGHT * 1.5), run_time=1.5)
        self.play(gradient.animate.shift(LEFT * 3.0), run_time=1.5)
        self.play(*(FadeOut(node) for node in (spotlight, outline, reveal, title)), run_time=0.6)
