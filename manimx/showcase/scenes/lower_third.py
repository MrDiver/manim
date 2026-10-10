"""LowerThird: a caption on its own clip, drawn over another clip.

Clips on a higher layer are drawn over lower ones; this one plays at the
same time as the Solids clip (see the timeline in manimx.json), so the
3D scene keeps its own camera and the caption stays flat on top.
"""

from __future__ import annotations

from manimx.animation import Group, Scene
from manimx.userland import DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, Rectangle, Typst

from .style import INK, MUTED, RED


class LowerThird(Scene):
    def construct(self) -> None:
        bar = Rectangle(0.12, 1.0, color=RED, fill_opacity=1.0, stroke_width=0.0)
        name = Typst("Solids", f'#set text(size: 28pt, weight: "bold", fill: rgb("{INK}"))').shift(RIGHT * 0.9 + UP * 0.2)
        line = Typst("materials, lights, shadows, bloom, particles", f'#set text(size: 15pt, fill: rgb("{MUTED}"))')
        line.shift(RIGHT * 2.1 + DOWN * 0.25)
        third = Group(bar, name, line).move_to(LEFT * 4.2 + DOWN * 3.1)
        self.wait(0.5)
        self.play(FadeIn(third, shift=RIGHT * 0.4), run_time=0.6)
        self.wait(3.0)
        self.play(FadeOut(third, shift=LEFT * 0.4), run_time=0.6)
