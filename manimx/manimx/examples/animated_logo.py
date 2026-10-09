"""The Manim logo assembling itself: the first animated manimx scene.

The shapes of ``ManimCELogo`` fade in one after another from below, slide
into the logo's layout while turning, the blackboard M fades in, and the
logo pulses once. It exercises lagged fades with shifts, overlapping
moves on independent axes, rotation, an absolute move_to, and Indicate.

    prepared = AnimatedLogo().prepare()
    image = WGPURenderer().render(prepared.state_at(2.0), config())
"""

from __future__ import annotations

import numpy as np

from ..animation import DOWN, LEFT, PI, RIGHT, UP, Circle, FadeIn, Group, Indicate, Scene, Shape2D, Square, Triangle
from ..animation.rate_functions import smooth
from ..renderer.config import RenderConfig
from .manim_logo import BACKGROUND, LOGO_BLACK, LOGO_BLUE, LOGO_GREEN, LOGO_RED, M_GLYPH_POINTS, manim_logo

# The static logo is centered by shifting its group (Manim's
# move_to(ORIGIN)); the animated shapes end up in the same place.
_CENTER = -np.array(manim_logo().transform.position)


class AnimatedLogo(Scene):
    def construct(self) -> None:
        circle = Circle(name="circle", color=LOGO_GREEN, fill_opacity=1)
        square = Square(name="square", color=LOGO_BLUE, fill_opacity=1)
        triangle = Triangle(name="triangle", color=LOGO_RED, fill_opacity=1)
        m = Shape2D(np.load(M_GLYPH_POINTS), name="m", color=LOGO_BLACK, fill_opacity=1, stroke_width=0)
        m.scale_by(7)

        # Start spread out in a row, then gather into the logo.
        for shape, x in ((triangle, 3.0), (square, 0.0), (circle, -3.0)):
            shape.move_to((x, -0.5, 0.0))
        self.play(*(FadeIn(shape, shift=UP) for shape in (triangle, square, circle)), lag_ratio=0.4, run_time=1.2)

        targets = {triangle: RIGHT, square: UP, circle: LEFT}
        self.play(
            *(shape.animate.move_to(target - _CENTER).rotate(PI / 2 if shape is square else 0.0) for shape, target in targets.items()),
            run_time=1.5,
            rate=smooth,
        )

        m.move_to(np.array([-2.25, 1.5, 0.0]) - _CENTER)
        self.play(FadeIn(m, shift=DOWN * 0.5), run_time=1.0)

        logo = Group(triangle, square, circle, m, name="logo")
        self.add(logo)
        self.play(Indicate(logo, scale_factor=1.1))
        self.wait(0.5)


def config(pixel_width: int = 1920, pixel_height: int = 1080) -> RenderConfig:
    return RenderConfig(pixel_width=pixel_width, pixel_height=pixel_height, background_color=BACKGROUND)
