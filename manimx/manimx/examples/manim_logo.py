"""Port of Manim's ``ManimCELogo`` example scene.

Original (docs/source/examples.rst)::

    class ManimCELogo(Scene):
        def construct(self):
            self.camera.background_color = "#ece6e2"
            logo_green = "#87c2a5"
            logo_blue = "#525893"
            logo_red = "#e07a5f"
            logo_black = "#343434"
            ds_m = MathTex(r"\\mathbb{M}", fill_color=logo_black).scale(7)
            ds_m.shift(2.25 * LEFT + 1.5 * UP)
            circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
            square = Square(color=logo_blue, fill_opacity=1).shift(UP)
            triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
            logo = VGroup(triangle, square, circle, ds_m)  # order matters
            logo.move_to(ORIGIN)
            self.add(logo)

The ``\\mathbb{M}`` glyph is baked from Manim's MathTex output into
``manim_logo_m.npy`` (centered at the origin, unscaled), because manimx has
no LaTeX pipeline yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..animation import Circle, Group, Shape2D, Square, Triangle
from ..renderer.config import RenderConfig
from ..renderer.scene_graph import bounding_box

BACKGROUND = "#ece6e2"
LOGO_GREEN = "#87c2a5"
LOGO_BLUE = "#525893"
LOGO_RED = "#e07a5f"
LOGO_BLACK = "#343434"

M_GLYPH_POINTS = Path(__file__).with_name("manim_logo_m.npy")


def manim_logo() -> Group:
    """The logo as a node tree: ``"triangle"``, ``"square"``, ``"circle"`` and
    the glyph ``"m"`` in the group ``"logo"``, centered at the origin."""
    triangle = Triangle(name="triangle", color=LOGO_RED, fill_opacity=1.0).move_to((1.0, 0.0, 0.0))
    square = Square(name="square", color=LOGO_BLUE, fill_opacity=1.0).move_to((0.0, 1.0, 0.0))
    circle = Circle(name="circle", color=LOGO_GREEN, fill_opacity=1.0).move_to((-1.0, 0.0, 0.0))
    # The blackboard-bold M is a glyph outline, so a plain Shape2D; MathTex has no stroke.
    m = Shape2D(np.load(M_GLYPH_POINTS), "m", color=LOGO_BLACK, fill_opacity=1.0, stroke_width=0.0)
    m.move_to((-2.25, 1.5, 0.0)).set_scale((7.0, 7.0, 1.0))

    logo = Group(triangle, square, circle, m, name="logo")  # later shapes are drawn on top
    # move_to(ORIGIN): shift the group so its bounding box is centered.
    low, high = bounding_box(logo.snapshot())
    logo.shift(-(low + high) / 2)
    return logo


def config(pixel_width: int = 1920, pixel_height: int = 1080) -> RenderConfig:
    return RenderConfig(
        pixel_width=pixel_width, pixel_height=pixel_height, background_color=BACKGROUND
    )
