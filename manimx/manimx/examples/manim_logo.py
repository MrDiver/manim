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

from ..objects import Fill, Group, Shape2D, Stroke, Transform
from ..objects.geometry import circle, square, triangle
from ..renderer.config import RenderConfig
from ..renderer.scene_graph import bounding_box

BACKGROUND = "#ece6e2"
LOGO_GREEN = "#87c2a5"
LOGO_BLUE = "#525893"
LOGO_RED = "#e07a5f"
LOGO_BLACK = "#343434"

M_GLYPH_POINTS = Path(__file__).with_name("manim_logo_m.npy")


def _colored(color: str) -> dict:
    """Manim's ``color=..., fill_opacity=1``: same color for fill and stroke."""
    return {"fill": Fill(color, opacity=1.0), "stroke": Stroke(color)}


def manim_logo() -> Group:
    ds_m = Shape2D(
        "ds_m",
        points=np.load(M_GLYPH_POINTS),
        fill=Fill(LOGO_BLACK, opacity=1.0),
        stroke=Stroke(LOGO_BLACK, width=0.0),  # MathTex has no stroke
        transform=Transform(position=(-2.25, 1.5, 0.0), scale=(7.0, 7.0, 1.0)),
    )
    shapes = [
        triangle("triangle", transform=Transform(position=(1.0, 0.0, 0.0)), **_colored(LOGO_RED)),
        square("square", transform=Transform(position=(0.0, 1.0, 0.0)), **_colored(LOGO_BLUE)),
        circle("circle", transform=Transform(position=(-1.0, 0.0, 0.0)), **_colored(LOGO_GREEN)),
        ds_m,
    ]  # order matters: later shapes are drawn on top

    logo = Group("logo", children=shapes)
    # move_to(ORIGIN): shift the group so its bounding box is centered.
    low, high = bounding_box(logo)
    logo.transform = logo.transform.translated(-(low + high) / 2)
    return logo


def config(pixel_width: int = 1920, pixel_height: int = 1080) -> RenderConfig:
    return RenderConfig(
        pixel_width=pixel_width, pixel_height=pixel_height, background_color=BACKGROUND
    )
