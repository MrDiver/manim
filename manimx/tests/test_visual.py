"""Graphical reference tests: render a scene, compare it with a reference PNG.

The references were reviewed by eye (``manim_logo.png`` was seeded from
Manim's current renderer as an independent cross-check). Regenerate them with

    uv run pytest -m visual --update-references

and review the changed PNGs before committing. On failure, ``*.actual.png``
and ``*.diff.png`` are written next to the reference.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from manimx import CapStyle, Fill, Group, JointType, Shape2D, Stroke, Transform
from manimx.examples.manim_logo import config as logo_config
from manimx.examples.manim_logo import manim_logo
from manimx.objects.geometry import circle, regular_polygon, square
from manimx.renderer import RenderConfig, WGPURenderer
from manimx.testing import ImageTolerance, assert_image_matches
from manimx.utils.color import BLUE, GREEN, ORANGE, PURPLE, RED, WHITE, YELLOW

REFERENCES = Path(__file__).parent / "references"

pytestmark = pytest.mark.visual


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def config(**options: object) -> RenderConfig:
    return RenderConfig(pixel_width=480, pixel_height=270, **options)  # type: ignore[arg-type]


def check(image: np.ndarray, name: str, update: bool, tolerance: ImageTolerance = ImageTolerance()) -> None:
    assert_image_matches(image, REFERENCES / f"{name}.png", tolerance=tolerance, update=update)


def test_manim_logo(renderer: WGPURenderer, update_references: bool) -> None:
    image = renderer.render(manim_logo(), logo_config(480, 270))

    # Looser outlier bound while the reference still comes from Manim, whose
    # edge antialiasing differs slightly from 4x MSAA.
    check(image, "manim_logo", update_references, ImageTolerance(outlier_fraction=0.01))


def test_joints_and_caps(renderer: WGPURenderer, update_references: bool) -> None:
    zigzag = [(-1.0, -0.5), (-0.5, 0.5), (0.0, -0.5), (0.5, 0.5), (1.0, -0.5)]
    styles = [
        (JointType.MITER, CapStyle.BUTT),
        (JointType.ROUND, CapStyle.ROUND),
        (JointType.BEVEL, CapStyle.SQUARE),
        (JointType.AUTO, CapStyle.AUTO),
    ]
    shapes = [
        Shape2D.from_corners(
            f"zigzag-{joint.name}",
            zigzag,
            stroke=Stroke(WHITE, width=30, joint=joint, cap=cap),
            transform=Transform(position=(-4.5 + 3.0 * i, 0.0, 0.0)),
        )
        for i, (joint, cap) in enumerate(styles)
    ]
    image = renderer.render(Group("root", children=shapes), config())

    check(image, "joints_and_caps", update_references)


def test_per_point_stroke_styles(renderer: WGPURenderer, update_references: bool) -> None:
    rainbow = circle(
        "rainbow",
        radius=1.5,
        stroke=Stroke([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED], width=25),
        transform=Transform(position=(-3.0, 0.0, 0.0)),
    )
    taper = Shape2D.from_corners(
        "taper",
        [(-1.5, -1.0), (0.0, 1.0), (1.5, -1.0)],
        stroke=Stroke([BLUE, GREEN], width=[2, 60], joint=JointType.ROUND, cap=CapStyle.ROUND),
        transform=Transform(position=(2.5, 0.0, 0.0)),
    )
    image = renderer.render(Group("root", children=[rainbow, taper]), config())

    check(image, "per_point_stroke_styles", update_references)


def test_transparency_and_gradients(renderer: WGPURenderer, update_references: bool) -> None:
    shapes = [
        regular_polygon("pentagon", 5, radius=1.5, fill=Fill([RED, BLUE], opacity=0.7), stroke=Stroke(YELLOW, width=12, opacity=0.5), transform=Transform(position=(-3.0, 0.0, 0.0))),
        square("back", fill=Fill(RED, opacity=0.5), stroke=Stroke(WHITE, width=8), transform=Transform(position=(1.5, 0.5, 0.0))),
        square("front", fill=Fill(BLUE, opacity=0.5), stroke=Stroke(WHITE, width=8), transform=Transform(position=(2.5, -0.5, 0.0))),
    ]
    image = renderer.render(Group("root", children=shapes), config(background_color="#222222"))

    check(image, "transparency_and_gradients", update_references)


def test_shapes_in_3d(renderer: WGPURenderer, update_references: bool) -> None:
    flipped = Group(
        "flipped",
        children=[
            circle("disc", fill=Fill(GREEN, opacity=1.0), stroke=Stroke(WHITE, width=8)),
            square("inner", side_length=0.8, fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0)),
        ],
        transform=Transform(position=(-3.0, 0.0, 0.0)).rotated(math.pi, (0, 1, 0)),
    )
    tilted = [
        square(
            f"tilt-{i}",
            fill=Fill(PURPLE, opacity=1.0),
            stroke=Stroke(YELLOW, width=10),
            transform=Transform(position=(2.5, 2.0 - 2.0 * i, 0.0)).rotated(angle, (1, 0, 0)).rotated(0.4, (0, 1, 0)),
        )
        for i, angle in enumerate([0.8, 1.3, 1.5])
    ]
    image = renderer.render(Group("root", children=[flipped, *tilted]), config())

    check(image, "shapes_in_3d", update_references)
