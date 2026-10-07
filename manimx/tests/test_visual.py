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

from manimx import CapStyle, ClippingGroup, Fill, Group, JointType, Shape2D, Stroke, Transform
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


def test_clipping(renderer: WGPURenderer, update_references: bool) -> None:
    stripes = Group(
        "stripes",
        children=[
            square(f"stripe-{i}", side_length=0.5, fill=Fill(color, opacity=1.0), stroke=Stroke(WHITE, width=4), transform=Transform(position=(-2.0 + 0.5 * i, 0.0, 0.0), scale=(1.0, 8.0, 1.0)))
            for i, color in enumerate([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED, ORANGE, YELLOW])
        ],
    )
    soft_mask = Group(
        "soft-mask",
        children=[
            circle("disc", radius=1.6, fill=Fill(opacity=1.0), stroke=Stroke(width=0)),
            square("veil", side_length=1.2, fill=Fill(opacity=0.4), stroke=Stroke(width=20, opacity=1.0), transform=Transform(position=(2.0, 0.0, 0.0))),
        ],
    )
    hard = ClippingGroup("hard", mask=regular_polygon("star", 6, radius=1.8, fill=Fill(opacity=1.0), stroke=Stroke(width=0)), content=stripes, transform=Transform(position=(-3.5, 0.0, 0.0)))
    soft = ClippingGroup("soft", mask=soft_mask, content=square("content", side_length=4.0, fill=Fill([BLUE, GREEN], opacity=1.0), stroke=Stroke(width=0)), transform=Transform(position=(1.5, 0.0, 0.0)).rotated(0.5, (0, 1, 0)))
    image = renderer.render(Group("root", children=[hard, soft]), config(background_color="#222222"))

    check(image, "clipping", update_references)


@pytest.mark.parametrize("view", ["default", "front", "side", "top", "below", "isometric"])
def test_blender_scene(renderer: WGPURenderer, update_references: bool, view: str) -> None:
    from manimx.examples.blender_scene import blender_scene, views

    image = renderer.render(blender_scene(), views(480, 270)[view])

    check(image, f"blender_scene_{view}", update_references)


def _perspective(eye: tuple, target: tuple) -> object:
    from manimx.renderer import Camera

    return Camera.look_at(eye, target, field_of_view=math.radians(45), near=0.5, far=200.0)


def test_meshes_and_lights(renderer: WGPURenderer, update_references: bool) -> None:
    from manimx import DirectionalLight, PbrMaterial, PointLight, Shape3D, SpotLight, Texture, box_mesh, sphere_mesh, torus_mesh
    from manimx.examples.blender_scene import _ground

    i, j = np.indices((64, 64))
    checker = ((i // 8 + j // 8) % 2 * 200 + 40).astype(np.uint8)
    texture = Texture(np.stack([checker, checker, checker, np.full_like(checker, 255)], axis=-1))
    shapes = [
        Shape3D("plastic", mesh=sphere_mesh(1.2), material=PbrMaterial(RED, roughness=0.3), transform=Transform(position=(-3, 0, 1.2))),
        Shape3D("gold", mesh=sphere_mesh(1.2), material=PbrMaterial("#d4af37", metallic=1.0, roughness=0.25), transform=Transform(position=(0, 0, 1.2))),
        Shape3D("torus", mesh=torus_mesh(1.0, 0.4), material=PbrMaterial(BLUE, roughness=0.6), transform=Transform(position=(3, 0, 1.2)).rotated(1.0, (1, 0, 0))),
        Shape3D("box", mesh=box_mesh(2, 2, 2), material=PbrMaterial(WHITE, base_color_texture=texture), transform=Transform(position=(0, -4, 1)).rotated(0.6, (0, 0, 1))),
        SpotLight.aimed("spot", (0, -4, 6), (0, -4, 0), color=YELLOW, intensity=60, inner_cone_angle=0.25, outer_cone_angle=0.35),
        DirectionalLight.aimed("sun", (-1, 1, -2), intensity=2.5),
        PointLight("lamp", color=GREEN, intensity=15, transform=Transform(position=(-3, -2, 2))),
    ]
    scene = Group("root", children=[_ground(), *shapes])

    image = renderer.render(scene, config(background_color="#3d3d3d", camera=_perspective((9, -12, 7), (0, -1, 1))))

    check(image, "meshes_and_lights", update_references)


def test_curves_3d(renderer: WGPURenderer, update_references: bool) -> None:
    from manimx import Curve3D
    from manimx.examples.blender_scene import _ground

    rainbow = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
    curves = [
        Curve3D.parametric("helix", lambda t: (2 * math.cos(t), 2 * math.sin(t), 0.25 * t), (0, 8 * math.pi), 128, stroke=Stroke(rainbow, width=12)),
        Curve3D.parametric(
            "trefoil",
            lambda t: (math.sin(t) + 2 * math.sin(2 * t), math.cos(t) - 2 * math.cos(2 * t), -math.sin(3 * t)),
            (0, 2 * math.pi),
            96,
            stroke=Stroke([WHITE, BLUE], width=[4, 30, 4]),
            transform=Transform(position=(7, -2, 3)),
        ),
        Curve3D.from_corners("zigzag", [(0, 0, 0), (1, 0, 2), (2, 0, 0), (3, 1, 2), (4, 2, 0)], stroke=Stroke(YELLOW, width=25, joint=JointType.ROUND, cap=CapStyle.ROUND), transform=Transform(position=(-8, 3, 0))),
    ]
    scene = Group("root", children=[_ground(), *curves])

    image = renderer.render(scene, config(background_color="#3d3d3d", camera=_perspective((14, -14, 10), (0, 0, 2))))

    check(image, "curves_3d", update_references)
