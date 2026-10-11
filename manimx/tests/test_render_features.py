"""What the core's renderer draws, feature by feature: 2D shapes (order,
strokes, transparency, both sides, masks), meshes and lights, shadows and
portals, as pixel checks, and larger pictures of each feature against
reference images in tests/references/ (``pytest --update-references``
rewrites them; review the changed PNGs before committing).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from manimx.animation import (
    ClippingGroup,
    Curve3D,
    DirectionalLight,
    Group,
    PointLight,
    Shape3D,
    SpotLight,
)
from manimx.drivers import render_image
from manimx.examples.blender_scene import _ground, blender_scene
from manimx.examples.blender_scene import views as blender_views
from manimx.examples.manim_logo import config as logo_config
from manimx.examples.manim_logo import manim_logo
from manimx.examples.portal_cube import portal_cube
from manimx.examples.portal_cube import views as portal_views
from manimx.renderer import Camera, RenderConfig
from manimx.testing import ImageTolerance, assert_image_matches
from manimx.userland import Circle, Polyline, RegularPolygon, Square
from manimx.userland.geometry import (
    box_mesh,
    corner_points,
    parametric_points,
    sphere_mesh,
    torus_mesh,
)
from manimx.utils.color import BLUE, GREEN, ORANGE, PURPLE, RED, WHITE, YELLOW

from manimx import CapStyle, JointType, PbrMaterial, Texture

REFERENCES = Path(__file__).parent / "references"
WIDTH, HEIGHT = 160, 90  # the default camera shows 8 units: 11.25 pixels per unit
BACKGROUND = (0, 0, 0, 255)
RED_PIXEL, BLUE_PIXEL = (255, 0, 0, 255), (0, 0, 255, 255)


def render(*nodes, **config) -> np.ndarray:
    return render_image(list(nodes), config=RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, **config))


def pixel(image: np.ndarray, x: float, y: float) -> tuple[int, ...]:
    """The pixel at world (x, y) with the default camera."""
    scale = HEIGHT / 8
    return tuple(int(c) for c in image[int(HEIGHT / 2 - y * scale), int(WIDTH / 2 + x * scale)])


def brightness(image: np.ndarray, x: float, y: float) -> int:
    return sum(pixel(image, x, y)[:3])


def square(side: float, color: str = "#FF0000", opacity: float = 1.0) -> Square:
    return Square(side, color=color, fill_opacity=opacity, stroke_width=0.0)


def disc(radius: float, color: str = "#0000FF", opacity: float = 1.0) -> Circle:
    return Circle(radius, color=color, fill_opacity=opacity, stroke_width=0.0)


# -- 2D shapes ----------------------------------------------------------------


def test_fills_cover_their_shape_over_the_background() -> None:
    image = render(square(2.0))

    assert image.shape == (HEIGHT, WIDTH, 4) and image.dtype == np.uint8
    assert pixel(image, 0, 0) == RED_PIXEL
    assert pixel(image, 3, 3) == BACKGROUND


def test_a_group_moves_its_children() -> None:
    image = render(Group(square(1.0)).move_to((3.0, 0.0, 0.0)))

    assert pixel(image, 3, 0) == RED_PIXEL
    assert pixel(image, 0, 0) == BACKGROUND


def test_later_shapes_paint_over_earlier_ones() -> None:
    assert pixel(render(square(2.0), disc(0.5)), 0, 0) == BLUE_PIXEL
    assert pixel(render(disc(0.5), square(2.0)), 0, 0) == RED_PIXEL


def test_late_shapes_stay_ordered_far_away_in_perspective() -> None:
    # Two overlapping shapes in an upright plane 20 units away, after 200
    # other shapes, so their layers are in the hundreds. Far away the depth
    # is close to 1, where a depth bias per layer is too coarse: the shape
    # below used to show through in streaks.
    others = [square(0.1).move_to((0.0, -40.0, 0.0)) for _ in range(200)]
    upright = Group(square(2.0).move_to((0.3, 0.4, 0.0)), disc(1.0)).rotate(math.pi / 2, (1, 0, 0)).move_to((0.0, 8.0, 4.0))
    camera = Camera.look_at((0.0, -12.0, 2.5), (0.0, 8.0, 4.0), field_of_view=math.radians(48), near=0.1, far=160.0)

    image = render_image([*others, upright], config=RenderConfig(pixel_width=960, pixel_height=540, camera=camera))

    middle = image[250:290, 460:500, :3].astype(int)  # well inside the circle at the image center
    assert (np.abs(middle - BLUE_PIXEL[:3]).sum(axis=-1) < 60).all()


def test_the_stroke_is_drawn_over_its_fill() -> None:
    shape = Square(2.0, fill_color="#FF0000", fill_opacity=1.0, stroke_color="#0000FF", stroke_width=20.0)

    assert pixel(render(shape), 1.0, 0.0) == BLUE_PIXEL


def test_transparent_fills_blend_over_the_background() -> None:
    r, g, b, _ = pixel(render(square(2.0, "#FFFFFF", 0.5)), 0, 0)

    assert 120 <= r <= 135 and r == g == b


def test_the_back_side_looks_like_the_front() -> None:
    np.testing.assert_array_equal(render(disc(1.0, "#FF0000")), render(disc(1.0, "#FF0000").rotate(math.pi, (0, 1, 0))))


def test_without_msaa() -> None:
    assert pixel(render(square(2.0), msaa_samples=1), 0, 0) == RED_PIXEL


# -- clipping -----------------------------------------------------------------


def test_content_shows_only_inside_its_mask() -> None:
    image = render(ClippingGroup(disc(1.0), square(4.0)))

    assert pixel(image, 0, 0) == RED_PIXEL
    assert pixel(image, 1.6, 1.6) == BACKGROUND  # inside the square, outside the circle


def test_a_soft_mask_multiplies_alpha() -> None:
    r, g, b, _ = pixel(render(ClippingGroup(square(2.0, opacity=0.5), square(2.0, "#FFFFFF"))), 0, 0)

    assert 120 <= r <= 135 and r == g == b


def test_nested_masks_intersect() -> None:
    inner = ClippingGroup(square(2.0).move_to((0.5, 0.0, 0.0)), square(6.0, "#0000FF"))
    image = render(ClippingGroup(square(2.0).move_to((-0.5, 0.0, 0.0)), inner))

    assert pixel(image, 0.0, 0.0) == BLUE_PIXEL  # in both bands
    assert pixel(image, -1.2, 0.0) == BACKGROUND  # only in the outer band
    assert pixel(image, 1.2, 0.0) == BACKGROUND  # only in the inner band


def test_a_one_sided_mask_closes_from_behind() -> None:
    def portal(flip: bool, one_sided: bool) -> ClippingGroup:
        world = Group(square(2.0).move_to((0.0, 0.0, -1.0)))
        mask = square(2.0, "#FFFFFF")
        if flip:
            mask.rotate(math.pi, (0, 1, 0))
        return ClippingGroup(mask, world, one_sided=one_sided)

    assert pixel(render(portal(flip=False, one_sided=True)), 0, 0)[:3] == RED_PIXEL[:3]
    assert pixel(render(portal(flip=True, one_sided=True)), 0, 0)[:3] == (0, 0, 0)
    assert pixel(render(portal(flip=True, one_sided=False)), 0, 0)[:3] == RED_PIXEL[:3]


# -- meshes and lights --------------------------------------------------------


def test_a_mesh_is_lit_from_the_lights_side() -> None:
    sphere = Shape3D(sphere_mesh(2.0), color="#FFFFFF", roughness=1.0)
    light = DirectionalLight(direction=(1.0, 0.0, 0.0))  # shines towards +x

    image = render(sphere, light, ambient_intensity=0.0)

    assert brightness(image, -1.5, 0.0) > 200  # faces the light
    assert brightness(image, 1.5, 0.0) == 0  # faces away, no ambient


def test_a_spot_light_lights_only_its_cone() -> None:
    floor = Shape3D(box_mesh(8.0, 8.0, 0.1), color="#FFFFFF", roughness=1.0)
    spot = SpotLight(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), intensity=40.0, outer_cone_angle=0.3)

    image = render(floor, spot, ambient_intensity=0.0)

    assert brightness(image, 0.0, 0.0) > 200  # inside the cone
    assert brightness(image, 3.0, 0.0) == 0  # outside the cone


def test_meshes_and_2d_shapes_share_depth() -> None:
    wall = Shape3D(box_mesh(4.0, 4.0, 0.2), color="#000000", roughness=1.0)
    behind = square(2.0).move_to((0.0, 0.0, -1.0))
    front = disc(0.5).move_to((1.0, 0.0, 1.0))

    image = render(front, wall, behind, ambient_intensity=0.0, default_lights=False)

    assert pixel(image, -1.5, 0.0)[:3] == (0, 0, 0)  # the wall hides the red square
    assert pixel(image, 1.0, 0.0) == BLUE_PIXEL  # the circle is in front


def test_a_light_inside_a_clipping_group_lights_only_its_content() -> None:
    def wall(x: float) -> Shape3D:
        return Shape3D(box_mesh(2.0, 2.0, 0.2), color="#FFFFFF", roughness=1.0).move_to((x, 0.0, 0.0))

    portal = ClippingGroup(square(2.0, "#FFFFFF"), Group(wall(0.0), DirectionalLight(direction=(0.0, 0.0, -1.0), intensity=3.0)))
    image = render(portal, wall(3.0), ambient_intensity=0.0, default_lights=False)

    assert brightness(image, 0.0, 0.0) > 300  # inside the portal: lit
    assert brightness(image, 3.0, 0.0) == 0  # outside: unlit


# -- shadows ------------------------------------------------------------------

SHADOW_CAMERA = Camera.look_at((0.0, 0.0, 12.0), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), field_of_view=math.radians(60), near=0.1, far=50.0)


def floor_brightness(x: float, caster: bool = True, size: float = 0.03, **config: object) -> float:
    """The floor's brightness at world x, beside a block 2 units high,
    under a sun coming in at 45 degrees; the camera looks straight down."""
    floor = Shape3D(box_mesh(12.0, 12.0, 0.1), color="#FFFFFF", roughness=1.0)
    block = Shape3D(box_mesh(2.0, 2.0, 2.0), color="#FFFFFF", cast_shadows=caster).move_to((0.0, 0.0, 1.0))
    sun = DirectionalLight(direction=(1.0, 0.0, -1.0), intensity=3.0, size=size)
    settings = RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, camera=SHADOW_CAMERA, ambient_intensity=0.0, default_lights=False, **config)
    image = render_image([floor, block, sun], config=settings)
    scale = HEIGHT / (2 * 12 * math.tan(math.radians(30)))
    return float(image[HEIGHT // 2, int(WIDTH / 2 + x * scale), :3].astype(int).mean())


def test_a_block_shadows_the_floor() -> None:
    lit = floor_brightness(4.5)

    assert lit > 100
    assert floor_brightness(2.0) < lit / 3
    assert floor_brightness(2.0, shadows=False) > lit / 2
    assert floor_brightness(2.0, caster=False) > lit / 2


def test_a_larger_light_widens_the_penumbra() -> None:
    def partial(size: float) -> int:
        profile = np.array([floor_brightness(x, size=size) for x in np.linspace(1.0, 4.5, 15)])
        return int(np.sum((profile > 0.15 * profile.max()) & (profile < 0.85 * profile.max())))

    assert partial(0.4) > partial(0.0)


# -- pictures -----------------------------------------------------------------


def config(**options: object) -> RenderConfig:
    return RenderConfig(pixel_width=480, pixel_height=270, **options)  # type: ignore[arg-type]


def perspective(eye: tuple, target: tuple) -> Camera:
    return Camera.look_at(eye, target, field_of_view=math.radians(45), near=0.5, far=200.0)


def check(image: np.ndarray, name: str, update: bool, tolerance: ImageTolerance = ImageTolerance()) -> None:
    assert_image_matches(image, REFERENCES / f"{name}.png", tolerance=tolerance, update=update)


def test_manim_logo(update_references: bool) -> None:
    check(render_image(manim_logo(), config=logo_config(480, 270)), "manim_logo", update_references)


def test_joints_and_caps(update_references: bool) -> None:
    zigzag = [(-1.0, -0.5), (-0.5, 0.5), (0.0, -0.5), (0.5, 0.5), (1.0, -0.5)]
    styles = [(JointType.MITER, CapStyle.BUTT), (JointType.ROUND, CapStyle.ROUND), (JointType.BEVEL, CapStyle.SQUARE), (JointType.AUTO, CapStyle.AUTO)]
    shapes = [Polyline(zigzag, color=WHITE, stroke_width=30, stroke_joint=joint, stroke_cap=cap).move_to((-4.5 + 3.0 * i, 0.0, 0.0)) for i, (joint, cap) in enumerate(styles)]

    check(render_image(shapes, config=config()), "joints_and_caps", update_references)


def test_per_point_stroke_styles(update_references: bool) -> None:
    rainbow = Circle(1.5, stroke_color=[RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED], stroke_width=25).move_to((-3.0, 0.0, 0.0))
    taper = Polyline([(-1.5, -1.0), (0.0, 1.0), (1.5, -1.0)], stroke_color=[BLUE, GREEN], stroke_width=[2, 60], stroke_joint=JointType.ROUND, stroke_cap=CapStyle.ROUND).move_to((2.5, 0.0, 0.0))

    check(render_image([rainbow, taper], config=config()), "per_point_stroke_styles", update_references)


def test_transparency_and_gradients(update_references: bool) -> None:
    shapes = [
        RegularPolygon(5, 1.5, fill_color=[RED, BLUE], fill_opacity=0.7, stroke_color=YELLOW, stroke_width=12, stroke_opacity=0.5).move_to((-3.0, 0.0, 0.0)),
        Square(2.0, fill_color=RED, fill_opacity=0.5, stroke_color=WHITE, stroke_width=8).move_to((1.5, 0.5, 0.0)),
        Square(2.0, fill_color=BLUE, fill_opacity=0.5, stroke_color=WHITE, stroke_width=8).move_to((2.5, -0.5, 0.0)),
    ]

    check(render_image(shapes, config=config(background_color="#222222")), "transparency_and_gradients", update_references)


def test_shapes_in_3d(update_references: bool) -> None:
    flipped = Group(Circle(1.0, fill_color=GREEN, fill_opacity=1.0, stroke_color=WHITE, stroke_width=8), square(0.8, "#FC6255")).move_to((-3.0, 0.0, 0.0)).rotate(math.pi, (0, 1, 0))
    tilted = [
        Square(2.0, fill_color=PURPLE, fill_opacity=1.0, stroke_color=YELLOW, stroke_width=10).move_to((2.5, 2.0 - 2.0 * i, 0.0)).rotate(angle, (1, 0, 0)).rotate(0.4, (0, 1, 0))
        for i, angle in enumerate([0.8, 1.3, 1.5])
    ]

    check(render_image([flipped, *tilted], config=config()), "shapes_in_3d", update_references)


def test_clipping(update_references: bool) -> None:
    colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED, ORANGE, YELLOW]
    stripes = Group(*(Square(0.5, fill_color=color, fill_opacity=1.0, stroke_color=WHITE, stroke_width=4).move_to((-2.0 + 0.5 * i, 0.0, 0.0)).set_scale((1.0, 8.0, 1.0)) for i, color in enumerate(colors)))
    hard = ClippingGroup(RegularPolygon(6, 1.8, fill_opacity=1.0, stroke_width=0.0), stripes).move_to((-3.5, 0.0, 0.0))
    soft_mask = Group(Circle(1.6, fill_opacity=1.0, stroke_width=0.0), Square(1.2, fill_opacity=0.4, stroke_width=20).move_to((2.0, 0.0, 0.0)))
    soft = ClippingGroup(soft_mask, Square(4.0, fill_color=[BLUE, GREEN], fill_opacity=1.0, stroke_width=0.0)).move_to((1.5, 0.0, 0.0)).rotate(0.5, (0, 1, 0))

    check(render_image([hard, soft], config=config(background_color="#222222")), "clipping", update_references)


@pytest.mark.parametrize("view", ["default", "front", "side", "top", "below", "isometric"])
def test_blender_scene(update_references: bool, view: str) -> None:
    check(render_image(blender_scene(), config=blender_views(480, 270)[view]), f"blender_scene_{view}", update_references)


@pytest.mark.parametrize("view", ["front", "right", "back", "left", "front-right", "back-left"])
def test_portal_cube(update_references: bool, view: str) -> None:
    check(render_image(portal_cube(), config=portal_views(480, 270)[view]), f"portal_cube_{view}", update_references)


def test_meshes_and_lights(update_references: bool) -> None:
    i, j = np.indices((64, 64))
    checker = ((i // 8 + j // 8) % 2 * 200 + 40).astype(np.uint8)
    texture = Texture(np.stack([checker, checker, checker, np.full_like(checker, 255)], axis=-1))
    nodes = [
        _ground(),
        Shape3D(sphere_mesh(1.2), color=RED, roughness=0.3).move_to((-3, 0, 1.2)),
        Shape3D(sphere_mesh(1.2), color="#d4af37", metallic=1.0, roughness=0.25).move_to((0, 0, 1.2)),
        Shape3D(torus_mesh(1.0, 0.4), color=BLUE, roughness=0.6).move_to((3, 0, 1.2)).rotate(1.0, (1, 0, 0)),
        Shape3D(box_mesh(2, 2, 2), material=PbrMaterial(WHITE, base_color_texture=texture)).move_to((0, -4, 1)).rotate(0.6, (0, 0, 1)),
        SpotLight(position=(0, -4, 6), target=(0, -4, 0), color=YELLOW, intensity=60, inner_cone_angle=0.25, outer_cone_angle=0.35),
        DirectionalLight(direction=(-1, 1, -2), intensity=2.5),
        PointLight(position=(-3, -2, 2), color=GREEN, intensity=15),
    ]

    image = render_image(nodes, config=config(background_color="#3d3d3d", camera=perspective((9, -12, 7), (0, -1, 1))))

    check(image, "meshes_and_lights", update_references)


def test_curves_3d(update_references: bool) -> None:
    rainbow = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
    curves = [
        Curve3D(parametric_points(lambda t: (2 * math.cos(t), 2 * math.sin(t), 0.25 * t), (0, 8 * math.pi), 128), color=rainbow, stroke_width=12),
        Curve3D(parametric_points(lambda t: (math.sin(t) + 2 * math.sin(2 * t), math.cos(t) - 2 * math.cos(2 * t), -math.sin(3 * t)), (0, 2 * math.pi), 96), color=[WHITE, BLUE], stroke_width=[4, 30, 4]).move_to((7, -2, 3)),
        Curve3D(corner_points([(0, 0, 0), (1, 0, 2), (2, 0, 0), (3, 1, 2), (4, 2, 0)]), color=YELLOW, stroke_width=25, stroke_joint=JointType.ROUND, stroke_cap=CapStyle.ROUND).move_to((-8, 3, 0)),
    ]

    image = render_image([_ground(), *curves], config=config(background_color="#3d3d3d", camera=perspective((14, -14, 10), (0, 0, 2))))

    check(image, "curves_3d", update_references)


def test_shadows(update_references: bool) -> None:
    nodes = [
        Shape3D(box_mesh(16, 16, 0.2), color="#d0d0d0", roughness=0.9).move_to((0, 0, -0.1)),
        Shape3D(sphere_mesh(1.0), color=RED, roughness=0.4).move_to((-3, 1, 1.0)),
        Shape3D(box_mesh(0.4, 0.4, 4), color=BLUE).move_to((3, 2, 2)),
        manim_logo().move_to((0, 0.5, 1.8)).set_scale(0.6).rotate(math.pi / 2, (1, 0, 0)),
        Curve3D(parametric_points(lambda t: (3 + 0.8 * math.cos(t), -2.5 + 0.8 * math.sin(t), 0.3 + 0.12 * t), (0, 6 * math.pi), 96), color=YELLOW, stroke_width=12),
        DirectionalLight(direction=(0.6, 1.0, -1.2), intensity=2.5),
        SpotLight(position=(-5, -3, 6), target=(-3, 1, 0), intensity=80, inner_cone_angle=0.35, outer_cone_angle=0.5, color="#ffd9a0", size=0.3),
        PointLight(position=(1.5, -2.0, 1.2), intensity=6, color="#a0c8ff", size=0.05),
    ]

    image = render_image(nodes, config=config(background_color="#3d3d3d", ambient_intensity=0.08, camera=perspective((9, -12, 8), (0, 0, 1))))

    check(image, "shadows", update_references)
