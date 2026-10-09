"""End-to-end tests of the WebGPU renderer on the real GPU (skipped without one)."""

import math

import numpy as np
import pytest
from manimx.userland.geometry import arc_points, rectangle_points
from manimx.renderer import Camera, RemoveObjects, RenderConfig, Renderer, RenderService
from manimx.robjects import RShape2D
from manimx.robjects.transform import rotation_towards
from manimx.utils.color import BLUE, RED

from manimx.robjects import Fill, RClippingGroup, RGroup, Stroke, Transform

WIDTH, HEIGHT = 160, 90  # 8 world units tall: 11.25 pixels per unit
BACKGROUND = (0, 0, 0, 255)


@pytest.fixture(scope="module")
def renderer() -> RenderService:
    try:
        return RenderService()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def render(renderer: RenderService, *children, **config) -> np.ndarray:
    return renderer.render(
        RGroup(children=list(children)),
        RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, **config),
    )


def pixel(image: np.ndarray, x: float, y: float) -> tuple[int, ...]:
    """Pixel at world coordinates (x, y) with the default camera."""
    scale = HEIGHT / 8
    return tuple(image[int(HEIGHT / 2 - y * scale), int(WIDTH / 2 + x * scale)])


def test_renderer_implements_interface(renderer: RenderService) -> None:
    assert isinstance(renderer, Renderer)


def test_renders_background_and_fill(renderer: RenderService) -> None:
    image = render(renderer, RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0)))

    assert image.shape == (HEIGHT, WIDTH, 4) and image.dtype == np.uint8
    assert pixel(image, 0, 0) == (*RED.to_int_rgb(), 255)
    assert pixel(image, 3, 3) == BACKGROUND


def test_group_transform_moves_children(renderer: RenderService) -> None:
    shape = RShape2D(points=rectangle_points(1.0, 1.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0))
    image = render(renderer, RGroup(children=[shape], transform=Transform(position=(3.0, 0.0, 0.0))))

    assert pixel(image, 3, 0) == (*RED.to_int_rgb(), 255)
    assert pixel(image, 0, 0) == BACKGROUND


def test_later_objects_paint_over_earlier_ones(renderer: RenderService) -> None:
    below = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0))
    above = RShape2D(points=arc_points(0.5), fill=Fill(BLUE, opacity=1.0), stroke=Stroke(width=0.0))

    assert pixel(render(renderer, below, above), 0, 0) == (*BLUE.to_int_rgb(), 255)
    assert pixel(render(renderer, above, below), 0, 0) == (*RED.to_int_rgb(), 255)


def test_late_layers_stay_ordered_far_away_in_perspective(renderer: RenderService) -> None:
    # Two overlapping shapes in an upright plane 20 units away, after 200
    # other objects, so their layers are in the hundreds. Far away the depth
    # is close to 1, where fwidth(depth) is too coarse for the layer bias:
    # the shape below used to show through in streaks.
    others = [
        RShape2D(points=rectangle_points(0.1, 0.1), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0), transform=Transform(position=(0.0, -40.0, 0.0)))
        for _ in range(200)
    ]
    below = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0), transform=Transform(position=(0.3, 0.4, 0.0)))
    above = RShape2D(points=arc_points(1.0, num_curves=16), fill=Fill(BLUE, opacity=1.0), stroke=Stroke(width=0.0))
    upright = Transform(position=(0.0, 8.0, 4.0), rotation=rotation_towards((0.0, 1.0, 0.0)))
    camera = Camera.look_at((0.0, -12.0, 2.5), (0.0, 8.0, 4.0), field_of_view=math.radians(48), near=0.1, far=160.0)

    image = renderer.render(
        RGroup(children=[*others, RGroup(children=[below, above], transform=upright)]),
        RenderConfig(pixel_width=960, pixel_height=540, camera=camera, msaa_samples=4),
    )

    disc = image[250:290, 460:500, :3].astype(int)  # well inside the circle at the image center
    red = (np.abs(disc - RED.to_int_rgb()).sum(axis=-1) < 60).sum()
    blue = (np.abs(disc - BLUE.to_int_rgb()).sum(axis=-1) < 60).sum()
    assert blue == disc.shape[0] * disc.shape[1] and red == 0


def test_stroke_is_drawn_over_its_fill(renderer: RenderService) -> None:
    shape = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(BLUE, width=20.0))

    assert pixel(render(renderer, shape), 1.0, 0.0) == (*BLUE.to_int_rgb(), 255)


def test_transparent_fill_blends_over_background(renderer: RenderService) -> None:
    shape = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill("#FFFFFF", opacity=0.5), stroke=Stroke(width=0.0))

    r, g, b, _ = pixel(render(renderer, shape), 0, 0)
    assert 120 <= r <= 135 and r == g == b


def test_back_side_renders_like_front(renderer: RenderService) -> None:
    def scene(flip: float) -> np.ndarray:
        shape = RShape2D(points=arc_points(), fill=Fill(RED, opacity=1.0), transform=Transform().rotated(flip, (0, 1, 0)))
        return render(renderer, shape)

    np.testing.assert_array_equal(scene(0.0), scene(math.pi))


def test_remove_command_clears_renderer_cache(renderer: RenderService) -> None:
    shape = RShape2D(points=arc_points(), fill=Fill(RED, opacity=1.0))
    render(renderer, shape)
    assert renderer.cache.get(shape.uid, "fill-mesh", shape.version("points")) is not None

    renderer.submit(RemoveObjects.subtree(shape))

    assert renderer.cache.get(shape.uid, "fill-mesh", shape.version("points")) is None


def test_without_msaa(renderer: RenderService) -> None:
    image = render(renderer, RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0)), msaa_samples=1)

    assert pixel(image, 0, 0) == (*RED.to_int_rgb(), 255)


def test_clipping_shows_content_only_inside_mask(renderer: RenderService) -> None:
    clip = RClippingGroup(
        mask=RShape2D(points=arc_points(1.0), fill=Fill(opacity=1.0), stroke=Stroke(width=0.0)),
        content=RShape2D(points=rectangle_points(4.0, 4.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0)),
    )

    image = render(renderer, clip)

    assert pixel(image, 0, 0) == (*RED.to_int_rgb(), 255)
    assert pixel(image, 1.6, 1.6) == BACKGROUND  # inside the square, outside the circle


def test_soft_mask_multiplies_alpha(renderer: RenderService) -> None:
    clip = RClippingGroup(
        mask=RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(opacity=0.5), stroke=Stroke(width=0.0)),
        content=RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill("#FFFFFF", opacity=1.0), stroke=Stroke(width=0.0)),
    )

    r, g, b, _ = pixel(render(renderer, clip), 0, 0)
    assert 120 <= r <= 135 and r == g == b


def test_nested_masks_intersect(renderer: RenderService) -> None:
    def band(x: float) -> object:
        return RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(opacity=1.0), stroke=Stroke(width=0.0), transform=Transform(position=(x, 0.0, 0.0)))

    content = RShape2D(points=rectangle_points(6.0, 6.0), fill=Fill(BLUE, opacity=1.0), stroke=Stroke(width=0.0))
    inner = RClippingGroup(mask=band(0.5), content=content)
    outer = RClippingGroup(mask=band(-0.5), content=inner)

    image = render(renderer, outer)

    assert pixel(image, 0.0, 0.0) == (*BLUE.to_int_rgb(), 255)  # in both bands
    assert pixel(image, -1.2, 0.0) == BACKGROUND  # only in the outer band
    assert pixel(image, 1.2, 0.0) == BACKGROUND  # only in the inner band


# --- meshes and lights ------------------------------------------------------------


def _brightness(image: np.ndarray, x: float, y: float) -> int:
    return sum(int(channel) for channel in pixel(image, x, y)[:3])


def test_mesh_is_lit_from_the_light_side(renderer: RenderService) -> None:
    from manimx import PbrMaterial
    from manimx.robjects import RDirectionalLight, RShape3D
    from manimx.userland.geometry import sphere_mesh

    sphere = RShape3D(mesh=sphere_mesh(2.0), material=PbrMaterial("#FFFFFF", roughness=1.0))
    light = RDirectionalLight(transform=Transform(rotation=rotation_towards((1.0, 0.0, 0.0))))  # shines towards +x

    image = render(renderer, sphere, light, ambient_intensity=0.0)

    assert _brightness(image, -1.5, 0.0) > 200  # faces the light
    assert _brightness(image, 1.5, 0.0) == 0  # faces away, no ambient


def test_spot_light_cone(renderer: RenderService) -> None:
    from manimx import PbrMaterial
    from manimx.robjects import RShape3D, RSpotLight
    from manimx.userland.geometry import box_mesh

    floor = RShape3D(mesh=box_mesh(8.0, 8.0, 0.1), material=PbrMaterial("#FFFFFF", roughness=1.0))
    spot = RSpotLight(transform=Transform(position=(0.0, 0.0, 4.0), rotation=rotation_towards(tuple(np.subtract((0.0, 0.0, 0.0), (0.0, 0.0, 4.0))))), intensity=40.0, outer_cone_angle=0.3)

    image = render(renderer, floor, spot, ambient_intensity=0.0)

    assert _brightness(image, 0.0, 0.0) > 200  # inside the cone
    assert _brightness(image, 3.0, 0.0) == 0  # outside the cone


def test_meshes_and_vectors_share_depth(renderer: RenderService) -> None:
    from manimx import PbrMaterial
    from manimx.robjects import RShape3D
    from manimx.userland.geometry import box_mesh

    wall = RShape3D(mesh=box_mesh(4.0, 4.0, 0.2), material=PbrMaterial("#000000", roughness=1.0))
    behind = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0), transform=Transform(position=(0.0, 0.0, -1.0)))
    front = RShape2D(points=arc_points(0.5), fill=Fill(BLUE, opacity=1.0), stroke=Stroke(width=0), transform=Transform(position=(1.0, 0.0, 1.0)))

    image = render(renderer, front, wall, behind, ambient_intensity=0.0, default_lights=False)

    assert pixel(image, -1.5, 0.0)[:3] == (0, 0, 0)  # the wall hides the red square
    assert pixel(image, 1.0, 0.0) == (*BLUE.to_int_rgb(), 255)  # the circle is in front


def test_frames_without_meshes_after_meshes_start_from_empty_3d_layers() -> None:
    """Mesh-free frames skip the deferred passes and rely on an empty
    G-buffer and lit layer: a mesh frame in between must not leave its
    depth (hiding shapes) or light behind."""
    from manimx import PbrMaterial
    from manimx.robjects import RShape3D
    from manimx.userland.geometry import sphere_mesh

    try:
        reused = RenderService()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")
    flat = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0), transform=Transform(position=(0.0, 0.0, -2.0)))
    sphere = RShape3D(mesh=sphere_mesh(2.0), material=PbrMaterial("#FFFFFF", emissive="#FFFFFF"))

    expected = render(RenderService(), flat)
    first = render(reused, flat)
    with_mesh = render(reused, flat, sphere)
    after = render(reused, flat)

    np.testing.assert_array_equal(first, expected)
    assert pixel(with_mesh, 0.0, 0.0) != pixel(expected, 0.0, 0.0)  # the sphere hides the square
    np.testing.assert_array_equal(after, expected)
