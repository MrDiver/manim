"""End-to-end tests of the WebGPU renderer on the real GPU (skipped without one)."""

import math

import numpy as np
import pytest

from manimx import ClippingGroup, Fill, Group, Stroke, Transform
from manimx.objects.geometry import circle, square
from manimx.renderer import RemoveObjects, RenderConfig, Renderer, WGPURenderer
from manimx.utils.color import BLUE, RED

WIDTH, HEIGHT = 160, 90  # 8 world units tall: 11.25 pixels per unit
BACKGROUND = (0, 0, 0, 255)


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def render(renderer: WGPURenderer, *children, **config) -> np.ndarray:
    return renderer.render(
        Group("root", children=list(children)),
        RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, **config),
    )


def pixel(image: np.ndarray, x: float, y: float) -> tuple[int, ...]:
    """Pixel at world coordinates (x, y) with the default camera."""
    scale = HEIGHT / 8
    return tuple(image[int(HEIGHT / 2 - y * scale), int(WIDTH / 2 + x * scale)])


def test_renderer_implements_interface(renderer: WGPURenderer) -> None:
    assert isinstance(renderer, Renderer)


def test_renders_background_and_fill(renderer: WGPURenderer) -> None:
    image = render(renderer, square("s", fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0)))

    assert image.shape == (HEIGHT, WIDTH, 4) and image.dtype == np.uint8
    assert pixel(image, 0, 0) == (*RED.to_int_rgb(), 255)
    assert pixel(image, 3, 3) == BACKGROUND


def test_group_transform_moves_children(renderer: WGPURenderer) -> None:
    shape = square("s", side_length=1.0, fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0))
    image = render(renderer, Group("g", children=[shape], transform=Transform(position=(3.0, 0.0, 0.0))))

    assert pixel(image, 3, 0) == (*RED.to_int_rgb(), 255)
    assert pixel(image, 0, 0) == BACKGROUND


def test_later_objects_paint_over_earlier_ones(renderer: WGPURenderer) -> None:
    below = square("below", fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0))
    above = circle("above", radius=0.5, fill=Fill(BLUE, opacity=1.0), stroke=Stroke(width=0.0))

    assert pixel(render(renderer, below, above), 0, 0) == (*BLUE.to_int_rgb(), 255)
    assert pixel(render(renderer, above, below), 0, 0) == (*RED.to_int_rgb(), 255)


def test_stroke_is_drawn_over_its_fill(renderer: WGPURenderer) -> None:
    shape = square("s", fill=Fill(RED, opacity=1.0), stroke=Stroke(BLUE, width=20.0))

    assert pixel(render(renderer, shape), 1.0, 0.0) == (*BLUE.to_int_rgb(), 255)


def test_transparent_fill_blends_over_background(renderer: WGPURenderer) -> None:
    shape = square("s", fill=Fill("#FFFFFF", opacity=0.5), stroke=Stroke(width=0.0))

    r, g, b, _ = pixel(render(renderer, shape), 0, 0)
    assert 120 <= r <= 135 and r == g == b


def test_back_side_renders_like_front(renderer: WGPURenderer) -> None:
    def scene(flip: float) -> np.ndarray:
        shape = circle("c", fill=Fill(RED, opacity=1.0), transform=Transform().rotated(flip, (0, 1, 0)))
        return render(renderer, shape)

    np.testing.assert_array_equal(scene(0.0), scene(math.pi))


def test_remove_command_clears_renderer_cache(renderer: WGPURenderer) -> None:
    shape = circle("c", fill=Fill(RED, opacity=1.0))
    render(renderer, shape)
    assert renderer.cache.get(shape.uid, "fill-mesh", shape.version("points")) is not None

    renderer.submit(RemoveObjects.subtree(shape))

    assert renderer.cache.get(shape.uid, "fill-mesh", shape.version("points")) is None


def test_without_msaa(renderer: WGPURenderer) -> None:
    image = render(renderer, square("s", fill=Fill(RED, opacity=1.0)), msaa_samples=1)

    assert pixel(image, 0, 0) == (*RED.to_int_rgb(), 255)


def test_clipping_shows_content_only_inside_mask(renderer: WGPURenderer) -> None:
    clip = ClippingGroup(
        "clip",
        mask=circle("mask", radius=1.0, fill=Fill(opacity=1.0), stroke=Stroke(width=0.0)),
        content=square("content", side_length=4.0, fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0)),
    )

    image = render(renderer, clip)

    assert pixel(image, 0, 0) == (*RED.to_int_rgb(), 255)
    assert pixel(image, 1.6, 1.6) == BACKGROUND  # inside the square, outside the circle


def test_soft_mask_multiplies_alpha(renderer: WGPURenderer) -> None:
    clip = ClippingGroup(
        "clip",
        mask=square("mask", fill=Fill(opacity=0.5), stroke=Stroke(width=0.0)),
        content=square("content", fill=Fill("#FFFFFF", opacity=1.0), stroke=Stroke(width=0.0)),
    )

    r, g, b, _ = pixel(render(renderer, clip), 0, 0)
    assert 120 <= r <= 135 and r == g == b


def test_nested_masks_intersect(renderer: WGPURenderer) -> None:
    def band(name: str, x: float) -> object:
        return square(name, side_length=2.0, fill=Fill(opacity=1.0), stroke=Stroke(width=0.0), transform=Transform(position=(x, 0.0, 0.0)))

    content = square("content", side_length=6.0, fill=Fill(BLUE, opacity=1.0), stroke=Stroke(width=0.0))
    inner = ClippingGroup("inner", mask=band("right", 0.5), content=content)
    outer = ClippingGroup("outer", mask=band("left", -0.5), content=inner)

    image = render(renderer, outer)

    assert pixel(image, 0.0, 0.0) == (*BLUE.to_int_rgb(), 255)  # in both bands
    assert pixel(image, -1.2, 0.0) == BACKGROUND  # only in the outer band
    assert pixel(image, 1.2, 0.0) == BACKGROUND  # only in the inner band


# --- meshes and lights ------------------------------------------------------------


def _brightness(image: np.ndarray, x: float, y: float) -> int:
    return sum(int(channel) for channel in pixel(image, x, y)[:3])


def test_mesh_is_lit_from_the_light_side(renderer: WGPURenderer) -> None:
    from manimx import DirectionalLight, PbrMaterial, Shape3D, sphere_mesh

    sphere = Shape3D("sphere", mesh=sphere_mesh(2.0), material=PbrMaterial("#FFFFFF", roughness=1.0))
    light = DirectionalLight.aimed("sun", (1.0, 0.0, 0.0))  # shines towards +x

    image = render(renderer, sphere, light, ambient_intensity=0.0)

    assert _brightness(image, -1.5, 0.0) > 200  # faces the light
    assert _brightness(image, 1.5, 0.0) == 0  # faces away, no ambient


def test_spot_light_cone(renderer: WGPURenderer) -> None:
    from manimx import PbrMaterial, Shape3D, SpotLight, box_mesh

    floor = Shape3D("floor", mesh=box_mesh(8.0, 8.0, 0.1), material=PbrMaterial("#FFFFFF", roughness=1.0))
    spot = SpotLight.aimed("spot", (0.0, 0.0, 4.0), (0.0, 0.0, 0.0), intensity=40.0, outer_cone_angle=0.3)

    image = render(renderer, floor, spot, ambient_intensity=0.0)

    assert _brightness(image, 0.0, 0.0) > 200  # inside the cone
    assert _brightness(image, 3.0, 0.0) == 0  # outside the cone


def test_meshes_and_vectors_share_depth(renderer: WGPURenderer) -> None:
    from manimx import PbrMaterial, Shape3D, box_mesh

    wall = Shape3D("wall", mesh=box_mesh(4.0, 4.0, 0.2), material=PbrMaterial("#000000", roughness=1.0))
    behind = square("behind", fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0), transform=Transform(position=(0.0, 0.0, -1.0)))
    front = circle("front", radius=0.5, fill=Fill(BLUE, opacity=1.0), stroke=Stroke(width=0), transform=Transform(position=(1.0, 0.0, 1.0)))

    image = render(renderer, front, wall, behind, ambient_intensity=0.0, default_lights=False)

    assert pixel(image, -1.5, 0.0)[:3] == (0, 0, 0)  # the wall hides the red square
    assert pixel(image, 1.0, 0.0) == (*BLUE.to_int_rgb(), 255)  # the circle is in front


def test_frames_without_meshes_after_meshes_start_from_empty_3d_layers() -> None:
    """Mesh-free frames skip the deferred passes and rely on an empty
    G-buffer and lit layer: a mesh frame in between must not leave its
    depth (hiding shapes) or light behind."""
    from manimx import PbrMaterial, Shape3D, sphere_mesh

    try:
        reused = WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")
    flat = square("flat", fill=Fill(RED, opacity=1.0), transform=Transform(position=(0.0, 0.0, -2.0)))
    sphere = Shape3D("sphere", mesh=sphere_mesh(2.0), material=PbrMaterial("#FFFFFF", emissive="#FFFFFF"))

    expected = render(WGPURenderer(), flat)
    first = render(reused, flat)
    with_mesh = render(reused, flat, sphere)
    after = render(reused, flat)

    np.testing.assert_array_equal(first, expected)
    assert pixel(with_mesh, 0.0, 0.0) != pixel(expected, 0.0, 0.0)  # the sphere hides the square
    np.testing.assert_array_equal(after, expected)
