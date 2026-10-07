"""End-to-end tests of the WebGPU renderer on the real GPU (skipped without one)."""

import math

import numpy as np
import pytest

from manimx import Fill, Group, Stroke, Transform
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
    assert renderer.cache.get(shape.uid, "fill-mesh", shape.version("points", "fill")) is not None

    renderer.submit(RemoveObjects.subtree(shape))

    assert renderer.cache.get(shape.uid, "fill-mesh", shape.version("points", "fill")) is None


def test_without_msaa(renderer: WGPURenderer) -> None:
    image = render(renderer, square("s", fill=Fill(RED, opacity=1.0)), msaa_samples=1)

    assert pixel(image, 0, 0) == (*RED.to_int_rgb(), 255)
