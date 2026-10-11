"""Image, and images in the fills of other shapes (Shape2D.fill_image)."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.drivers import render_image
from manimx.renderer import RenderConfig
from manimx.userland import Circle, Image
from manimx.userland.images import PIXELS_PER_UNIT, load_pixels
from PIL import Image as Pillow

from manimx import Texture

WIDTH, HEIGHT = 160, 90  # the default camera shows 8 units: 11.25 pixels per unit
RED, GREEN, BLUE, WHITE = (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 255, 255)


def quadrants() -> np.ndarray:
    """2 by 2 pixels: red, green on top; blue, white below."""
    return np.array([[RED, GREEN], [BLUE, WHITE]], np.uint8)


def render(*nodes) -> np.ndarray:
    return render_image(list(nodes), config=RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, background_color="#000000"))


def pixel(image: np.ndarray, x: float, y: float) -> tuple[int, ...]:
    scale = HEIGHT / 8
    return tuple(int(c) for c in image[int(HEIGHT / 2 - y * scale), int(WIDTH / 2 + x * scale)])


def near(seen: tuple[int, ...], expected: tuple[int, ...], tolerance: int = 8) -> bool:
    """Equal up to ``tolerance`` per channel: with MSAA each sample looks
    the image up at its own position, a little off the texel's center, and
    filtering mixes in a few percent of the neighbor."""
    return all(abs(a - b) <= tolerance for a, b in zip(seen, expected, strict=False))


def test_an_image_shows_its_pixels_upright_with_their_colors() -> None:
    image = render(Image(quadrants(), height=6.0))
    assert near(pixel(image, -1.5, 1.5), RED)
    assert near(pixel(image, 1.5, 1.5), GREEN)
    assert near(pixel(image, -1.5, -1.5), BLUE)
    assert near(pixel(image, 1.5, -1.5), WHITE)
    assert pixel(image, 3.5, 0.0)[:3] == (0, 0, 0)  # beyond the 6 by 6 square


def test_images_of_different_sizes_share_the_texture() -> None:
    big = np.zeros((64, 96, 4), np.uint8)
    big[...] = (0, 128, 255, 255)
    image = render(Image(big, height=2.0).move_to((-3.0, 0.0, 0.0)), Image(quadrants(), height=3.0).move_to((3.0, 0.0, 0.0)))
    assert near(pixel(image, -3.0, 0.0), (0, 128, 255, 255))
    assert near(pixel(image, 2.25, 0.75), RED)  # the small one keeps to its own pixels
    assert near(pixel(image, 3.75, -0.75), WHITE)


def test_fill_color_tints_and_transparent_pixels_show_what_is_behind() -> None:
    pixels = quadrants()
    pixels[0, 0, 3] = 0  # the red pixel transparent
    behind = Circle(0.5, color="#00FF00", fill_opacity=1.0, stroke_width=0.0).move_to((-1.5, 1.5, 0.0))
    image = render(behind, Image(pixels, height=6.0, fill_color="#FF0000"))
    assert near(pixel(image, -1.5, 1.5), GREEN)  # through the transparent pixel
    assert near(pixel(image, 1.5, -1.5), RED)  # white tinted red
    assert near(pixel(image, 1.5, 1.5)[:3], (0, 0, 0))  # green times red


def test_any_shape_can_show_an_image() -> None:
    circle = Circle(3.0, fill_image=Texture(quadrants()), fill_color="#FFFFFF", fill_opacity=1.0, stroke_width=0.0)
    image = render(circle)
    assert near(pixel(image, -1.5, 1.5), RED)  # the circle's box: 6 by 6, like the square above
    assert near(pixel(image, 1.5, -1.5), WHITE)
    assert pixel(image, 2.9, 2.9)[:3] == (0, 0, 0)  # outside the circle


def test_sizes_and_sources() -> None:
    pixels = np.zeros((270, 540, 4), np.uint8)
    points = Image(pixels).points.value
    np.testing.assert_allclose(np.ptp(points, axis=0), [540 / PIXELS_PER_UNIT, 270 / PIXELS_PER_UNIT])
    np.testing.assert_allclose(np.ptp(Image(pixels, width=4.0).points.value, axis=0), [4.0, 2.0])
    np.testing.assert_allclose(np.ptp(Image(pixels, height=1.0).points.value, axis=0), [2.0, 1.0])
    np.testing.assert_allclose(np.ptp(Image(pixels, 1.0, 1.0).points.value, axis=0), [1.0, 1.0])

    rgb = np.ones((2, 3, 3))
    assert load_pixels(rgb).shape == (2, 3, 4)
    assert load_pixels(rgb).max() == 255
    assert load_pixels(np.zeros((2, 3), np.uint8)).shape == (2, 3, 4)
    assert load_pixels(Pillow.new("L", (3, 2))).shape == (2, 3, 4)
    with pytest.raises(ValueError, match="pixels"):
        load_pixels(np.zeros((2, 3, 2)))


def test_images_load_from_files(tmp_path) -> None:
    path = tmp_path / "quadrants.png"
    Pillow.fromarray(quadrants()).save(path)
    np.testing.assert_array_equal(load_pixels(path), quadrants())
    assert near(pixel(render(Image(str(path), height=6.0)), -1.5, -1.5), BLUE)
