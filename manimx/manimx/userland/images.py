"""Images in the scene: :class:`Image` is a rectangle filled with a picture
(Manim's ``ImageMobject``).

It is a Shape2D like any other (``Shape2D.fill_image``): it moves, scales,
fades, clips and morphs like one, sits in the painter's order of 2D
shapes, and its colors come out exactly as in the file (2D colors are never
tone mapped). Any other shape can show an image too:
``Circle(2.0, fill_image=Texture(pixels), fill_color=WHITE, fill_opacity=1)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..animation import Shape2D
from ..robjects import Texture
from .geometry import rectangle_points

PIXELS_PER_UNIT = 1080 / 8
"""An image this many pixels high is one unit high by default: a 1080
pixel high image fills the default camera's 8 units, as in Manim."""


class Image(Shape2D):
    """``source`` (a file path, an ``(H, W, 3 or 4)`` array of uint8 or of
    floats from 0 to 1, a Pillow image or a Texture) on a rectangle
    centered on the origin.

    It is ``height`` high and ``width`` wide; with only one of them the
    other keeps the image's aspect, with neither the image is
    ``PIXELS_PER_UNIT`` pixels per unit. The fill is white and opaque and
    there is no outline unless ``style`` says otherwise; a fill color tints
    the image.
    """

    def __init__(self, source: Any, height: float | None = None, width: float | None = None, name: str | None = None, **style: Any) -> None:
        texture = source if isinstance(source, Texture) else Texture(load_pixels(source))
        pixel_width, pixel_height = texture.size
        aspect = pixel_width / pixel_height
        if height is None and width is None:
            height = pixel_height / PIXELS_PER_UNIT
        if height is None:
            height = width / aspect  # type: ignore[operator]
        if width is None:
            width = height * aspect
        style.setdefault("fill_color", "#FFFFFF")
        style.setdefault("fill_opacity", 1.0)
        style.setdefault("stroke_width", 0.0)
        super().__init__(rectangle_points(width, height), name, fill_image=texture, **style)
        self.texture = texture


def load_pixels(source: Any) -> np.ndarray:
    """``source`` as ``(H, W, 4)`` uint8 RGBA pixels (see :class:`Image`)."""
    if isinstance(source, (str, Path)):
        from PIL import Image as Pillow

        with Pillow.open(source) as opened:
            return np.asarray(opened.convert("RGBA"))
    if hasattr(source, "convert") and hasattr(source, "size"):  # a Pillow image
        return np.asarray(source.convert("RGBA"))
    pixels = np.asarray(source)
    if pixels.ndim == 2:
        pixels = np.stack([pixels] * 3, axis=-1)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(f"an image needs (H, W), (H, W, 3) or (H, W, 4) pixels, got shape {pixels.shape}")
    if pixels.dtype != np.uint8:
        pixels = np.round(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)
    if pixels.shape[2] == 3:
        pixels = np.concatenate([pixels, np.full((*pixels.shape[:2], 1), 255, np.uint8)], axis=2)
    return np.ascontiguousarray(pixels)


__all__ = ["Image"]
