"""Depth of field: the lens signals of Camera and depth_of_field.wgsl."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from manimx.animation import Camera, Scene
from manimx.drivers import render_image
from manimx.renderer import RenderConfig
from manimx.userland import Square

CONFIG = RenderConfig(pixel_width=160, pixel_height=90, background_color="#000000")


class TwoSquares(Scene):
    """A near square left, a far one right; the focus moves from the near
    one to the far one."""

    def __init__(self, aperture: float = 0.04) -> None:
        super().__init__()
        self.aperture = aperture

    def construct(self) -> None:
        self.camera.set_field_of_view(0.8)
        self.camera.set_aperture(self.aperture)
        self.add(Square(1.0, color="#ffffff", fill_opacity=1.0, stroke_width=0.0).move_to((-1.2, 0.0, 6.0)))
        self.add(Square(4.0, color="#ffffff", fill_opacity=1.0, stroke_width=0.0).move_to((4.0, 0.0, -10.0)))
        self.camera.set_focus(5.0)
        self.wait(0.5)
        self.play(self.camera.animate.focus_on((4.0, 0.0, -10.0)), run_time=1.0)


def _soft_pixels(image: np.ndarray, columns: slice) -> int:
    """Pixels between black and white: the softer the edges, the more."""
    values = image[:, columns, 0].astype(int)
    return int(((values > 20) & (values < 235)).sum())


def test_lens_signals_and_arguments() -> None:
    camera = Camera(position=(0.0, 0.0, 10.0))
    assert camera.render_camera().aperture == 0.0
    assert camera.render_camera().focus_distance == pytest.approx(11.0)
    camera.focus_on((3.0, 1.0, 2.0))
    assert camera.render_camera().focus_distance == pytest.approx(8.0)  # along the view axis
    camera.set_aperture(0.02)
    assert camera.render_camera().aperture == 0.02
    with pytest.raises(ValueError, match="in front"):
        camera.focus_on((0.0, 0.0, 20.0))
    with pytest.raises(ValueError, match="positive"):
        camera.set_focus(0.0)
    with pytest.raises(ValueError, match="negative"):
        camera.set_aperture(-1.0)


def test_things_out_of_focus_are_blurred() -> None:
    prepared = TwoSquares().prepare()
    sharp = render_image(TwoSquares(aperture=0.0).prepare(), 0.0, CONFIG)
    near_focus = render_image(prepared, 0.0, CONFIG)
    far_focus = render_image(prepared, 1.5, dataclasses.replace(CONFIG, time=1.5))
    near, far = slice(0, 80), slice(80, 160)
    # Focused on the near square: it stays sharp, the far one blurs.
    assert _soft_pixels(near_focus, near) <= _soft_pixels(sharp, near) + 12
    assert _soft_pixels(near_focus, far) > _soft_pixels(sharp, far) + 20
    # The focus pulled to the far square: now the near one blurs.
    assert _soft_pixels(far_focus, near) > _soft_pixels(sharp, near) + 20
    assert _soft_pixels(far_focus, far) < _soft_pixels(near_focus, far)


