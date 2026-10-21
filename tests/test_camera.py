from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from manim import (
    BLUE,
    LEFT,
    ORIGIN,
    RED,
    RIGHT,
    UP,
    UR,
    Camera,
    CapStyleType,
    ImageMobjectFromCamera,
    LineJointType,
    Mobject,
    MovingCamera,
    MultiCamera,
    Scene,
    Square,
    ThreeDCamera,
    VMobject,
    config,
    tempconfig,
)
from manim.renderer.insanity_adapter import InsanityAdapter
from manim.renderer.protocol import CapabilityException
from manim.utils.color import color_to_int_rgba


def test_movingcamera_auto_zoom():
    camera = MovingCamera()
    square = Square()
    margin = 0.5
    camera.auto_zoom([square], margin=margin, animate=False)
    assert camera.frame.height == square.height + margin


@pytest.mark.parametrize("camera_class", [Camera, ThreeDCamera])
@pytest.mark.parametrize("pixel_shape", [(96, 160), (192, 96)])
def test_default_camera_preserves_square_geometry(camera_class, pixel_shape):
    width, height = pixel_shape
    with tempconfig({"pixel_width": width, "pixel_height": height}):
        camera = camera_class()
        assert camera.frame_width == pytest.approx(config.frame_width)
        assert camera.frame_width / camera.frame_height == pytest.approx(width / height)
        frame_points = camera.frame.points.copy()
        renderer = InsanityAdapter(camera=camera)
        try:
            renderer.render_mobjects(
                [Square(fill_color="#ffffff", fill_opacity=1, stroke_width=0)],
            )
            pixels = renderer.get_frame()
            rows, columns = np.where(pixels[:, :, 0] > 128)
            assert len(rows) > 0
            assert np.ptp(columns) == pytest.approx(np.ptp(rows), abs=1)
            np.testing.assert_array_equal(camera.frame.points, frame_points)
        finally:
            renderer.close()


@pytest.mark.parametrize(
    ("dimensions", "expected_width", "expected_height"),
    [
        ({"frame_width": 6}, 6, 12),
        ({"frame_height": 6}, 3, 6),
        ({"frame_width": 6, "frame_height": 3}, 6, 3),
    ],
)
def test_camera_resolves_only_unspecified_dimensions(
    dimensions, expected_width, expected_height
):
    with tempconfig({"pixel_width": 60, "pixel_height": 120}):
        camera = Camera(**dimensions)
        assert camera.frame_width == pytest.approx(expected_width)
        assert camera.frame_height == pytest.approx(expected_height)


def test_default_scene_camera_auto_zoom():
    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene = Scene()
        square = Square().move_to([2, 0, 0])
        scene.play(scene.camera.auto_zoom([square], margin=0.5))

    assert scene.camera.frame_center.tolist() == square.get_center().tolist()
    assert scene.camera.frame_height == square.height + 0.5


def test_mobject_get_image_uses_temporary_renderer():
    with tempconfig({"pixel_width": 32, "pixel_height": 18}):
        image = Square(color=BLUE, fill_opacity=1).get_image()

    pixels = np.asarray(image)
    assert image.size == (32, 18)
    assert np.any(pixels[:, :, 2] > 0)


def test_renderer_owns_background_readback():
    with tempconfig({"pixel_width": 8, "pixel_height": 4}):
        renderer = InsanityAdapter(
            camera=Camera(background_color=RED, background_opacity=0.5),
        )
        try:
            renderer.update_frame(None, mobjects=[Mobject()])
            pixels = renderer.get_frame()
            # color_to_int_rgba truncates 0.5 * 255; the renderer rounds it.
            expected = color_to_int_rgba(RED, 0.5).astype(int)
            assert np.abs(pixels.astype(int) - expected).max() <= 1

            pixels[:] = 0
            assert np.abs(renderer.get_frame().astype(int) - expected).max() <= 1
        finally:
            renderer.close()


def test_background_images_are_not_drawn_yet(tmp_path):
    image_path = tmp_path / "background.png"
    Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(image_path)

    with tempconfig({"pixel_width": 2, "pixel_height": 2}):
        renderer = InsanityAdapter(camera=Camera(background_image=str(image_path)))
        try:
            with pytest.raises(CapabilityException, match="background image"):
                renderer.render_mobjects([Mobject()])
        finally:
            renderer.close()


@pytest.mark.parametrize(
    "operation",
    [
        lambda renderer: renderer.render_mobjects([]),
        lambda renderer: renderer.get_frame(),
        lambda renderer: renderer.get_image(),
    ],
    ids=["draw", "pixels", "image"],
)
def test_closed_renderer_rejects_drawing(operation):
    with tempconfig({"pixel_width": 8, "pixel_height": 4}):
        renderer = InsanityAdapter(skip_animations=True)
        renderer.close()
        with pytest.raises(RuntimeError, match="closed"):
            operation(renderer)


def test_multicamera_reports_recursive_view_controls_without_recursing_cycles():
    three_d_camera = ThreeDCamera()
    nested_camera = MultiCamera([ImageMobjectFromCamera(three_d_camera)])
    primary_camera = MultiCamera([ImageMobjectFromCamera(nested_camera)])
    nested_camera.add_image_mobject_from_camera(ImageMobjectFromCamera(primary_camera))

    indicators = primary_camera.get_mobjects_indicating_movement()

    assert primary_camera.frame in indicators
    assert nested_camera.frame in indicators
    assert three_d_camera.theta_tracker in indicators
    assert three_d_camera.zoom_tracker in indicators


def _angled_line(**kwargs):
    return VMobject(stroke_width=20, **kwargs).set_points_as_corners(
        [LEFT, ORIGIN, LEFT + UP]
    )


def _render_pixels(mobjects):
    renderer = InsanityAdapter(camera=Camera())
    try:
        renderer.render_mobjects(mobjects)
        return renderer.get_frame().copy()
    finally:
        renderer.close()


@pytest.mark.parametrize(
    "styled_kwargs",
    [
        {"cap_style": CapStyleType.ROUND},
        {"cap_style": CapStyleType.SQUARE},
        {"joint_type": LineJointType.ROUND},
        {"joint_type": LineJointType.BEVEL},
    ],
)
def test_auto_stroke_style_does_not_inherit_previous_style(styled_kwargs):
    # The two mobjects do not overlap, so the drawing order must not matter.
    styled = _angled_line(**styled_kwargs).shift(3 * LEFT)
    default = _angled_line().shift(3 * RIGHT)

    np.testing.assert_array_equal(
        _render_pixels([styled, default]), _render_pixels([default, styled])
    )


@pytest.mark.parametrize("change", ["assign", "in_place"])
def test_redrawn_frames_follow_point_changes(change, monkeypatch):
    from manim.mobject import fields

    monkeypatch.setattr(fields, "STRICT", False)

    def frame_of(mobject, renderer):
        renderer.render_mobjects([mobject])
        return renderer.get_frame().copy()

    with tempconfig({"pixel_width": 64, "pixel_height": 36}):
        square = Square(fill_opacity=1)
        renderer, fresh = (
            InsanityAdapter(camera=Camera()),
            InsanityAdapter(camera=Camera()),
        )
        try:
            before = frame_of(square, renderer)
            assert np.array_equal(frame_of(square, renderer), before)
            if change == "assign":
                square.shift(RIGHT)
            else:
                square.points[:, 0] += 1  # deprecated, but must still be drawn
            after = frame_of(square, renderer)
            assert not np.array_equal(after, before)
            np.testing.assert_array_equal(after, frame_of(square, fresh))
        finally:
            renderer.close()
            fresh.close()


def test_mobjects_fixed_in_frame_are_not_drawn_yet():
    with tempconfig({"pixel_width": 64, "pixel_height": 36}):
        renderer = InsanityAdapter(camera=MovingCamera())
        hud = Square(1, fill_opacity=1, stroke_width=0).to_corner(UR).fix_in_frame()
        try:
            with pytest.raises(CapabilityException, match="fixed in frame"):
                renderer.render_mobjects([hud])
        finally:
            renderer.close()
