"""Camera views: other cameras' pictures on planes in the scene
(manimx.animation.camera_view and the renderers' camera view texture)."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Camera, CameraView, ClippingGroup, Scene
from manimx.animation.plan import planned_frames
from manimx.drivers.core_pipeline import CoreFramePipeline
from manimx.renderer import RenderConfig, RenderService
from manimx.userland import Circle, FadeOut, Square

CONFIG = RenderConfig(pixel_width=160, pixel_height=90, background_color="#000000")
RED = "#ff0000"


@pytest.fixture(scope="module")
def service() -> RenderService:
    try:
        return RenderService.shared()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


class Monitor(Scene):
    """A red square far off screen, seen through a camera view at the
    center; the side camera moves away from it."""

    def construct(self) -> None:
        side = Camera("side", position=(20.0, 0.0, 11.0), frame_height=4.0)
        square = Square(3.0, color=RED, fill_opacity=1.0, stroke_width=0.0).move_to((20.0, 0.0, 0.0))
        self.add(square, CameraView(side, 4.0, 2.25, name="monitor"))
        self.wait(0.5)
        self.play(side.animate.shift((10.0, 0.0, 0.0)), run_time=1.0)


class Masked(Scene):
    """The monitor inside a clipping group with a small circle as mask."""

    def construct(self) -> None:
        side = Camera("side", position=(20.0, 0.0, 11.0), frame_height=4.0)
        square = Square(3.0, color=RED, fill_opacity=1.0, stroke_width=0.0).move_to((20.0, 0.0, 0.0))
        window = ClippingGroup(Circle(0.5, fill_opacity=1.0), CameraView(side, 4.0, 2.25), "window")
        self.add(square, window)
        self.wait(0.5)


def _center(image: np.ndarray, dx: int = 0) -> np.ndarray:
    return image[45, 80 + dx, :3]


def test_view_states_follow_the_scene() -> None:
    prepared = Monitor().prepare()
    (view,) = prepared.frame_at(0.0)
    (camera_view,) = view.camera_views
    assert camera_view.width == 4.0 and camera_view.aspect_ratio == pytest.approx(4.0 / 2.25)
    assert camera_view.camera.transform.position[0] == pytest.approx(20.0)
    (later,) = prepared.frame_at(1.5)
    assert later.camera_views[0].camera.transform.position[0] == pytest.approx(30.0)
    # The core reads the side camera's signals too.
    (planned,) = planned_frames(prepared).frame_at(1.5)
    assert planned.camera_views == later.camera_views


def test_faded_out_views_are_not_rendered() -> None:
    class Gone(Scene):
        def construct(self) -> None:
            view = CameraView(Camera("side"), 2.0, 2.0)
            self.add(view)
            self.play(FadeOut(view))

    prepared = Gone().prepare()
    assert prepared.frame_at(0.0)[0].camera_views
    assert prepared.frame_at(1.0)[0].camera_views == ()


def test_camera_view_arguments() -> None:
    with pytest.raises(TypeError, match="Camera"):
        CameraView(Square(), 1.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        CameraView(Camera(), 0.0, 1.0)


def test_the_picture_shows_the_other_camera(service) -> None:
    prepared = Monitor().prepare()
    image = service.render(prepared.frame_at(0.0), CONFIG)
    assert tuple(_center(image)) == (255, 0, 0)  # the square, through the monitor
    assert tuple(image[5, 5, :3]) == (0, 0, 0)  # outside the monitor: nothing
    moved = service.render(prepared.frame_at(1.5), CONFIG)
    assert tuple(_center(moved)) == (0, 0, 0)  # the side camera left the square


def test_every_path_draws_the_same(service) -> None:
    prepared = Monitor().prepare()
    planned = planned_frames(prepared)
    pipeline = CoreFramePipeline((), CONFIG)
    for time in (0.0, 0.9):
        reference = service.render(prepared.frame_at(time), CONFIG)
        np.testing.assert_array_equal(service.render(planned.frame_at(time), CONFIG), reference)
        future = pipeline.submit(0, time, planned.frame_at(time))
        pipeline.flush()
        np.testing.assert_array_equal(future.result(), reference)


def test_views_are_masked_and_never_show_views(service) -> None:
    image = service.render(Masked().prepare().frame_at(0.0), CONFIG)
    assert tuple(_center(image)) == (255, 0, 0)
    assert tuple(_center(image, 25)) == (0, 0, 0)  # inside the monitor, outside the mask
    # A camera looking at its own monitor sees the scene without monitors.
    class Mirror(Scene):
        def construct(self) -> None:
            self.add(CameraView(self.camera, 4.0, 2.25))

    mirror = service.render(Mirror().prepare().frame_at(0.0), CONFIG)
    assert not mirror[..., :3].any()
