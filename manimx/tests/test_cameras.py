"""Cameras and views (section 8.1 of the design doc)."""

import numpy as np
import pytest

from manimx.animation import LEFT, RIGHT, Camera, Dot, Scene, Timeline, View, World3D
from manimx.animation.rate_functions import linear
from manimx.robjects import Transform


def prepared(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def forward(render_camera):
    """Viewing direction: the camera's local -z in world space."""
    rotation = render_camera.transform
    return Transform(rotation=rotation.rotation).apply([(0.0, 0.0, -1.0)])[0]


def test_a_camera_defaults_to_manims_view() -> None:
    camera = Camera(name="camera").render_camera()

    assert camera.transform.position == (0.0, 0.0, 11.0)
    assert camera.frame_height == 8.0 and camera.field_of_view is None
    np.testing.assert_allclose(forward(camera), (0, 0, -1), atol=1e-12)


def test_camera_scenes_move_cameras_and_hold_outside_their_clip() -> None:
    camera = Camera(name="camera")

    def dolly(scene):
        scene.play(camera.animate.shift(RIGHT * 2), run_time=2, rate=linear)
        scene.play(camera.animate.zoom(4), run_time=2, rate=linear)

    timeline = Timeline()
    timeline.add(prepared(dolly), start=1.0)

    assert timeline.camera_at(camera, 0.0).transform.position[0] == 0.0  # before the clip: first state
    assert timeline.camera_at(camera, 2.0).transform.position[0] == pytest.approx(1.0)
    assert timeline.camera_at(camera, 4.0).frame_height == pytest.approx(4.0)  # halfway to 2: uniform zoom
    assert timeline.camera_at(camera, 9.0).frame_height == pytest.approx(2.0)  # after the clip: last state
    assert timeline.state_at(2.0).children[0].children == []  # cameras are never drawn


def test_look_at_and_perspective() -> None:
    camera = Camera(name="camera", position=(0, -10, 10))

    def aim(scene):
        scene.add(camera)
        camera.look_at((0, 0, 0))
        scene.wait(1)
        scene.play(camera.animate.set_field_of_view(0.5), run_time=1)

    timeline = Timeline()
    timeline.add(prepared(aim))

    np.testing.assert_allclose(forward(timeline.camera_at(camera, 0.5)), np.array([0, 1, -1]) / np.sqrt(2), atol=1e-9)
    assert timeline.camera_at(camera, 0.5).field_of_view is None
    assert timeline.camera_at(camera, 2.0).field_of_view == pytest.approx(0.5)
    straight_down = Camera(name="top")
    with pytest.raises(ValueError, match="own position"):
        straight_down.look_at((0, 0, 11))
    straight_down.look_at()  # looking along the default up: +y is up on screen instead
    np.testing.assert_allclose(forward(straight_down.render_camera()), (0, 0, -1), atol=1e-12)


def test_views_pair_a_world_with_a_camera_placed_by_its_clip() -> None:
    camera = Camera(name="camera")
    stage = World3D("stage")

    def content(scene):
        scene.add(Dot(name="dot"), camera)
        scene.wait(1)

    timeline = Timeline(worlds=[stage])
    timeline.add(prepared(content)).place(position=(1, 0, 0), scale=2)

    state = timeline.view_at(View(stage, camera), 0.5)
    assert state.root.uid == stage.uid and len(state.root.children) == 1
    assert state.camera.transform.position == (1.0, 0.0, 22.0)  # the clip's transform applies
    assert state.camera.frame_height == 16.0  # and its scale, so the view shows the same

    with pytest.raises(ValueError, match="plays in world"):
        timeline.view_at(View(World3D("elsewhere"), camera), 0.5)
    loose = Camera(name="loose", position=LEFT)
    assert timeline.view(View(stage, loose))(0.0).camera.transform.position == (-1.0, 0.0, 0.0)
