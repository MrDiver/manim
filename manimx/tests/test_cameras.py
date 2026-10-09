"""Cameras: one per scene, moved by the scene, seen through by its clip."""

import numpy as np
import pytest
from manimx.animation import Camera, Scene, Timeline
from manimx.robjects import Transform
from manimx.userland import RIGHT, Dot
from manimx.userland.rate_functions import linear


def prepared(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def forward(render_camera):
    """Viewing direction: the camera's local -z in world space."""
    rotation = render_camera.transform
    return Transform(rotation=rotation.rotation).apply([(0.0, 0.0, -1.0)])[0]


def camera_at(timeline, t, index=0):
    return timeline.frame_at(t)[index].camera


def dolly(scene):
    scene.play(scene.camera.animate.shift(RIGHT * 2), run_time=2, rate=linear)
    scene.play(scene.camera.animate.zoom(4), run_time=2, rate=linear)


def test_a_camera_defaults_to_manims_view() -> None:
    camera = Camera(name="camera").render_camera()

    assert camera.transform.position == (0.0, 0.0, 11.0)
    assert camera.frame_height == 8.0 and camera.field_of_view is None
    np.testing.assert_allclose(forward(camera), (0, 0, -1), atol=1e-12)


def test_every_scene_has_its_own_camera_that_cannot_be_replaced() -> None:
    first, second = Scene(), Scene()

    assert first.camera is not second.camera and first.camera.name == "camera"
    with pytest.raises(AttributeError):
        first.camera = Camera()


def test_the_scene_camera_moves_with_the_clip() -> None:
    timeline = Timeline()
    timeline.add(prepared(dolly), start=1.0)

    assert timeline.frame_at(0.5) == ()  # before the clip: nothing to show
    assert camera_at(timeline, 2.0).transform.position[0] == pytest.approx(1.0)
    assert camera_at(timeline, 4.0).frame_height == pytest.approx(4.0)  # halfway to 2: uniform zoom
    assert timeline.frame_at(2.0)[0].root.children == []  # cameras are never drawn


def test_retiming_a_clip_retimes_its_camera() -> None:
    timeline = Timeline()
    clip = timeline.add(prepared(dolly)).slow_motion(0.0, 2.0, 2.0)  # the dolly takes 4 s
    timeline.resolve()

    assert clip.compiled_time(2.0) < 1.5
    for t in (1.0, 2.0, 3.0):  # the dolly moves 1 unit per second of scene time
        assert camera_at(timeline, t).transform.position[0] == pytest.approx(clip.compiled_time(t))
    assert camera_at(timeline, 4.0).transform.position[0] == pytest.approx(2.0)


def test_camera_changes_are_recorded_at_their_time() -> None:
    def later(scene):
        scene.wait(1)
        scene.camera.shift(RIGHT)  # an instant change after one second, without add()
        scene.wait(1)

    timeline = Timeline()
    timeline.add(prepared(later))

    assert camera_at(timeline, 0.5).transform.position[0] == 0.0
    assert camera_at(timeline, 1.5).transform.position[0] == 1.0


def test_a_camera_follows_a_node() -> None:
    def chase(scene):
        dot = Dot(name="dot")
        scene.add(dot)
        scene.camera.follow(dot, offset=(0, 0, 11))
        scene.play(dot.animate.shift(RIGHT * 2), run_time=2, rate=linear)

    timeline = Timeline()
    timeline.add(prepared(chase))

    assert camera_at(timeline, 1.0).transform.position == pytest.approx((1.0, 0.0, 11.0))


def test_look_at_and_perspective() -> None:
    def aim(scene):
        scene.camera.move_to((0, -10, 10))
        scene.camera.look_at((0, 0, 0))
        scene.wait(1)
        scene.play(scene.camera.animate.set_field_of_view(0.5), run_time=1)

    timeline = Timeline()
    timeline.add(prepared(aim))

    np.testing.assert_allclose(forward(camera_at(timeline, 0.5)), np.array([0, 1, -1]) / np.sqrt(2), atol=1e-9)
    assert camera_at(timeline, 0.5).field_of_view is None
    assert camera_at(timeline, 2.0).field_of_view == pytest.approx(0.5)
    straight_down = Camera(name="top")
    with pytest.raises(ValueError, match="own position"):
        straight_down.look_at((0, 0, 11))
    straight_down.look_at()  # looking along the default up: +y is up on screen instead
    np.testing.assert_allclose(forward(straight_down.render_camera()), (0, 0, -1), atol=1e-12)


def test_overlapping_clips_give_one_view_each_in_layer_order() -> None:
    def still(scene):
        scene.add(Dot(name="dot"))
        scene.wait(2)

    timeline = Timeline()
    top = timeline.add(prepared(still), name="top", layer=1)
    bottom = timeline.add(prepared(dolly), name="bottom")

    views = timeline.frame_at(1.0)
    assert [view.root.uid for view in views] == [bottom.uid, top.uid]
    assert views[0].camera.transform.position[0] == pytest.approx(1.0)  # each view has its own camera
    assert views[1].camera.transform.position[0] == 0.0
    assert [child.uid for child in timeline.state_at(1.0).children] == [bottom.uid, top.uid]


def test_a_prepared_scene_renders_through_its_camera() -> None:
    scene = prepared(dolly)

    (view,) = scene.frame_at(1.0)
    assert view.camera.transform.position[0] == pytest.approx(1.0)
    assert view.root.children == []
