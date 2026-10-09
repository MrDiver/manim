"""Worlds and clip placement: transforms, layers and stills (section 8 of
the design doc)."""

import numpy as np
import pytest

from manimx.animation import RIGHT, SEGMENT_GAP, Dot, Scene, Timeline, World2D, World3D
from manimx.animation.rate_functions import linear
from manimx.objects import Transform, quaternion_from_axis_angle
from naming import by_name, clip_names, timeline_roots


def prepared(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def moving(scene):
    """A dot moving right 1 unit per second for 2 s, then (after "next") 2 s more."""
    dot = Dot(name="dot")
    scene.add(dot)
    scene.play(dot.animate.shift(RIGHT * 2), run_time=2, rate=linear)
    scene.wait_for("next")
    scene.play(dot.animate.shift(RIGHT * 2), run_time=2, rate=linear)


def dot_x(timeline, group):
    return by_name(group.children, timeline_roots(timeline))["dot"].transform.position[0]


def test_clips_play_in_the_default_world_with_stable_identities() -> None:
    timeline = Timeline(events={"next": 3.0})
    clip = timeline.add(prepared(moving), name="intro")

    first, later = timeline.state_at(0.5), timeline.state_at(4.0)
    assert isinstance(timeline.worlds[0], World3D)
    assert first.uid == later.uid == timeline.worlds[0].uid
    assert first.children[0].uid == later.children[0].uid == clip.uid
    assert clip.world is timeline.world("world")
    with pytest.raises(KeyError):
        timeline.world("nowhere")


def test_clip_transform_places_the_whole_scene() -> None:
    timeline = Timeline()
    clip = timeline.add(prepared(moving)).place(position=(0, 0, -3), scale=2)

    group = timeline.state_at(1.0).children[0]
    assert group.transform.position == (0.0, 0.0, -3.0) and group.transform.scale == (2.0, 2.0, 2.0)
    assert dot_x(timeline, group) == pytest.approx(1.0)  # the scene itself is unchanged

    clip.place(layer=3)  # other arguments stay
    assert clip.transform.position == (0.0, 0.0, -3.0) and clip.layer == 3


def test_layers_order_drawing_and_ties_keep_insertion_order() -> None:
    timeline = Timeline()
    timeline.add(prepared(moving), name="top", layer=2)
    timeline.add(prepared(moving), name="first")
    timeline.add(prepared(moving), name="second")

    assert clip_names(timeline.state_at(0.0).children, timeline) == ["first", "second", "top"]


def test_worlds_keep_their_clips_apart() -> None:
    hud = World2D("hud")
    timeline = Timeline()
    timeline.add(prepared(moving), name="stage")
    timeline.add(prepared(moving), name="label", world=hud)

    assert timeline.worlds[1] is hud
    assert clip_names(timeline.state_at(0.0).children, timeline) == ["stage"]
    assert clip_names(timeline.state_at(0.0, world=hud).children, timeline) == ["label"]


def test_2d_worlds_only_allow_placement_in_their_plane() -> None:
    hud = World2D("hud")
    timeline = Timeline()
    clip = timeline.add(prepared(moving), world=hud)

    clip.place(position=(3, -2, 0), rotation=quaternion_from_axis_angle((0, 0, 1), np.pi / 4))
    with pytest.raises(ValueError, match="z = 0 plane"):
        clip.place(position=(0, 0, 1))
    with pytest.raises(ValueError, match="around the z axis"):
        clip.place(rotation=quaternion_from_axis_angle((1, 0, 0), 0.3))
    with pytest.raises(ValueError, match="z = 0 plane"):
        timeline.add(prepared(moving), world=hud, transform=Transform(position=(0, 0, 2)))
    assert clip.transform.position == (3.0, -2.0, 0.0)  # rejected edits change nothing


def test_a_prepared_scene_backs_only_one_clip() -> None:
    scene = prepared(moving)
    timeline = Timeline()
    timeline.add(scene)

    with pytest.raises(ValueError, match="prepare the scene again"):
        timeline.add(scene)


def test_stills_ignore_start_cut_and_time_map() -> None:
    timeline = Timeline(events={"next": 9.0})
    clip = timeline.add(prepared(moving), start=5.0).cut(1.0, 3.5).slow_motion(1.0, 2.0, 4.0)
    clip.place(position=(1, 0, 0))

    still = clip.still(3.0)  # playing time 3: one second into the second segment
    assert dot_x(timeline, still) == pytest.approx(3.0)
    assert still.uid == clip.uid and still.transform.position == (1.0, 0.0, 0.0)
    assert dot_x(timeline, clip.still(2.0)) == pytest.approx(2.0)  # the boundary shows the first segment's end
    assert dot_x(timeline, clip.still(99.0)) == pytest.approx(4.0)  # clamped to the end


def test_compiled_time_at_inverts_playing_time() -> None:
    scene = prepared(moving)

    assert scene.compiled_time_at(1.0) == 1.0
    assert scene.compiled_time_at(2.0) == 2.0  # end of the first segment, not the start of the second
    assert scene.compiled_time_at(3.0) == 3.0 + SEGMENT_GAP
    assert scene.compiled_time_at(-1.0) == 0.0 and scene.compiled_time_at(10.0) == scene.duration
    for t in (0.5, 2.0, 3.5):
        assert scene.playing_time_at(scene.compiled_time_at(t)) == pytest.approx(t)
