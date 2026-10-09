"""Segments, events and the global timeline (section 6 of the design doc)."""

import pytest

from manimx.animation import SEGMENT_GAP, Scene, Timeline, UnresolvedEvents, derive
from manimx.userland import RIGHT, UP, Dot, FadeIn
from manimx.userland.rate_functions import linear
from naming import by_name, timeline_roots


def prepared(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def x_of(timeline, state, clip=0, name="dot"):
    group = state.children[clip]
    return by_name(group.children, timeline_roots(timeline))[name].transform.position[0]


def two_part(scene):
    """Moves right for 2 s, waits for "next", moves right again for 2 s."""
    dot = Dot(name="dot")
    scene.add(dot)
    scene.play(dot.animate.shift(RIGHT * 2), run_time=2, rate=linear)
    scene.wait_for("next")
    scene.play(dot.animate.shift(RIGHT * 2), run_time=2, rate=linear)


# --- compiled layout --------------------------------------------------------------


def test_segments_are_laid_out_with_gaps() -> None:
    scene = prepared(two_part)

    first, second = scene.segments
    assert (first.offset, first.duration, first.waits_for) == (0.0, 2.0, None)
    assert (second.offset, second.duration, second.waits_for) == (2.0 + SEGMENT_GAP, 2.0, "next")
    assert second.play_offset == 2.0 and scene.playing_time == 4.0
    assert scene.duration == second.end
    # Playing time skips the gap: scene_time does not jump at the boundary.
    assert scene.playing_time_at(2.5) == 2.0 and scene.playing_time_at(second.offset + 1.0) == 3.0


def test_segment_duration_includes_running_animations() -> None:
    def construct(scene):
        dot = Dot(name="dot")
        scene.add(dot)
        scene.play(dot.animate.shift(RIGHT), run_time=3)
        scene.wait_for("go")

    assert prepared(construct).segments[0].duration == 3.0


def test_emits_are_recorded_in_segment_time() -> None:
    def construct(scene):
        scene.add(Dot())
        scene.wait(1)
        scene.wait_for("a")
        scene.wait(0.5)
        scene.emit("b")

    scene = prepared(construct)
    assert scene.segments[1].emits[0].event == "b" and scene.segments[1].emits[0].time == 0.5


# --- holds and retiming -----------------------------------------------------------


def test_holds_until_the_event_fires() -> None:
    timeline = Timeline()
    timeline.fire("next", 5.0)
    timeline.add(prepared(two_part))

    assert x_of(timeline, timeline.state_at(1.0)) == 1.0
    assert x_of(timeline, timeline.state_at(3.0)) == 2.0  # holding
    assert x_of(timeline, timeline.state_at(4.99)) == 2.0
    assert x_of(timeline, timeline.state_at(6.0)) == 3.0
    assert timeline.duration == 7.0
    assert timeline.state_at(7.5).children == []  # the clip has ended


def test_an_early_event_waits_for_the_previous_segment() -> None:
    timeline = Timeline(events={"next": 0.5})
    clip = timeline.add(prepared(two_part))

    assert clip.segment_starts == [0.0, 2.0]  # starts when segment 0 ends


def test_scaling_a_segment_stretches_it() -> None:
    timeline = Timeline(events={"next": 2.0})
    timeline.add(prepared(two_part), scales={1: 2.0})

    assert x_of(timeline, timeline.state_at(4.0)) == 3.0  # half speed: 2 s in, half done
    assert timeline.duration == 6.0


def test_clip_start_offsets_everything() -> None:
    timeline = Timeline(events={"next": 10.0})
    timeline.add(prepared(two_part), start=3.0)

    assert timeline.state_at(2.0).children == []  # not started yet
    assert x_of(timeline, timeline.state_at(4.0)) == 1.0


def test_hold_shows_the_end_of_the_previous_segment_only() -> None:
    """Nothing that happens at the very start of the next segment leaks into a hold."""

    def construct(scene):
        scene.add(Dot(name="dot"))
        scene.wait(1)
        scene.wait_for("go")
        scene.add(Dot(name="late"))  # added at the start of segment 1

    timeline = Timeline(events={"go": 5.0})
    timeline.add(prepared(construct))

    holding = set(by_name(timeline.state_at(3.0).children[0].children, timeline_roots(timeline)))
    after = set(by_name(timeline.state_at(5.0).children[0].children, timeline_roots(timeline)))
    assert holding == {"dot"} and after == {"dot", "late"}


def test_scene_time_freezes_during_holds_and_global_time_runs() -> None:
    def construct(scene):
        scene_driven = Dot(name="scene")
        global_driven = Dot(name="global")
        scene_driven.position.x = derive(lambda t: t, scene.time.scene_time)
        global_driven.position.x = derive(lambda t: t, scene.time.global_time)
        scene.add(scene_driven, global_driven)
        scene.wait(1)
        scene.wait_for("go")
        scene.wait(1)

    timeline = Timeline(events={"go": 4.0})
    timeline.add(prepared(construct))

    holding = timeline.state_at(3.0)
    assert x_of(timeline, holding, name="scene") == 1.0 and x_of(timeline, holding, name="global") == 3.0
    assert x_of(timeline, timeline.state_at(4.5), name="scene") == 1.5


# --- events across clips ----------------------------------------------------------


def test_emits_trigger_other_clips() -> None:
    def leader(scene):
        scene.add(Dot(name="dot"))
        scene.wait(1.5)
        scene.emit("cue")
        scene.wait(1)

    def follower(scene):
        dot = Dot(name="dot")
        scene.add(dot)
        scene.wait_for("cue")
        scene.play(dot.animate.shift(UP), run_time=1)

    timeline = Timeline()
    timeline.add(prepared(follower))  # added first, resolved anyway
    timeline.add(prepared(leader), start=2.0)

    assert timeline.resolve()["cue"] == 3.5
    assert timeline.clips[0].segment_starts == [0.0, 3.5]


def test_emit_chains_resolve_in_dependency_order() -> None:
    def relay(listen, say):
        def construct(scene):
            scene.add(Dot())
            if listen:
                scene.wait_for(listen)
            scene.wait(1)
            scene.emit(say)

        return construct

    timeline = Timeline()
    for listen, say in (("b", "c"), ("a", "b"), (None, "a")):  # worst order
        timeline.add(prepared(relay(listen, say)))

    assert timeline.resolve() == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_missing_events_and_cycles_are_errors_offline() -> None:
    lenient = Timeline()
    lenient.add(prepared(two_part))  # adding never fails: the event may come later
    with pytest.raises(UnresolvedEvents, match="next"):
        lenient.resolve()

    def waits_then_says(listen, say):
        def construct(scene):
            scene.add(Dot())
            scene.wait_for(listen)
            scene.emit(say)

        return construct

    timeline = Timeline()
    timeline.add(prepared(waits_then_says("x", "y")))
    timeline.add(prepared(waits_then_says("y", "x")))
    with pytest.raises(UnresolvedEvents, match=r"\['x', 'y'\]"):
        timeline.resolve()


def test_live_mode_holds_until_a_presenter_fires() -> None:
    timeline = Timeline(live=True)
    clip = timeline.add(prepared(two_part))

    assert x_of(timeline, timeline.state_at(100.0)) == 2.0  # holds indefinitely
    timeline.fire("next", 101.0)  # the presenter clicks
    assert x_of(timeline, timeline.state_at(102.0)) == 3.0
    assert clip.end == 103.0


def test_lists_segments_for_display() -> None:
    timeline = Timeline(events={"next": 4.0})
    timeline.add(prepared(two_part), name="intro")

    assert [(c.name, i, s, e) for c, i, s, e in timeline.segments()] == [("intro", 0, 0.0, 2.0), ("intro", 1, 4.0, 6.0)]


def test_fade_in_after_an_event() -> None:
    def construct(scene):
        scene.wait_for("show")
        scene.play(FadeIn(Dot(name="dot")), rate=linear)

    timeline = Timeline(events={"show": 2.0})
    timeline.add(prepared(construct))

    assert timeline.state_at(1.0).children[0].children == []
    assert timeline.state_at(2.5).children[0].children[0].fill.opacity[0] == pytest.approx(0.5)
