"""Time maps: smooth monotone retiming, cues, slow motion and cuts
(section 6.4 of the design doc)."""

import numpy as np
import pytest

from manimx.animation import InvalidCue, MonotoneMap, Ramp, Scene, Timeline, UnresolvedEvents
from manimx.userland import RIGHT, Dot
from manimx.userland.rate_functions import linear
from naming import by_name, timeline_roots


def prepared(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def x_at(timeline, t, name="dot"):
    group = timeline.state_at(t).children[0]
    return by_name(group.children, timeline_roots(timeline))[name].transform.position[0]


def counting(scene):
    """A dot that moves 1 unit per second for 6 s, with cues at 1, 3 and 5 s
    ("one", "two", "three"): x equals the scene's playing time."""
    dot = Dot(name="dot")
    scene.add(dot)
    for cue, wait in (("one", 1), ("two", 2), ("three", 2)):
        scene.play(dot.animate.shift(RIGHT * wait), run_time=wait, rate=linear)
        scene.cue(cue)
    scene.play(dot.animate.shift(RIGHT), run_time=1, rate=linear)


# --- MonotoneMap --------------------------------------------------------------------


def test_monotone_map_passes_through_knots_and_never_runs_backwards() -> None:
    curve = MonotoneMap([0, 2, 3, 6, 7], [0, 1, 3, 4, 4.5])
    xs = np.linspace(-1, 8, 901)
    ys = np.array([curve(x) for x in xs])

    assert [curve(x) for x in curve.x] == pytest.approx(list(curve.y))
    assert np.all(np.diff(ys) >= -1e-12)
    assert curve(-1) == 0.0 and curve(8) == 4.5  # clamped outside
    inside = np.linspace(0, 7, 701)
    values = np.array([curve(x) for x in inside])
    speed = np.diff(values) / np.diff(inside)
    assert np.abs(np.diff(speed)).max() < 0.1  # no jumps in speed between knots
    assert curve.inverse(curve(2.5)) == pytest.approx(2.5)
    assert curve.inverse(-5) == 0.0 and curve.inverse(99) == 7.0
    assert (curve.start, curve.end) == (0.0, 7.0)


def test_monotone_map_handles_flat_stretches_and_two_knots() -> None:
    flat = MonotoneMap([0, 1, 2, 3], [0, 1, 1, 2])  # a hold in the middle
    assert flat(1.5) == pytest.approx(1.0)
    line = MonotoneMap([0, 4], [0, 2])
    assert line(1.0) == 0.5


def test_monotone_map_rejects_bad_knots() -> None:
    with pytest.raises(ValueError, match="at least two"):
        MonotoneMap([0], [0])
    with pytest.raises(ValueError, match="backwards"):
        MonotoneMap([0, 1, 1], [0, 1, 2])
    with pytest.raises(ValueError, match="backwards"):
        MonotoneMap([0, 1, 2], [0, 2, 1])


# --- cues ---------------------------------------------------------------------------


def test_cues_land_exactly_on_their_events() -> None:
    timeline = Timeline(events={"one": 2.0, "two": 3.0, "three": 7.0})
    timeline.add(prepared(counting))

    assert x_at(timeline, 0.0) == pytest.approx(0.0)
    assert x_at(timeline, 2.0) == pytest.approx(1.0)  # "one" is said at 2 s
    assert x_at(timeline, 3.0) == pytest.approx(3.0)  # "two" at 3 s: it hurries
    assert x_at(timeline, 7.0) == pytest.approx(5.0)  # "three" at 7 s: it slows down
    assert timeline.duration == pytest.approx(8.0)  # the tail plays at normal speed


def test_cue_retiming_is_smooth_and_monotone() -> None:
    timeline = Timeline(events={"one": 2.0, "two": 3.0, "three": 7.0})
    timeline.add(prepared(counting))
    times = np.linspace(0, 8, 801)
    xs = np.array([x_at(timeline, t) for t in times])

    assert np.all(np.diff(xs) >= -1e-9)
    speed = np.diff(xs) / np.diff(times)
    assert np.abs(np.diff(speed)).max() < 0.2


def test_cues_can_be_bound_to_other_events_and_unknown_ones_are_free() -> None:
    timeline = Timeline(events={"word:square": 4.0})
    clip = timeline.add(prepared(counting))
    clip.bind_cue("two", "word:square")

    assert x_at(timeline, 4.0) == pytest.approx(3.0)
    assert clip.event_of("two") == "word:square" and clip.event_of("one") == "one"


def test_backwards_cues_are_skipped_while_editing_and_rejected_strictly() -> None:
    timeline = Timeline(events={"one": 3.0, "two": 2.0, "three": 9.0})  # "two" before "one"
    timeline.add(prepared(counting))

    assert x_at(timeline, 3.0) == pytest.approx(1.0)  # lenient: "two" is ignored
    with pytest.raises(InvalidCue, match="backwards"):
        timeline.resolve()


def test_cues_from_emits_of_other_clips() -> None:
    def announcer(scene):
        scene.add(Dot(name="dot"))
        scene.wait(4)
        scene.emit("two")

    timeline = Timeline()
    timeline.add(prepared(counting))
    timeline.add(prepared(announcer))

    assert timeline.resolve()["two"] == 4.0
    assert x_at(timeline, 4.0) == pytest.approx(3.0)


# --- slow motion and scales ----------------------------------------------------------


def test_slow_motion_stretches_a_range() -> None:
    timeline = Timeline()
    clip = timeline.add(prepared(counting)).slow_motion(2.0, 4.0, factor=3.0)

    assert timeline.duration == pytest.approx(10.0)  # 2 s become 6 s
    assert x_at(timeline, 2.0) == pytest.approx(2.0)
    assert x_at(timeline, 8.0) == pytest.approx(4.0)
    assert clip.ramps == [Ramp(2.0, 4.0, 3.0)]
    with pytest.raises(ValueError, match="positive"):
        Ramp(1.0, 2.0, 0.0)


def test_segment_scale_can_be_changed_after_adding() -> None:
    timeline = Timeline()
    clip = timeline.add(prepared(counting))
    assert timeline.duration == pytest.approx(6.0)

    clip.set_scale(0, 2.0)  # marks the timeline for re-resolution
    assert timeline.duration == pytest.approx(12.0)
    with pytest.raises(ValueError, match="positive"):
        clip.set_scale(0, 0.0)


# --- cuts ---------------------------------------------------------------------------


def test_cut_trims_the_playing_time() -> None:
    timeline = Timeline()
    clip = timeline.add(prepared(counting), start=10.0).cut(2.0, 5.0)

    assert x_at(timeline, 10.0) == pytest.approx(2.0)  # starts at playing time 2
    assert timeline.duration == pytest.approx(13.0)
    assert timeline.state_at(13.5).children == []
    assert (clip.cut_in, clip.cut_out) == (2.0, 5.0)
    with pytest.raises(ValueError, match="start < end"):
        clip.cut(3.0, 3.0)


def two_parts(scene):
    dot = Dot(name="dot")
    scene.add(dot)
    scene.play(dot.animate.shift(RIGHT * 2), run_time=2, rate=linear)
    scene.emit("first-done")
    scene.wait_for("next")
    scene.play(dot.animate.shift(RIGHT * 2), run_time=2, rate=linear)
    scene.emit("second-done")


def test_cut_skips_whole_segments_and_their_waits_and_emits() -> None:
    timeline = Timeline()
    timeline.add(prepared(two_parts)).cut(start=2.5)  # half a second into the second segment

    events = timeline.resolve()  # "next" is not needed any more
    assert "first-done" not in events
    assert events["second-done"] == pytest.approx(1.5)  # 2 s into the segment, which starts 0.5 s in
    assert x_at(timeline, 1.0) == pytest.approx(3.5)


def test_cut_ending_in_the_first_segment_needs_no_event() -> None:
    timeline = Timeline()
    timeline.add(prepared(two_parts)).cut(end=1.0)

    timeline.resolve()
    assert timeline.duration == pytest.approx(1.0)


def test_strict_mode_still_needs_events_inside_the_cut() -> None:
    timeline = Timeline()
    timeline.add(prepared(two_parts))

    with pytest.raises(UnresolvedEvents, match="next"):
        timeline.resolve()


def test_segments_listing_and_zero_length_segments() -> None:
    def instant_first(scene):
        scene.wait_for("go")  # the first segment is empty
        dot = Dot(name="dot")
        scene.add(dot)
        scene.wait(1)

    timeline = Timeline(events={"go": 2.0})
    timeline.add(prepared(instant_first), name="clip")

    assert [(index, start, end) for _, index, start, end in timeline.segments()] == [(0, 0.0, 0.0), (1, 2.0, 3.0)]
    assert timeline.state_at(1.0).children[0].children == []  # holding before the dot exists
