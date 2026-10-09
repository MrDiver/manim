"""Scenes: authoring order, direct edits, presence, groups, derived signals,
and the purity of state_at (sections 4 and 5 of the design doc)."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from manimx.robjects import RShape2D
from manimx.animation import Animation, Change, Group, Scene, Target, derive
from manimx.userland import DOWN, LEFT, ORIGIN, RIGHT, TAU, UP, Circle, Dot, FadeIn, FadeOut, Indicate, Rectangle, RegularPolygon, Rotate, Square, Triangle
from manimx.userland.rate_functions import linear
from naming import by_name


def prepare(construct):
    """Prepare a scene whose construct() is the given function."""

    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def objects(prepared, t):
    return by_name(prepared.state_at(t).children, prepared.roots)


def position(prepared, t, name):
    return objects(prepared, t)[name].transform.position


def test_play_places_animations_and_moves_the_cursor() -> None:
    def construct(scene):
        square = Square(name="square")
        scene.add(square)
        scene.play(square.animate.shift(RIGHT * 2), run_time=2, rate=linear)
        scene.wait(1)
        scene.play(square.animate.shift(UP), rate=linear)

    prepared = prepare(construct)

    assert prepared.duration == 4.0
    assert position(prepared, 1.0, "square") == (1.0, 0.0, 0.0)
    assert position(prepared, 2.5, "square") == (2.0, 0.0, 0.0)  # waiting
    assert position(prepared, 3.5, "square") == (2.0, 0.5, 0.0)


def test_run_time_of_play_is_the_animations_run_time() -> None:
    @dataclass(eq=False)
    class Seconds(Animation):
        """Moves its node to x = the seconds since it started."""

        node: Square
        run_time: float = 1.0
        rate: Any = staticmethod(linear)

        def targets(self):
            return [Target(self.node.position.x, Change.ABSOLUTE)]

        def contribution(self, signal, alpha, start_value):
            return alpha * self.run_time

    def construct(scene):
        square = Square(name="square")
        scene.add(square)
        scene.play(Seconds(square), run_time=4)

    prepared = prepare(construct)

    assert prepared.duration == 4.0
    assert position(prepared, 3.0, "square") == (3.0, 0.0, 0.0)


def test_axes_animate_independently_and_absolute_overrides() -> None:
    def construct(scene):
        dot = Dot(name="dot")
        scene.add(dot)
        scene.play(dot.animate.shift(RIGHT * 4), dot.animate.shift(DOWN * 2), run_time=2, rate=linear)
        scene.play(dot.animate.move_to(ORIGIN), rate=linear)

    prepared = prepare(construct)

    assert position(prepared, 1.0, "dot") == (2.0, -1.0, 0.0)
    assert position(prepared, 2.5, "dot") == (2.0, -1.0, 0.0)  # halfway back
    assert position(prepared, 3.0, "dot") == (0.0, 0.0, 0.0)


def test_direct_edits_do_not_rewrite_the_past() -> None:
    def construct(scene):
        square = Square(name="square")
        square.shift(LEFT)  # before add: part of the setup (the base value)
        scene.add(square)
        scene.wait(1)
        square.shift(UP)  # on the timeline: happens at t = 1
        scene.wait(1)

    prepared = prepare(construct)

    assert position(prepared, 0.5, "square") == (-1.0, 0.0, 0.0)
    assert position(prepared, 1.0, "square") == (-1.0, 1.0, 0.0)


def test_authoring_reads_see_everything_recorded_so_far() -> None:
    seen = {}

    def construct(scene):
        a, b = Dot(name="a"), Dot(name="b")
        scene.add(a, b)
        scene.play(a.animate.shift(RIGHT * 3))
        seen["center"] = a.get_center()
        scene.play(b.animate.move_to(a.get_center() + UP))

    prepared = prepare(construct)

    np.testing.assert_allclose(seen["center"], [3, 0, 0])
    assert position(prepared, 2.0, "b") == (3.0, 1.0, 0.0)


def test_presence_follows_add_remove_and_fades() -> None:
    def construct(scene):
        square = Square(name="square", fill_opacity=1.0)
        circle = Circle(name="circle")
        scene.add(square)
        scene.wait(1)
        scene.play(FadeIn(circle), rate=linear)  # adds the circle at t = 1
        scene.play(FadeOut(square), rate=linear)  # removes the square at t = 3
        scene.wait(1)

    prepared = prepare(construct)

    assert list(objects(prepared, 0.5)) == ["square"]
    assert list(objects(prepared, 1.5)) == ["square", "circle"]
    assert objects(prepared, 1.5)["circle"].stroke.opacity[0] == pytest.approx(0.5)
    assert objects(prepared, 2.5)["square"].fill.opacity[0] == pytest.approx(0.5)
    assert list(objects(prepared, 3.5)) == ["circle"]


def test_lagged_fade_in_hides_mobjects_until_their_own_start() -> None:
    def construct(scene):
        dots = [Dot(name=f"dot-{i}") for i in range(3)]
        scene.play(*(FadeIn(d) for d in dots), lag_ratio=0.5, run_time=1, rate=linear)

    prepared = prepare(construct)

    assert list(objects(prepared, 0.25)) == ["dot-0"]
    assert list(objects(prepared, 1.0)) == ["dot-0", "dot-1", "dot-2"]
    assert objects(prepared, 1.0)["dot-2"].fill.opacity[0] == pytest.approx(0.0)
    assert prepared.duration == 2.0


def test_fade_with_shift_and_removal_by_hand() -> None:
    def construct(scene):
        dot = Dot(name="dot")
        scene.play(FadeIn(dot, shift=UP), rate=linear)
        scene.play(FadeOut(dot, shift=RIGHT), rate=linear)
        scene.add(dot)
        scene.remove(dot)
        scene.remove(dot)  # removing twice is harmless

    prepared = prepare(construct)

    assert position(prepared, 0.5, "dot") == (0.0, -0.5, 0.0)
    assert position(prepared, 1.5, "dot") == (0.5, 0.0, 0.0)
    assert objects(prepared, 2.0) == {}


def test_rotate_sweeps_full_turns_and_indicate_returns() -> None:
    def construct(scene):
        square = Square(name="square")
        scene.play(Rotate(square, TAU), rate=linear)
        scene.play(Indicate(square))

    prepared = prepare(construct)

    half = objects(prepared, 0.5)["square"].transform.rotation
    assert abs(half.w) == pytest.approx(0.0, abs=1e-9)  # half a turn
    assert objects(prepared, 1.5)["square"].transform.scale[0] == pytest.approx(1.2)
    assert objects(prepared, 2.0)["square"].transform.scale[0] == pytest.approx(1.0)


def test_style_changes_and_animate_rotate_and_scale() -> None:
    def construct(scene):
        square = Square(name="square", fill_opacity=1.0)
        scene.add(square)
        scene.play(square.animate.set_color("#FF0000").scale_by(2).rotate(math.pi / 2).set_stroke(width=8, opacity=0.5), rate=linear)
        scene.play(square.animate.set_fill(opacity=0.25).set_alpha(0.5), rate=linear)

    prepared = prepare(construct)
    done = objects(prepared, 1.0)["square"]

    np.testing.assert_allclose(done.fill.color[0], [1, 0, 0])
    assert done.transform.scale == (2.0, 2.0, 2.0)
    assert done.stroke.width[0] == 8.0 and done.stroke.opacity[0] == 0.5
    faded = objects(prepared, 2.0)["square"]
    assert faded.fill.opacity[0] == pytest.approx(0.125)


def test_groups_move_their_children_and_pass_alpha_on() -> None:
    def construct(scene):
        a, b = Square(name="a", fill_opacity=1.0), Circle(name="b", fill_opacity=1.0)
        group = Group(a, name="group").add(b)
        scene.play(FadeIn(a))  # a is not added on its own: it joins the scene...
        scene.add(group)  # ...and the group joins too
        scene.play(group.animate.shift(RIGHT).set_alpha(0.5), rate=linear)

    prepared = prepare(construct)
    state = objects(prepared, 2.0)

    assert set(objects(prepared, 0.5)) == {"a"}  # on its own until the group arrives
    assert set(state) == {"group"}  # then drawn once, through the group
    group = state["group"]
    assert group.transform.position == (1.0, 0.0, 0.0)
    assert list(by_name(group.children, prepared.roots)) == ["a", "b"]
    assert group.children[1].fill.opacity[0] == pytest.approx(0.5)


def test_derived_signals_follow_animated_ones() -> None:
    def construct(scene):
        dot = Dot(name="dot")
        label = Dot(name="label").follow(dot, offset=UP * 0.5)
        size = derive(lambda t: 1.0 + t, scene.time.scene_time)
        label.stroke_width = size  # a derived style signal, driven by time
        scene.add(dot, label)
        scene.play(dot.animate.shift(RIGHT * 2), rate=linear)

    prepared = prepare(construct)

    assert position(prepared, 0.5, "label") == (1.0, 0.5, 0.0)
    assert objects(prepared, 0.5)["label"].stroke.width[0] == 1.5


def test_derived_signals_cannot_be_animated_or_set() -> None:
    def animate_derived(scene):
        dot, label = Dot(), Dot()
        label.follow(dot)
        scene.play(label.animate.shift(RIGHT))

    def set_derived(scene):
        dot, label = Dot(), Dot()
        label.follow(dot)
        label.shift(RIGHT)

    with pytest.raises(TypeError, match="derived"):
        prepare(animate_derived)
    with pytest.raises(TypeError, match="derived"):
        prepare(set_derived)


def test_state_at_is_pure_and_order_independent() -> None:
    def construct(scene):
        square = Square(name="square", fill_opacity=1.0)
        scene.play(FadeIn(square))
        scene.play(square.animate.shift(RIGHT * 3).rotate(1.0).set_color("#00FF00"), run_time=2)
        scene.play(square.animate.move_to(UP), FadeOut(square))

    prepared = prepare(construct)
    times = np.linspace(-0.5, 5.0, 23)

    def signature(t):
        state = prepared.state_at(t)
        return [(c.uid, c.transform.position, tuple(np.round(c.fill.color[0], 9)), float(c.fill.opacity[0])) for c in state.children]

    forward = [signature(t) for t in times]
    shuffled = {t: signature(t) for t in np.random.default_rng(7).permutation(times)}
    assert forward == [shuffled[t] for t in times]


def test_snapshots_keep_identity_and_cache_versions() -> None:
    def construct(scene):
        square = Square(name="square", fill_opacity=1.0)
        scene.add(square)
        scene.play(square.animate.shift(RIGHT))

    prepared = prepare(construct)
    first, second = objects(prepared, 0.2)["square"], objects(prepared, 0.7)["square"]

    assert isinstance(first, RShape2D) and first.uid == second.uid
    assert first.version("points", "fill") == second.version("points", "fill")  # only moved
    assert first.transform.position != second.transform.position


def test_prepare_runs_construct_once_and_api_needs_construct() -> None:
    calls = []

    class Counted(Scene):
        def construct(self):
            calls.append(1)
            self.add(Square())

    scene = Counted()
    assert scene.prepare() is scene.prepare()
    assert calls == [1]
    with pytest.raises(RuntimeError, match="inside construct"):
        scene.add(Square())
    with pytest.raises(ValueError, match="at least one"):
        prepare(lambda s: s.play())
    with pytest.raises(ValueError, match="negative"):
        prepare(lambda s: s.wait(-1))
    with pytest.raises(AttributeError, match="cannot be animated"):
        Square().animate.get_center()


def test_builder_overrides_and_shapes() -> None:
    def construct(scene):
        shapes = [Rectangle(name="rect"), RegularPolygon(5, name="pentagon"), Triangle(name="triangle")]
        scene.add(*shapes)
        scene.play(shapes[0].animate.shift(RIGHT).shift(RIGHT).set_run_time(4).set_rate(linear))
        scene.play(shapes[1].animate.move_to(UP).shift(RIGHT), rate=linear)

    prepared = prepare(construct)

    assert prepared.duration == 5.0
    assert position(prepared, 2.0, "rect") == (1.0, 0.0, 0.0)
    assert position(prepared, 5.0, "pentagon") == (1.0, 1.0, 0.0)
    assert repr(Square(name="s")) == "Square('s')"
    prepared.refresh()  # nothing moved: values unchanged
    assert position(prepared, 5.0, "pentagon") == (1.0, 1.0, 0.0)


def test_grow_from_center_and_shrink_to_center() -> None:
    from manimx.userland import GrowFromCenter, ShrinkToCenter

    def construct(scene):
        square = Square(name="square", fill_opacity=1.0)
        square.set_scale(2.0)
        scene.play(GrowFromCenter(square), rate=linear)
        scene.play(ShrinkToCenter(square), rate=linear)

    prepared = prepare(construct)

    assert objects(prepared, 0.0) == {}  # scale 0: nothing to draw
    assert objects(prepared, 0.5)["square"].transform.scale == (1.0, 1.0, 1.0)
    assert objects(prepared, 1.0)["square"].transform.scale == (2.0, 2.0, 2.0)
    assert objects(prepared, 1.5)["square"].transform.scale == (1.0, 1.0, 1.0)
    assert objects(prepared, 2.0) == {}  # shrunk and removed


def test_clipping_groups_snapshot_to_render_clipping_groups() -> None:
    from manimx.robjects import RClippingGroup
    from manimx.animation import ClippingGroup
    from manimx.userland import Polyline

    def construct(scene):
        mask = Circle(name="mask", fill_opacity=1.0)
        lines = Polyline([(-2, 0), (2, 0)], name="line")
        clip = ClippingGroup(mask, lines, name="clip", one_sided=True)
        scene.add(clip)
        scene.play(clip.animate.shift(UP), rate=linear)
        scene.play(mask.animate.set_alpha(0.0), rate=linear)

    prepared = prepare(construct)
    clip = objects(prepared, 1.0)["clip"]

    assert isinstance(clip, RClippingGroup) and clip.one_sided
    assert list(by_name([clip.mask, clip.content], prepared.roots)) == ["mask", "line"]
    assert clip.transform.position == (0.0, 1.0, 0.0)
    assert prepared.state_at(2.0).children[0].mask.fill.opacity[0] == 0.0  # mask faded: content hidden by the renderer


def test_clipping_group_without_visible_parts_is_skipped() -> None:
    from manimx.animation import ClippingGroup
    from manimx.userland import GrowFromCenter

    def construct(scene):
        content = Square(name="content")
        clip = ClippingGroup(Circle(name="mask"), content, name="clip")
        scene.play(GrowFromCenter(content), rate=linear)  # content collapsed at t = 0
        scene.add(clip)

    prepared = prepare(construct)

    assert "clip" not in objects(prepared, 0.0) and "clip" in objects(prepared, 1.0)
