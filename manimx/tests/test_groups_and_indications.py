"""AnimationGroup, Succession, LaggedStart, the Grow family, FadeTransform,
letter by letter, Homotopy and the indications."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import (
    AnimationGroup,
    LaggedStart,
    RootSignal,
    Scene,
    Shape2D,
    Succession,
)
from manimx.userland import (
    DOWN,
    RIGHT,
    UP,
    AddTextLetterByLetter,
    ApplyWave,
    Arrow,
    Circle,
    Circumscribe,
    FadeIn,
    FadeTransform,
    Flash,
    GrowArrow,
    GrowFromEdge,
    GrowFromPoint,
    Homotopy,
    RemoveTextLetterByLetter,
    ShowPassingFlash,
    SpinInFromNothing,
    Square,
    Typst,
    Wiggle,
    current_bounds,
)
from manimx.userland.indications import _PassingFlash


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def test_groups_schedule_their_animations_and_nest() -> None:
    a, b, c = Square(), Square(), Square()
    succession = Succession(a.animate.shift(RIGHT), b.animate.shift(RIGHT).set_run_time(2.0))
    assert [(start, length) for _, start, length in succession.schedule()] == [(0.0, 1.0), (1.0, 2.0)]
    assert succession.run_time == 3.0
    lagged = LaggedStart(a.animate.shift(UP), b.animate.shift(UP), lag_ratio=0.5)
    assert [start for _, start, _ in lagged.schedule()] == [0.0, 0.5]
    nested = AnimationGroup(succession, c.animate.shift(UP), run_time=6.0)
    assert [(start, length) for _, start, length in nested.schedule()] == [(0.0, 2.0), (2.0, 4.0), (0.0, 2.0)]
    with pytest.raises(ValueError, match="at least one"):
        AnimationGroup()

    prepared = prepare(lambda s: (s.add(a, b), s.play(succession), s.play(lagged, run_time=3.0)))
    prepared.evaluate(1.0)
    assert a.position.x.value == pytest.approx(1.0) and b.position.x.value == pytest.approx(0.0)
    prepared.evaluate(3.0)
    assert b.position.x.value == pytest.approx(1.0)
    prepared.evaluate(4.0)  # the lagged start fitted to 3 s: b starts 1 s in
    assert a.position.y.value > 0.0 and b.position.y.value == pytest.approx(0.0)
    prepared.evaluate(6.0)
    assert a.position.y.value == pytest.approx(1.0) and b.position.y.value == pytest.approx(1.0)


def test_growing_from_a_point_scales_about_it() -> None:
    square = Square(2.0).move_to((1.0, 0.0, 0.0))
    grow = GrowFromPoint(square, (3.0, 0.0, 0.0))
    prepared = prepare(lambda s: s.play(grow, rate=lambda t: t))
    prepared.evaluate(0.5)
    # Halfway: half the size, still reaching from the point.
    assert square.scale.x.value == pytest.approx(0.5) and square.position.x.value == pytest.approx(2.0)
    edge = GrowFromEdge(Square(2.0), DOWN)
    assert edge._start_point()[1] == pytest.approx(-1.0)


def test_grow_arrow_and_spin_in() -> None:
    arrow = Arrow((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    square = Square()
    prepared = prepare(lambda s: s.play(GrowArrow(arrow), SpinInFromNothing(square, angle=math.pi)))
    prepared.evaluate(0.0)
    assert arrow.end.x.value == pytest.approx(0.0)
    prepared.evaluate(1.0)
    assert arrow.end.x.value == pytest.approx(2.0) and abs(square.rotation.value.w) == pytest.approx(1.0)
    with pytest.raises(TypeError, match="start and an end"):
        GrowArrow(Square())


def test_fade_transform_moves_each_side_onto_the_other() -> None:
    small = Square(1.0).move_to((-2.0, 0.0, 0.0))
    big = Circle(1.0).move_to((2.0, 0.0, 0.0))
    fade = FadeTransform(small, big)
    prepared = prepare(lambda s: (s.add(small), s.play(fade)))
    prepared.evaluate(0.0)
    np.testing.assert_allclose(current_bounds(big).mean(axis=0)[:2], [-2.0, 0.0], atol=1e-9)  # starts on the square
    assert big.alpha.value == 0.0
    prepared.evaluate(1.0)
    assert not small.present.value and big.alpha.value == 1.0
    np.testing.assert_allclose(small.scale.x.value, 2.0)  # the square stretched onto the circle
    even = FadeTransform(Square(1.0), Square(1.0).scale_by((2.0, 3.0, 1.0)), stretch=False)
    assert even is not None


def test_letters_appear_and_go_one_by_one() -> None:
    text = Typst("abc")
    add = AddTextLetterByLetter(text, time_per_char=0.2)
    assert add.run_time == pytest.approx(0.6)
    prepared = prepare(lambda s: (s.play(add), s.play(RemoveTextLetterByLetter(text, run_time=0.6))))
    prepared.evaluate(0.25)
    assert [shape.alpha.value for shape in text.shapes] == [1.0, 1.0, 0.0]
    prepared.evaluate(0.6 + 0.25)
    assert [shape.alpha.value for shape in text.shapes] == [1.0, 0.0, 0.0]
    prepared.evaluate(1.3)
    assert not text.present.value


def test_homotopy_moves_points_through_a_function() -> None:
    text = Typst("x")
    start = text.shapes[0].points.value.copy()
    prepared = prepare(lambda s: s.play(Homotopy(lambda x, y, z, t: (x + t, y, z), text, run_time=1.0)))
    prepared.evaluate(1.0)
    np.testing.assert_allclose(text.shapes[0].points.value, start + [1.0, 0.0])
    circle = Circle(1.0).move_to((2.0, 0.0, 0.0))
    bend = Homotopy(lambda x, y, z, t: (x, y + t, z), circle, run_time=1.0)
    plain = bend.unpacked[circle.uid]
    assert isinstance(plain.points, RootSignal) and plain.position.x.value == pytest.approx(2.0)
    np.testing.assert_allclose(plain.points.value, circle.points.value)
    prepared = prepare(lambda s: (s.add(circle), s.play(bend)))
    prepared.evaluate(1.0)
    assert circle.alpha.value == 0.0 and plain.present.value
    np.testing.assert_allclose(plain.points.value[:, 1], circle.points.value[:, 1] + 1.0)


def test_unpack_makes_a_plain_shape_of_a_parametric_one() -> None:
    circle = Circle(0.5, color="#ff0000", fill_opacity=0.5).move_to((1.0, 2.0, 0.0)).scale_by(2.0)
    plain = circle.unpack()
    assert type(plain) is Shape2D and isinstance(plain.points, RootSignal)
    np.testing.assert_allclose(plain.points.value, circle.points.value)
    assert plain.fill_opacity.value == 0.5 and plain.position.y.value == 2.0 and plain.scale.x.value == 2.0
    np.testing.assert_allclose(plain.fill_color.value, circle.fill_color.value)


def test_passing_flashes_draw_a_moving_stretch() -> None:
    line = Square()
    flash = ShowPassingFlash(line, time_width=0.5)
    prepared = prepare(lambda s: s.play(flash, rate=lambda t: t))
    prepared.evaluate(0.5)
    assert line.draw_end.value == pytest.approx(0.75) and line.draw_start.value == pytest.approx(0.25)
    prepared.evaluate(1.0)
    assert not line.present.value
    burst = Flash(point=(1.0, 2.0, 0.0), num_lines=8, style={"color": "#ffffff"})
    assert len(burst.lines.children) == 8 and burst.lines.position.y.value == 2.0
    frame = Circumscribe(Square(2.0), shape="circle", buff=0.0)
    assert frame.frame.radius.value == pytest.approx(math.sqrt(2.0))
    with pytest.raises(ValueError, match="rectangle"):
        Circumscribe(Square(), shape="star")
    assert isinstance(frame, _PassingFlash)


def test_wiggles_and_waves_return_to_rest() -> None:
    square = Square()
    text = Typst("wave")
    circle = Circle(0.5)
    start = [shape.points.value.copy() for shape in text.shapes]
    prepared = prepare(lambda s: (s.add(square, text, circle), s.play(Wiggle(square), ApplyWave(text), ApplyWave(circle), run_time=2.0)))
    prepared.evaluate(1.0)
    assert square.scale.x.value > 1.0
    assert any(not np.allclose(shape.points.value, points) for shape, points in zip(text.shapes, start, strict=True))
    prepared.evaluate(2.0)
    assert square.scale.x.value == pytest.approx(1.0) and abs(square.rotation.value.w) == pytest.approx(1.0)
    for shape, points in zip(text.shapes, start, strict=True):
        np.testing.assert_allclose(shape.points.value, points, atol=1e-9)
    assert circle.position.y.value == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(TypeError):
        Wiggle(FadeIn(Square()).node.alpha)  # type: ignore[arg-type]
