"""Marks that follow the bounds of other nodes."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Group, Scene
from manimx.userland import (
    DOWN,
    PI,
    RIGHT,
    BackgroundRectangle,
    BraceLabel,
    Circle,
    Cross,
    Square,
    SurroundingRectangle,
    TypstMath,
    Underline,
    bounds,
)


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def box(points) -> np.ndarray:
    points = np.asarray(points)
    return np.array([points.min(axis=0), points.max(axis=0)])


def test_bounds_go_through_every_transform_to_the_points() -> None:
    inner = Square(2.0).shift(RIGHT).rotate(PI / 4)
    group = Group(inner, Circle(0.5).shift(DOWN * 3)).shift(RIGHT).scale_by(2.0)
    low, high = bounds(group).value
    r = np.sqrt(2)
    np.testing.assert_allclose(low[:2], [0.0, -7.0], atol=1e-9)  # the circle: x from 2 * -0.5 + 1
    np.testing.assert_allclose(high[:2], [1 + 2 * (1 + r), 2 * r], atol=1e-9)  # the turned square
    assert np.all(bounds(Group()).value == 0.0)


def test_marks_follow_the_node_they_mark() -> None:
    circle = Circle(1.0)
    around = SurroundingRectangle(circle, buff=0.25)
    under, cross, behind = Underline(circle, buff=0.1), Cross(circle), BackgroundRectangle(circle, color="#000000")
    np.testing.assert_allclose(box(around.points.value), [[-1.25, -1.25], [1.25, 1.25]], atol=1e-9)
    assert behind.fill_opacity.value[0] == 0.75 and behind.stroke_width.value[0] == 0.0
    prepared = prepare(lambda s: (s.add(circle, around, under, cross, behind), s.play(circle.animate.set_radius(2.0).shift(RIGHT))))
    prepared.evaluate(1.0)
    np.testing.assert_allclose(box(around.points.value), [[-1.25, -2.25], [3.25, 2.25]], atol=1e-9)
    np.testing.assert_allclose(under.points.value[[0, -1]], [[-1.0, -2.1], [3.0, -2.1]], atol=1e-9)
    np.testing.assert_allclose(box(cross.points.value), [[-1.0, -2.0], [3.0, 2.0]], atol=1e-9)


def test_a_brace_label_sits_under_a_formula() -> None:
    formula = TypstMath("a^2 + b^2 = c^2")
    label = TypstMath("\"pythagoras\"")
    braced = BraceLabel(formula, label, DOWN)
    low, high = bounds(formula).value
    brace_box = box(braced.brace.points.value)
    assert brace_box[1, 1] < low[1] and brace_box[0, 0] == pytest.approx(low[0])
    assert label.position.y.value < brace_box[0, 1]
