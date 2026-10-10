"""Shapes as proxies for their parameters (userland/shapes2d.py): points
derived from parameter signals, animated through them."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Change, Scene
from manimx.animation.authoring import change
from manimx.animation.plan import ScenePlan
from manimx.userland import (
    Circle,
    Create,
    Morph,
    Polyline,
    Rectangle,
    RegularPolygon,
    Square,
    Triangle,
)
from manimx.userland.geometry import arc_points, rectangle_points


def _prepare(construct):
    class Composed(Scene):
        def construct(self) -> None:
            construct(self)

    return Composed().prepare()


def test_points_follow_the_parameters() -> None:
    circle = Circle(2.0)
    np.testing.assert_allclose(circle.points.value, arc_points(2.0))
    rectangle = Rectangle(3.0, 1.0)
    rectangle.set_size(width=5.0)
    np.testing.assert_allclose(rectangle.points.value, rectangle_points(5.0, 1.0))
    assert len(RegularPolygon(5).points.value) == 20 and Triangle().n.value == 3
    with pytest.raises(ValueError, match="positive"):
        Circle(-1.0)
    with pytest.raises(ValueError, match="3 corners"):
        RegularPolygon(2)
    with pytest.raises(TypeError, match="derived"):  # the parameters own the points
        change(circle.points, Change.ABSOLUTE, np.zeros((0, 2)))


def test_a_parameter_animates_the_shape_and_the_core_agrees() -> None:
    circle = Circle(1.0)
    square = Square(2.0)
    line = Polyline([(0, 0), (1, 0), (1, 1)])
    prepared = _prepare(
        lambda s: (
            s.add(circle, square, line),
            s.play(circle.animate.set_radius(3.0), square.animate.set_side_length(1.0), line.animate.set_corners([(0, 0), (2, 0), (2, 2)]), run_time=2),
        )
    )
    prepared.evaluate(1.0)
    radius = circle.radius.value
    assert 1.0 < radius < 3.0
    np.testing.assert_allclose(circle.points.value, arc_points(radius))
    plan = ScenePlan(prepared)
    for time in (0.0, 0.7, 1.0, 2.0):
        prepared.evaluate(time)
        plan.evaluate(time)
        for shape in (circle, square, line):
            np.testing.assert_allclose(plan.value(shape.drawn_points), shape.drawn_points.value, atol=1e-12)


def test_a_shape_is_drawn_while_its_parameter_changes() -> None:
    circle = Circle(1.0)
    prepared = _prepare(lambda s: s.play(Create(circle), circle.animate.set_radius(2.0), run_time=2))
    prepared.evaluate(1.0)
    drawn = np.asarray(circle.drawn_points.value)
    assert 0 < len(drawn) < len(circle.points.value)
    assert np.linalg.norm(drawn[0]) == pytest.approx(circle.radius.value)  # on the growing circle
    prepared.evaluate(2.0)
    np.testing.assert_allclose(circle.drawn_points.value, arc_points(2.0))


def test_a_morph_leaves_the_target_with_its_parameters() -> None:
    circle, square = Circle(1.0), Square(1.0)

    def construct(scene) -> None:
        scene.add(circle)
        scene.play(Morph(circle, square))
        scene.play(square.animate.set_side_length(3.0))

    prepared = _prepare(construct)
    prepared.evaluate(prepared.duration)
    assert square.side_length.value == 3.0
    np.testing.assert_allclose(square.points.value, rectangle_points(3.0, 3.0))
    assert circle.radius.value == 1.0  # the source is left as it was
