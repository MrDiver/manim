"""Boolean shapes (Union, Intersection, Difference, Exclusion) and TracedPath."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Group, Scene
from manimx.animation.authoring import current
from manimx.userland import Circle, Difference, Dot, Exclusion, Intersection, MoveAlongPath, Rectangle, Square, TracedPath, Union
from manimx.userland.geometry import boolean_points, subpaths


def area(points: np.ndarray) -> float:
    """The area inside a path of cubic curves, by the shoelace formula on
    its curves' end points (exact for straight-edged shapes)."""
    total = 0.0
    for piece in subpaths(points):
        corners = piece[::4]
        total += 0.5 * float(np.sum(corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1]))
    return abs(total)


def two_squares() -> tuple[Square, Square]:
    return Square(2.0), Square(2.0).move_to((1.0, 1.0, 0.0))  # overlapping in a 1 by 1 square


def test_boolean_shapes_outline_the_combined_areas() -> None:
    assert area(current(Union(*two_squares()).points)) == pytest.approx(7.0)
    assert area(current(Intersection(*two_squares()).points)) == pytest.approx(1.0)
    assert area(current(Difference(*two_squares()).points)) == pytest.approx(3.0)
    exclusion = current(Exclusion(*two_squares()).points)
    assert len(subpaths(exclusion)) == 2
    np.testing.assert_allclose(exclusion[:, 0].min(), -1.0)
    np.testing.assert_allclose(exclusion[:, 0].max(), 2.0)


def test_groups_count_as_the_area_of_all_their_paths() -> None:
    left, right = Square(1.0).move_to((-2.0, 0.0, 0.0)), Square(1.0).move_to((2.0, 0.0, 0.0))
    group = Group(left, right).shift((0.0, 1.0, 0.0))
    band = Rectangle(10.0, 1.0).move_to((0.0, 1.0, 0.0))  # through both
    assert area(current(Intersection(group, band).points)) == pytest.approx(2.0)


def test_a_boolean_shape_follows_its_inputs() -> None:
    a, b = two_squares()
    union = Union(a, b)

    class Built(Scene):
        def construct(self):
            self.add(union)
            self.play(b.animate.move_to((5.0, 0.0, 0.0)))  # apart: two squares, no overlap

    prepared = Built().prepare()
    prepared.evaluate(0.0)
    assert area(union.points.value) == pytest.approx(7.0)
    prepared.evaluate(prepared.duration)
    assert area(union.points.value) == pytest.approx(8.0)
    assert len(subpaths(union.points.value)) == 2


def test_boolean_shapes_need_two_nodes_and_known_operations() -> None:
    with pytest.raises(ValueError, match="two nodes"):
        Union(Square())
    with pytest.raises(ValueError, match="unknown boolean operation"):
        boolean_points("merge", [])
    assert len(boolean_points("union", [])) == 0


def traced(dissipating_time: float | None = None) -> tuple:
    dot = Dot().move_to((1.0, 0.0, 0.0))
    trails = []

    class Built(Scene):
        def construct(self):
            self.add(dot)
            self.wait(1.0)  # before the trail
            trails.append(TracedPath(dot, dissipating_time=dissipating_time, samples_per_second=120))
            self.add(trails[0])
            self.play(MoveAlongPath(dot, Circle(1.0)), run_time=2.0, rate=lambda t: t)

    prepared = Built().prepare()
    return prepared, trails[0]


def test_a_traced_path_follows_where_the_point_has_been() -> None:
    prepared, trail = traced()
    prepared.evaluate(0.5)
    assert len(trail.points.value) == 0  # not made yet
    prepared.evaluate(2.0)  # half way round
    half = trail.points.value
    assert half.shape[1] == 3
    np.testing.assert_allclose(half[0], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(half[-1], [-1.0, 0.0, 0.0], atol=1e-6)  # ends at the point
    assert np.all(half[:, 1] >= -1e-6)  # the upper half circle
    prepared.evaluate(3.0)
    whole = trail.points.value
    assert np.linalg.norm(whole[:, :2], axis=1) == pytest.approx(np.ones(len(whole)), abs=1e-3)
    assert whole[:, 1].min() < -0.99


def test_a_dissipating_trail_keeps_only_its_last_seconds() -> None:
    prepared, trail = traced(dissipating_time=0.5)
    prepared.evaluate(3.0)  # the last quarter of the circle
    tail = trail.points.value
    assert tail[:, 0].min() > -0.01 and tail[:, 1].max() < 0.01


def test_a_traced_path_is_made_in_construct_and_traces_points() -> None:
    with pytest.raises(RuntimeError, match="inside construct"):
        TracedPath(Dot())
    errors = []

    class Built(Scene):
        def construct(self):
            for bad in (lambda: TracedPath(Dot(), samples_per_second=0), lambda: TracedPath(Dot(), dissipating_time=0.0), lambda: TracedPath("dot")):
                try:
                    bad()
                except (TypeError, ValueError) as error:
                    errors.append(type(error))
            self.add(TracedPath(Square().point_at(0.25)))
            self.wait(0.1)

    Built().prepare()
    assert errors == [ValueError, ValueError, TypeError]
