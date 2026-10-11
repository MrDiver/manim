"""Boolean shapes (Union, Intersection, Difference, Exclusion)."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Group, Scene
from manimx.animation.authoring import current
from manimx.userland import Difference, Exclusion, Intersection, Rectangle, Square, Union
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

