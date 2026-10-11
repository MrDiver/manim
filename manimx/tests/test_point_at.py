"""Points along a path: point_at, Shape2D.point_at and MoveAlongPath."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Group, Scene
from manimx.animation.signals.paths import point_at
from manimx.userland import RIGHT, UP, Circle, Dot, MoveAlongPath, Polyline
from manimx.userland.geometry import arc_points, corner_points


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def test_by_length_spaces_points_evenly_and_by_curves_does_not() -> None:
    # One short curve and one long one.
    path = corner_points([(0.0, 0.0), (0.1, 0.0), (10.0, 0.0)])
    np.testing.assert_allclose(point_at(path, 0.5), [5.0, 0.0])
    np.testing.assert_allclose(point_at(path, 0.5, by_length=False), [0.1, 0.0])
    np.testing.assert_allclose(point_at(path, 0.0), [0.0, 0.0])
    np.testing.assert_allclose(point_at(path, 1.0), [10.0, 0.0])
    # Along a circle, equal steps are equal arcs.
    angles = [np.arctan2(*point_at(arc_points(), t)[::-1]) % (2 * np.pi) for t in (0.1, 0.35)]
    np.testing.assert_allclose(angles, [0.2 * np.pi, 0.7 * np.pi], atol=1e-3)
    np.testing.assert_allclose(point_at(np.zeros((0, 2)), 0.5), [0.0, 0.0])


def test_a_node_follows_a_point_of_a_changing_shape() -> None:
    circle = Circle(1.0).shift(RIGHT * 2)
    dot = Dot().follow(circle.point_at(0.25))
    prepared = prepare(lambda s: (s.add(circle, dot), s.play(circle.animate.set_radius(2.0).shift(UP))))
    prepared.evaluate(0.0)
    np.testing.assert_allclose(dot.position.value, (2.0, 1.0, 0.0), atol=1e-9)
    prepared.evaluate(1.0)
    np.testing.assert_allclose(dot.position.value, (2.0, 3.0, 0.0), atol=1e-9)  # the top of the moved, larger circle


def test_move_along_path_goes_through_the_path_in_the_nodes_space() -> None:
    path = Polyline([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0)]).shift(UP)
    dot = Dot()
    holder = Group(dot).shift(RIGHT)  # the dot's own space is shifted
    prepared = prepare(lambda s: (s.add(path, holder), s.play(MoveAlongPath(dot, path, rate=lambda t: t), run_time=2)))
    prepared.evaluate(1.0)
    np.testing.assert_allclose(dot.position.value, (2.0, 1.0, 0.0), atol=1e-6)  # half the length (3 of 6) up one, minus the group's shift
    prepared.evaluate(2.0)
    np.testing.assert_allclose(dot.position.value, (3.0, 3.0, 0.0), atol=1e-6)
    with pytest.raises(TypeError, match="Shape2D or Curve3D"):
        MoveAlongPath(dot, holder)
