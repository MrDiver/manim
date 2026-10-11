"""Number lines, axes, planes and graphs whose ranges are signals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import Scene
from manimx.userland import (
    Axes,
    Dot,
    FunctionGraph,
    NumberLine,
    NumberPlane,
    ParametricFunction,
    Value,
)


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def shown(line: NumberLine) -> list[float]:
    return [number.value.value for number in line.numbers if number.alpha.value > 0]


def test_a_number_line_places_ticks_and_labels_by_its_range() -> None:
    line = NumberLine((-2, 2, 1), length=8.0, include_numbers=True)
    ticks = np.asarray(line.ticks.points.value).reshape(-1, 4, 2)[:, 0, 0]
    np.testing.assert_allclose(ticks, [-4, -2, 0, 2, 4])
    assert shown(line) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    np.testing.assert_allclose(line.n2p(1.0), [2.0, 0.0, 0.0])
    first = line.numbers[0]
    assert first.position.x.value == pytest.approx(-4.0) and first.position.y.value == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="maximum > minimum"):
        NumberLine((1, 1))


def test_zooming_axes_moves_labels_and_rescales_plots() -> None:
    axes = Axes((0, 5, 1), (0, 25, 5), include_numbers=True)
    graph = axes.plot(lambda x: x**2)
    prepared = prepare(lambda s: (s.add(axes, graph), s.play(axes.animate.set_ranges(x=(0, 10, 2), y=(0, 100, 20)))))
    assert len(axes.x_axis.numbers) >= 7  # the pool grew for the wider range
    prepared.evaluate(1.0)
    assert shown(axes.x_axis) == [2.0, 4.0, 6.0, 8.0]  # 0 is left out where the axes cross, 10 under the tip
    points = np.asarray(graph.points.value)
    np.testing.assert_allclose(points[[0, -1]], [[-6.0, -3.0], [6.0, 3.0]], atol=1e-9)  # x^2 still corner to corner
    middle = points[len(points) // 2]  # x = 5: y = 25 of 100
    np.testing.assert_allclose(middle, [0.0, -3.0 + 6.0 * 0.25], atol=1e-9)


def test_points_on_axes_follow_values_and_the_axes() -> None:
    axes = Axes((0, 4, 1), (0, 16, 4), x_length=8.0, y_length=8.0)
    t = Value(1.0)
    rider = Dot().follow(axes.graph_point(t.value, lambda x: x * x))
    prepared = prepare(lambda s: (s.add(axes, rider, t), s.play(t.animate.set_value(3.0), axes.animate.shift((1, 0, 0)))))
    prepared.evaluate(1.0)
    np.testing.assert_allclose(rider.position.value, (axes.c2p(3.0, 9.0)), atol=1e-9)
    np.testing.assert_allclose(rider.position.value, (3.0, 0.5, 0.0), atol=1e-9)


def test_a_plane_has_a_grid_under_its_axes_and_plain_graphs_draw_as_is() -> None:
    plane = NumberPlane((-2, 2, 1), (-1, 1, 1), x_length=4.0, y_length=2.0)
    assert plane.children[:3] == [plane.minor_x_lines, plane.minor_y_lines, plane.grid]
    assert len(np.asarray(plane.grid.points.value)) == (3 + 1) * 4  # inside the plane, not on its edges
    graph = FunctionGraph(math.sin, (0, math.pi))
    np.testing.assert_allclose(np.asarray(graph.points.value)[[0, -1]], [[0, 0], [math.pi, 0]], atol=1e-12)
    circle = ParametricFunction(lambda t: (math.cos(t), math.sin(t)), (0, math.tau))
    assert np.allclose(np.linalg.norm(np.asarray(circle.points.value)[::4], axis=1), 1.0)


def lines(shape) -> list[float]:
    """Where a horizontal line's ticks are along it."""
    return [round(float(x), 9) for x in np.asarray(shape.points.value).reshape(-1, 4, 2)[:, 0, 0]]


def test_adaptive_ticks_fade_in_finer_levels_zooming_in_and_coarser_ones_zooming_out() -> None:
    line = NumberLine((-2, 2, 1), length=8.0, include_numbers=True)
    prepared = prepare(lambda s: (s.add(line), s.play(line.animate.set_range(-0.5, 0.5)), s.play(line.animate.set_range(-40, 40))))
    prepared.evaluate(0.0)
    assert lines(line.ticks) == [-4, -2, 0, 2, 4] and line.minor_ticks.alpha.value == 0.0  # as made: the step only
    prepared.evaluate(0.5)  # zooming in: the half steps fade in between
    assert 0.0 < line.minor_ticks.alpha.value < 1.0 and len(lines(line.minor_ticks)) > 0
    prepared.evaluate(1.0)
    assert line.minor_ticks.alpha.value > 0.9 and shown(line) == [0.0]  # no labels with places the line has none of
    prepared.evaluate(2.0)  # zoomed out 10 times: steps of 5 and 10
    assert all(abs(value) % 10 == 0 for value in shown(line)) and all(round(x * 10) % 5 == 0 for x in lines(line.ticks))


def test_fixed_ticks_stay_on_the_step() -> None:
    line = NumberLine((-2, 2, 1), length=8.0, adaptive=False)
    prepared = prepare(lambda s: (s.add(line), s.play(line.animate.set_range(-0.5, 0.5))))
    prepared.evaluate(1.0)
    assert lines(line.ticks) == [0.0] and lines(line.minor_ticks) == []
