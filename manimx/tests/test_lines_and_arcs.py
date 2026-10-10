"""Lines, arrows, angle marks and the arc-like shapes of userland."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import Scene
from manimx.userland import (
    PI,
    UP,
    Angle,
    AnnularSector,
    Annulus,
    Arc,
    ArcBetweenPoints,
    Arrow,
    Circle,
    CubicBezier,
    CurvedArrow,
    DashedLine,
    Dot,
    DoubleArrow,
    Ellipse,
    Line,
    Polygon,
    RightAngle,
    RoundedRectangle,
    Sector,
    Star,
    Vector,
)
from manimx.userland.geometry import subpaths


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def ends(points) -> np.ndarray:
    points = np.asarray(points)
    return points[[0, -1]]


def test_a_line_moves_its_ends_and_can_keep_them_on_nodes_and_points() -> None:
    line = Line((0, 0), (2, 0), buff=0.5)
    np.testing.assert_allclose(ends(line.points.value), [[0.5, 0.0], [1.5, 0.0]])
    assert line.get_length() == pytest.approx(2.0) and line.get_angle() == pytest.approx(0.0)

    dot, circle = Dot().move_to((0, 3, 0)), Circle(1.0)
    tied = Line().connect(start=dot, end=circle.point_at(0.25))
    prepared = prepare(lambda s: (s.add(line, dot, circle, tied), s.play(line.animate.put_start_and_end_on((0, 0), (0, 4)), dot.animate.shift(UP), circle.animate.set_radius(2.0))))
    prepared.evaluate(1.0)
    np.testing.assert_allclose(ends(line.points.value), [[0.0, 0.5], [0.0, 3.5]], atol=1e-12)
    np.testing.assert_allclose(ends(tied.points.value), [[0.0, 4.0], [0.0, 2.0]], atol=1e-9)


def test_dashes_stay_whole() -> None:
    dashed = DashedLine((0, 0), (1, 0), dash_length=0.1, dashed_ratio=0.5)
    parts = subpaths(np.asarray(dashed.points.value))
    assert len(parts) == 5
    assert all(np.linalg.norm(part[-1] - part[0]) == pytest.approx(0.1) for part in parts)


def test_arrows_have_tips_that_shrink_on_short_arrows() -> None:
    arrow = Arrow((0, 0), (4, 0), color="#ff0000")
    np.testing.assert_allclose(arrow.tip.points.value[0], [3.75, 0.0])  # the point, buff short of the end
    assert arrow.tip.fill_opacity.value[0] == 1.0 and arrow.line.fill_opacity.value[0] == 0.0
    assert arrow.line.points.value[-1][0] < 3.75  # under the tip, not past it
    short = Arrow((0, 0), (0.8, 0), buff=0.0)
    tip = np.asarray(short.tip.points.value)
    assert tip[:, 0].max() - tip[:, 0].min() == pytest.approx(0.2)  # a quarter of the arrow
    double = DoubleArrow((0, 0), (4, 0))
    np.testing.assert_allclose([double.tips[0].points.value[0], double.tips[1].points.value[0]], [[3.75, 0.0], [0.25, 0.0]])
    np.testing.assert_allclose(Vector((0, 2)).tip.points.value[0], [0.0, 2.0])


def test_curved_arrows_and_arcs_between_points_turn_as_asked() -> None:
    arc = ArcBetweenPoints((0, 0), (2, 0), angle=PI)
    np.testing.assert_allclose(ends(arc.points.value), [[0, 0], [2, 0]], atol=1e-12)
    assert np.asarray(arc.points.value)[:, 1].min() == pytest.approx(-1.0)  # counterclockwise: below
    curved = CurvedArrow((0, 0), (2, 0), angle=PI / 2)
    np.testing.assert_allclose(curved.tip.points.value[0], [2.0, 0.0], atol=1e-12)
    assert np.linalg.norm(curved.line.points.value[-1] - np.array([2.0, 0.0])) < 0.35


def test_angle_marks_follow_their_lines() -> None:
    first, second = Line((0, 0), (2, 0)), Line((0, 0), (0, 2))
    angle, right = Angle(first, second, radius=0.5), RightAngle(first, second, length=0.3)
    assert angle.radians.value == pytest.approx(PI / 2)
    assert Angle(first, second, other_angle=True).radians.value == pytest.approx(3 * PI / 2)
    np.testing.assert_allclose(ends(right.points.value), [[0.3, 0.0], [0.0, 0.3]], atol=1e-12)
    prepared = prepare(lambda s: (s.add(first, second, angle), s.play(second.animate.set_end((-2, 2)))))
    prepared.evaluate(1.0)
    assert angle.radians.value == pytest.approx(3 * PI / 4)
    np.testing.assert_allclose(ends(angle.points.value)[1], [-0.5 / math.sqrt(2), 0.5 / math.sqrt(2)], atol=1e-9)


@pytest.mark.parametrize(
    ("shape", "change", "check"),
    [
        (Arc(1.0, 0.0, PI / 2), lambda a: a.animate.set_angles(angle=PI), lambda p: np.allclose(p[-1], [-1.0, 0.0], atol=1e-9)),
        (Ellipse(2.0, 1.0), lambda e: e.animate.set_size(width=4.0), lambda p: np.isclose(p[:, 0].max(), 2.0)),
        (Annulus(0.5, 1.0), lambda a: a.animate.set_radii(inner_radius=0.8), lambda p: np.isclose(np.linalg.norm(p[32]), 0.8)),
        (Sector(1.0), lambda s: s.animate.set_radius(2.0), lambda p: np.isclose(np.abs(p).max(), 2.0)),
        (AnnularSector(0.5, 1.0), lambda s: s.animate.set_inner_radius(0.25), lambda p: np.isclose(np.linalg.norm(p, axis=1).min(), 0.25)),
        (Star(5, 1.0), lambda s: s.animate.set_radii(inner_radius=0.2), lambda p: np.isclose(np.linalg.norm(p, axis=1).min(), 0.2)),
        (RoundedRectangle(4.0, 2.0, 0.5), lambda r: r.animate.set_corner_radius(1.0), lambda p: np.isclose(np.abs(p).max(axis=0), [2.0, 1.0]).all()),
        (CubicBezier((0, 0), (1, 1), (2, 1), (3, 0)), lambda c: c.animate.set_controls((0, 0), (0, 2), (3, 2), (3, 0)), lambda p: np.allclose(p[1], [0, 2])),
    ],
    ids=lambda value: type(value).__name__ if not callable(value) else "",
)
def test_shapes_animate_their_parameters(shape, change, check) -> None:
    prepared = prepare(lambda s: (s.add(shape), s.play(change(shape))))
    prepared.evaluate(1.0)
    assert check(np.asarray(shape.points.value))


def test_polygon_is_a_closed_polyline() -> None:
    triangle = Polygon((0, 0), (1, 0), (0, 1))
    points = np.asarray(triangle.points.value)
    assert len(points) == 12 and np.allclose(points[0], points[-1])
