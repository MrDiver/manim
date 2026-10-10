"""Morph, Create, Uncreate, Write and the path operations under them."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Group, Scene
from manimx.animation.paths import partial_curve, partial_path
from manimx.robjects import RShape2D
from manimx.userland import (
    RIGHT,
    UP,
    Circle,
    Create,
    DrawBorderThenFill,
    Morph,
    Square,
    Uncreate,
    Write,
)
from manimx.userland.geometry import (
    align_paths,
    arc_points,
    insert_curves,
    rectangle_points,
    subpaths,
)


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def shapes(prepared, t):
    return [node for node in prepared.state_at(t).children if isinstance(node, RShape2D)]


# --- path operations -------------------------------------------------------------


def test_partial_curve_is_the_piece_between_two_parameters() -> None:
    curve = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]])
    piece = partial_curve(curve, 0.25, 0.75)
    whole = lambda t: ((1 - t) ** 3) * curve[0] + 3 * ((1 - t) ** 2) * t * curve[1] + 3 * (1 - t) * t * t * curve[2] + t**3 * curve[3]  # noqa: E731
    np.testing.assert_allclose(piece[0], whole(0.25))
    np.testing.assert_allclose(piece[3], whole(0.75))
    np.testing.assert_allclose(partial_curve(curve, 1.0, 1.0), [curve[3]] * 4)


def test_partial_path_keeps_whole_curves_and_cuts_the_ends() -> None:
    points = rectangle_points(2.0, 2.0)  # four straight curves
    np.testing.assert_array_equal(partial_path(points, 0.0, 1.0), points)
    assert len(partial_path(points, 0.0, 0.0)) == 0  # nothing drawn yet: no dot
    half = partial_path(points, 0.0, 0.5)  # ends on a curve boundary: the next curve has no length yet
    assert len(half) == 12 and np.allclose(half[8:], points[7])
    third = partial_path(points, 0.0, 0.6)
    assert len(third) == 12 and np.allclose(third[-1], points[8] + (points[11] - points[8]) * 0.4)
    middle = partial_path(points, 0.3, 0.4)
    assert len(middle) == 4


def test_aligned_paths_have_equal_shapes_and_keep_their_ends() -> None:
    circle, square = arc_points(), rectangle_points()
    a, b = align_paths(circle, square)
    assert a.shape == b.shape == (len(circle), 2)
    np.testing.assert_allclose(b[0], square[0])
    np.testing.assert_allclose(b[-1], square[-1])
    assert len(insert_curves(square, 3)) == len(square) + 12
    # Two subpaths against one: the single one gets an empty second subpath.
    two = np.concatenate([square, square + 5.0])
    a, b = align_paths(square, two)
    assert a.shape == b.shape and len(subpaths(b)) == 2
    np.testing.assert_allclose(a[len(square) :], np.repeat(square[-1:], len(square), axis=0))
    # An empty path becomes a point at the other's center.
    a, b = align_paths(np.zeros((0, 2)), square)
    assert a.shape == b.shape and np.allclose(a, square.mean(axis=0))


# --- animations --------------------------------------------------------------------


def test_morph_turns_the_source_into_the_target_and_swaps_them_at_the_end() -> None:
    circle = Circle(color="#ff0000", fill_opacity=0.5)
    square = Square(color="#0000ff", fill_opacity=1.0).shift(RIGHT * 2)
    prepared = prepare(lambda s: (s.add(circle), s.play(Morph(circle, square), run_time=2)))

    middle = {node.uid: node for node in shapes(prepared, 1.0)}
    assert set(middle) == {circle.uid, square.uid}
    assert middle[square.uid].alpha == 0.0  # hidden until it replaces the source
    halfway = middle[circle.uid]
    assert 0.5 < halfway.fill.opacity[0] < 1.0
    end = shapes(prepared, 2.0)
    assert [node.uid for node in end] == [square.uid] and end[0].alpha == 1.0
    # Just before the end the morphed source covers the target exactly.
    almost = {node.uid: node for node in shapes(prepared, 2.0 - 1e-9)}[circle.uid]
    corners = almost.points[::4]
    assert np.allclose(corners.min(axis=0), [1.0, -1.0], atol=1e-6) and np.allclose(corners.max(axis=0), [3.0, 1.0], atol=1e-6)


def test_morph_grows_and_shrinks_extra_shapes() -> None:
    one = Group(Square(), name="one")
    three = Group(Square(), Circle().shift(RIGHT * 3), Circle().shift(UP * 3), name="three")
    prepared = prepare(lambda s: (s.add(one), s.play(Morph(one, three))))
    start = prepared.state_at(0.0).children
    grown = [child for group in start for child in getattr(group, "children", [])]
    extra = [node for node in grown if node.uid in {three.children[1].uid, three.children[2].uid}]
    assert extra and all(np.ptp(node.points, axis=0).max() < 1e-9 for node in extra)  # a point at first

    with pytest.raises(ValueError, match="Morph needs"):
        Morph(Group(), Square())


def test_create_draws_paths_one_after_another_and_uncreate_removes() -> None:
    a, b = Square(), Square().shift(RIGHT * 3)
    group = Group(a, b)
    prepared = prepare(lambda s: (s.play(Create(group), run_time=2), s.play(Uncreate(group), run_time=2)))

    def points(t):
        (snapshot,) = prepared.state_at(t).children
        return [len(child.points) for child in snapshot.children]

    assert points(0.0) == [0, 0]
    assert points(0.5)[1] == 0 and points(0.5)[0] > 0  # the first square first (lag ratio 1)
    assert points(2.0) == [16, 16]
    assert points(3.0)[0] == 16 and points(3.0)[1] < 16  # Uncreate: the last one first
    assert prepared.state_at(4.0).children == []


def test_write_draws_the_outline_then_fills_with_manims_timing() -> None:
    glyphs = Group(*(Square(side_length=0.5, fill_opacity=1.0).shift(RIGHT * i) for i in range(20)))
    write = Write(glyphs)
    assert write.run_time == 2.0 and write.lag_ratio == pytest.approx(0.2)
    assert Write(Square()).run_time == 1.0

    square = Square(fill_opacity=1.0, stroke_width=0.0)
    prepared = prepare(lambda s: s.play(DrawBorderThenFill(square, run_time=2, rate=lambda t: t)))
    (early,) = shapes(prepared, 0.5)
    assert early.fill.opacity[0] == 0.0 and early.stroke.width[0] == 2.0 and 0 < len(early.points) < 16
    (late,) = shapes(prepared, 1.5)
    assert 0.0 < late.fill.opacity[0] < 1.0 and len(late.points) == 16
    (end,) = shapes(prepared, 2.0)
    assert end.fill.opacity[0] == 1.0 and end.stroke.width[0] == 0.0
