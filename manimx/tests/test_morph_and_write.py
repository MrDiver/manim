"""Morph, Create, Uncreate, Write and the path operations under them."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import Group, Scene
from manimx.animation.paths import partial_curve, partial_path
from manimx.robjects import RShape2D
from manimx.userland import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Create,
    DrawBorderThenFill,
    Morph,
    MorphMatching,
    Square,
    Triangle,
    Uncreate,
    Write,
)
from manimx.userland.geometry import (
    align_paths,
    arc_points,
    corner_points,
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


def test_aligned_paths_have_equal_shapes_and_start_close() -> None:
    circle, square = arc_points(), rectangle_points()
    a, b = align_paths(circle, square)
    assert a.shape == b.shape == (len(circle), 2)
    # Both closed: the square starts where the circle does (on the right),
    # not at its first corner, so the morph does not turn.
    np.testing.assert_allclose(b[0], [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(b[-1], b[0], atol=1e-12)
    # Open paths keep their ends.
    line, arc = corner_points([(0.0, 0.0), (1.0, 0.0)]), arc_points(angle=math.pi)
    a, b = align_paths(arc, line)
    np.testing.assert_allclose(b[[0, -1]], line[[0, -1]])
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


def _drawn(prepared, t: float) -> list[RShape2D]:
    """Every RShape2D drawn at ``t`` (its alpha times every group's above
    it is above 0), anywhere in the tree, with its own alpha."""
    found: list[RShape2D] = []

    def visit(node, alpha: float) -> None:
        alpha *= node.alpha
        if isinstance(node, RShape2D) and alpha > 0.0:
            found.append(node)
        for child in getattr(node, "children", []):
            visit(child, alpha)

    visit(prepared.state_at(t), 1.0)
    return found


def _world_box(node: RShape2D) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(node.points) * np.asarray(node.transform.scale)[:2] + np.asarray(node.transform.position)[:2]
    return points.min(axis=0), points.max(axis=0)


def test_morph_shows_in_between_shapes_and_leaves_the_target() -> None:
    circle = Circle(color="#ff0000", fill_opacity=0.5)
    square = Square(color="#0000ff", fill_opacity=1.0).shift(RIGHT * 2)
    morph = Morph(circle, square)
    prepared = prepare(lambda s: (s.add(circle), s.play(morph, run_time=2)))
    (inbetween,) = morph.inbetweens

    middle = _drawn(prepared, 1.0)
    assert [node.uid for node in middle] == [inbetween.uid]  # source and target hidden
    assert 0.5 < middle[0].fill.opacity[0] < 1.0
    # Just before the end the in-between shape covers the target exactly.
    (almost,) = _drawn(prepared, 2.0 - 1e-9)
    low, high = _world_box(almost)
    assert np.allclose(low, [1.0, -1.0], atol=1e-6) and np.allclose(high, [3.0, 1.0], atol=1e-6)
    end = shapes(prepared, 2.0)
    assert [node.uid for node in end] == [square.uid] and end[0].alpha == 1.0
    # Neither side was changed: their points are their own.
    assert all(track.signal not in (circle.points, square.points) for track in prepared.tracks)
    assert morph.target is square


def test_extra_shapes_start_and_end_as_copies() -> None:
    """Paths are paired evenly: an extra target starts as a copy of its
    source, an extra source ends as a copy of its target."""
    one = Group(Square(), name="one")
    three = Group(Square(), Circle().shift(RIGHT * 3), Circle().shift(UP * 3), name="three")
    prepared = prepare(lambda s: (s.add(one), s.play(Morph(one, three))))
    start = _drawn(prepared, 0.0)
    assert len(start) == 3
    for node in start:  # all where the square is, not points
        low, high = _world_box(node)
        assert np.allclose(low, [-1.0, -1.0], atol=1e-6) and np.allclose(high, [1.0, 1.0], atol=1e-6)

    many = Group(Square(), Circle().shift(RIGHT * 3), name="many")
    single = Group(Square(side_length=1.0).shift(UP * 2), name="single")
    prepared = prepare(lambda s: (s.add(many), s.play(Morph(many, single))))
    almost = _drawn(prepared, 0.999)
    assert len(almost) == 2
    for node in almost:  # both end on the one target
        low, high = _world_box(node)
        assert np.allclose(low, [-0.5, 1.5], atol=1e-2) and np.allclose(high, [0.5, 2.5], atol=1e-2)

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

    square = Square(fill_opacity=1.0, stroke_width=0.0, stroke_opacity=0.0)  # like a glyph: no stroke
    prepared = prepare(lambda s: s.play(DrawBorderThenFill(square, run_time=2, rate=lambda t: t)))
    (early,) = shapes(prepared, 0.5)
    assert early.fill.opacity[0] == 0.0 and early.stroke.width[0] == 2.0 and 0 < len(early.points) < 16
    assert early.stroke.opacity[0] == 1.0  # the outline shows
    (late,) = shapes(prepared, 1.5)
    assert 0.0 < late.fill.opacity[0] < 1.0 and len(late.points) == 16
    (end,) = shapes(prepared, 2.0)
    assert end.fill.opacity[0] == 1.0 and end.stroke.width[0] == 0.0 and end.stroke.opacity[0] == 0.0


def test_morph_matching_moves_equal_shapes_and_fades_the_rest() -> None:
    kept = Square().shift(LEFT * 2)
    gone = Circle().shift(RIGHT * 2)
    source = Group(kept, gone, name="source")
    moved = Square().shift(UP * 2)
    new = Triangle().shift(DOWN * 2)
    target = Group(new, moved, name="target")
    morph = MorphMatching(source, target)
    prepared = prepare(lambda s: (s.add(source), s.play(morph, run_time=2)))
    sliding, fading_out, fading_in = morph.inbetweens

    def drawn(t: float) -> dict[int, RShape2D]:
        return {node.uid: node for node in _drawn(prepared, t)}

    halfway = drawn(1.0)
    low, high = _world_box(halfway[sliding.uid])
    assert np.allclose((low + high) / 2, [-1.0, 1.0], atol=1e-6)  # halfway to its match, the same shape
    assert np.allclose(high - low, [2.0, 2.0], atol=1e-6)
    # The others cross-fade over the whole morph; the circle drifts towards the triangle.
    assert set(halfway) == {sliding.uid, fading_out.uid, fading_in.uid}
    assert 0.0 < halfway[fading_out.uid].alpha < 1.0 and 0.0 < halfway[fading_in.uid].alpha < 1.0
    triangle_center = (np.asarray(new.points.value).min(axis=0) + np.asarray(new.points.value).max(axis=0)) / 2 + [0.0, -2.0]
    np.testing.assert_allclose(halfway[fading_out.uid].transform.position[:2], ([2.0, 0.0] + triangle_center) / 2, atol=1e-6)
    assert {node.uid for node in _drawn(prepared, 2.0)} == {new.uid, moved.uid}
    with pytest.raises(ValueError, match="MorphMatching needs"):
        MorphMatching(Group(), Square())


def test_create_and_write_draw_a_part_and_never_touch_points() -> None:
    """They animate draw_end; points stay the shape's own, so the shape can
    change while it is drawn, and drawn_points is the part on screen."""
    square = Square()
    full = np.asarray(square.points.value).copy()
    prepared = prepare(lambda s: (s.play(Create(square), square.animate.scale_by(2.0), run_time=2), s.play(Write(Square().shift(UP)), run_time=1)))
    assert all(track.signal is not square.points for track in prepared.tracks)
    prepared.evaluate(1.0)
    drawn = np.asarray(square.drawn_points.value)
    assert 0 < len(drawn) < len(full) and np.allclose(drawn[0], full[0])
    assert square.scale.x.value > 1.0  # the scale animates at the same time
    prepared.evaluate(2.0)
    np.testing.assert_array_equal(square.drawn_points.value, full)


def test_morph_matching_matches_shapes_of_any_size() -> None:
    """As Manim's key: the same shape at another size (a glyph as an
    exponent) matches and grows or shrinks on the way."""
    small = Square(side_length=0.5).shift(LEFT)
    large = Square(side_length=2.0).shift(RIGHT)
    morph = MorphMatching(small, large)
    prepared = prepare(lambda s: (s.add(small), s.play(morph, run_time=2)))
    (inbetween,) = morph.inbetweens
    (middle,) = _drawn(prepared, 1.0)
    assert middle.uid == inbetween.uid and middle.alpha == 1.0  # moved, not faded


def test_morph_matching_pairs_paths_by_index_or_name() -> None:
    from manimx.userland import TypstMath

    two, three = TypstMath("x + x = 2 x"), TypstMath("x + x + x = 3 x")
    morph = MorphMatching(two, three, pairs=[("2", "3"), (0, 0), (0, 2)])
    starts = [inbetween.name for inbetween in morph.inbetweens[:3]]
    assert starts == ["2-morph", "x-morph", "x-morph"]  # the first x became two
    prepare(lambda s: (s.add(two), s.play(morph)))
    with pytest.raises(ValueError, match=r"3 paths named 'x' \(at the indices \[0, 2, 5\]"):
        MorphMatching(two, three, pairs=[("x", 0)])
    with pytest.raises(IndexError, match="no index 9"):
        MorphMatching(two, three, pairs=[(9, 0)])
