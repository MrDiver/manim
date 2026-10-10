"""DecimalNumber and Brace: shapes derived from other shapes' parameters."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Scene
from manimx.userland import DOWN, LEFT, RIGHT, Brace, Circle, DecimalNumber, Rectangle
from manimx.userland.numbers import glyph


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def test_glyphs_compile_once_with_origin_and_advance() -> None:
    zero = glyph("0")
    assert glyph("0") is zero
    assert zero.advance > 0 and zero.points[:, 0].min() >= 0.0 and zero.points[:, 1].min() < 0.01


def test_decimal_number_lays_out_glyphs_without_typst() -> None:
    number = DecimalNumber(1.5, anchor=LEFT)
    assert [len(slot.points.value) > 0 for slot in number.slots] == [True, True, True, True, False]  # "1.50" and room for a sign
    left = min(slot.points.value[:, 0].min() for slot in number.slots if len(slot.points.value))
    assert 0.0 <= left < 0.05  # the left end at the origin
    number.set_value(-0.004)  # rounds to 0.00, not "-0.00"
    assert number.layout.text(-0.004) == "0.00" and number.layout.text(-2.0) == "\u22122.00"
    number.set_value(-12.0)
    with pytest.raises(ValueError, match="room for 5"):
        number.slots[0].points.value  # noqa: B018


def test_a_number_and_a_brace_follow_the_parameters_they_show() -> None:
    circle = Circle(0.5)
    radius = DecimalNumber(circle.radius, decimals=1)
    rectangle = Rectangle(2.0, 2.0)
    brace = Brace(rectangle.edge(DOWN), depth=0.2, buff=0.1)
    width = brace.attach(DecimalNumber(rectangle.width))
    prepared = prepare(lambda s: (s.add(circle, radius, rectangle, brace, width), s.play(circle.animate.set_radius(1.0), rectangle.animate.set_size(width=1.0))))

    prepared.evaluate(1.0)
    assert radius.layout.text(radius.value.value) == "1.0"
    points = brace.points.value
    np.testing.assert_allclose(points[[0, -1]], [[-0.5, -1.1], [0.5, -1.1]], atol=1e-12)
    assert np.isclose(points[:, 1].min(), -1.3)  # the tip, below the side
    assert width.position.x.value == pytest.approx(0.0) and width.position.y.value < -1.3

    with pytest.raises(TypeError, match="follows"):
        brace.set_segment([(0, 0), (1, 0)])
    with pytest.raises(ValueError, match="UP, DOWN"):
        rectangle.edge(RIGHT + DOWN)
