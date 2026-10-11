"""Integer, Variable, Table, Matrix, BarChart and BulletedList."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import Scene, Shape2D, derive
from manimx.userland import (
    BarChart,
    BulletedList,
    Circle,
    DecimalMatrix,
    Integer,
    IntegerMatrix,
    Matrix,
    Table,
    Typst,
    Value,
    Variable,
    current_bounds,
)
from manimx.userland.charts import _colors, _nice_range
from manimx.userland.geometry import subpaths


def prepare(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def test_integers_and_variables_show_signals() -> None:
    assert Integer(41.6).layout.text(41.6) == "42"
    angle = Value(0.5)
    theta = Variable(angle.value, "$theta$", decimals=1)
    assert theta.value is angle.value
    label_right = current_bounds(theta.label)[1, 0]
    assert current_bounds(theta.number)[0, 0] > label_right  # the number after the label
    assert abs(current_bounds(theta)[0, 0] + current_bounds(theta)[1, 0]) < 0.05  # centered
    own = Variable(1.0, "$x$")
    prepared = prepare(lambda s: (s.add(own), s.play(own.animate.set_value(2.5))))
    prepared.evaluate(1.0)
    assert own.value.value == 2.5


def test_typst_text_knows_its_baseline() -> None:
    assert Typst("x").baseline is not None
    assert Typst('#box(width: 1cm, height: 1cm, fill: black)').baseline is None


def test_a_table_lays_entries_out_on_a_grid_with_labels_and_lines() -> None:
    marker = Circle(0.3)
    table = Table([[1, "two"], [marker, None]], row_labels=["a", "b"], column_labels=["x", "y"], top_left="", outer_lines=True, line_style={"stroke_width": 1.0})
    assert [len(row) for row in table.cells] == [3, 3, 3]
    assert table.cell(2, 1) is marker and table.cell(2, 2) is None
    assert table.entries[1][0] is marker and table.row_labels[0] is table.cell(1, 0) and table.column_labels[1] is table.cell(0, 2)
    assert len(table.horizontal_lines.children) == 4 and len(table.vertical_lines.children) == 4
    assert table.row(2) == [table.cell(2, 0), marker] and table.column(2) == [table.cell(0, 2), table.cell(1, 2)]
    np.testing.assert_allclose(current_bounds(marker).mean(axis=0)[:2], table.grid.center(2, 1)[:2], atol=1e-9)
    # Text in a row shares a baseline.
    one, two = table.cell(1, 1), table.cell(1, 2)
    assert one.position.y.value + one.baseline == pytest.approx(two.position.y.value + two.baseline)
    center, width, height = table.region(1, 1, 2, 2)
    assert width == pytest.approx(table.grid.widths[1:].sum()) and height == pytest.approx(table.grid.heights[1:].sum())
    table.shift((1.0, 0.0, 0.0))
    assert table.region(1, 1, 2, 2)[0][0] == pytest.approx(center[0] + 1.0)
    with pytest.raises(ValueError, match="row labels"):
        Table([[1]], row_labels=["a", "b"])
    with pytest.raises(TypeError, match="DecimalNumber"):
        Table([[Value().value]])


def test_matrices_have_brackets_around_their_entries() -> None:
    matrix = Matrix([[1, "x^2"], [-3.5, None]])
    assert [entry is None for row in matrix.entries for entry in row] == [False, False, False, True]
    left, right = matrix.brackets
    assert isinstance(left, Shape2D) and len(subpaths(left.points.value)) == 1  # the pieces are one outline
    entries = current_bounds(matrix.entries[0][0])
    assert current_bounds(left)[1, 0] < entries[0, 0] and current_bounds(right)[0, 0] > current_bounds(matrix.entries[0][1])[1, 0]
    assert matrix.row(1) == [matrix.entries[1][0]] and matrix.column(1) == [matrix.entries[0][1]]
    for bracket in ("(", "{", "|", "‖"):
        left, right = Matrix([[1], [2]], bracket=bracket).brackets
        assert current_bounds(left)[1, 0] < current_bounds(right)[0, 0]
    assert Matrix([[1]], bracket="").brackets == []
    height = float(np.ptp(current_bounds(left)[:, 1]))
    assert height == pytest.approx(Matrix([[1], [2]], bracket="‖").grid.top * 2, abs=1e-3)  # stretched to the rows
    with pytest.raises(ValueError, match="bracket"):
        Matrix([[1]], bracket="<")


def test_decimal_matrices_follow_values_and_signals() -> None:
    values = DecimalMatrix([[1.0, 2.0], [3.0, 4.0]])
    integers = IntegerMatrix([[1, 2]])
    assert integers.entries[0][0].layout.text(1.4) == "1"
    angle = Value(0.0)
    rotation = DecimalMatrix([[derive(math.cos, angle.value)]], decimals=2)
    prepared = prepare(lambda s: (s.add(values, rotation), s.play(values.animate.set_values([[0, 0], [0, 9]]), angle.animate.set_value(math.pi))))
    prepared.evaluate(1.0)
    assert [signal.value for signal in values.values()] == [0.0, 0.0, 0.0, 9.0]
    assert rotation.entries[0][0].value.value == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="4 entries"):
        values.set_values([[1.0]])


def test_bar_charts_grow_their_bars_with_their_values() -> None:
    assert _nice_range([3, 5, 2]) == (0.0, 5.0, 1.0)
    assert _nice_range([-30, 120]) == (-50.0, 150.0, 50.0)
    assert _colors(None, 2) == [None, None] and _colors("#ff0000", 2) == ["#ff0000"] * 2
    assert _colors((1.0, 0.0, 0.0), 2) == [(1.0, 0.0, 0.0)] * 2 and _colors(["a", "b"], 3) == ["a", "b", "a"]
    chart = BarChart([2, 4], ["a", "b"], y_range=(0, 8, 2), y_length=4.0, include_values=True, bar_colors=["#ff0000", "#0000ff"])
    tops = lambda: [float(bar.points.value[:, 1].max()) for bar in chart.bars]  # noqa: E731
    bottom = float(chart.bars[0].points.value[:, 1].min())
    assert tops()[0] - bottom == pytest.approx(1.0) and tops()[1] - bottom == pytest.approx(2.0)
    assert len(chart.names) == 2 and len(chart.numbers) == 2
    prepared = prepare(lambda s: (s.add(chart), s.play(chart.animate.set_values([8, 10]))))
    prepared.evaluate(1.0)
    assert tops()[0] - bottom == pytest.approx(4.0) and tops()[1] - bottom == pytest.approx(4.0)  # 10 stops at the range's top
    assert chart.numbers[0].position.y.value > tops()[0]
    with pytest.raises(ValueError, match="2 bars"):
        chart.set_values([1.0])
    with pytest.raises(ValueError, match="at least one"):
        BarChart([])


def test_bulleted_lists_line_items_up_on_the_left() -> None:
    items = BulletedList("one", "two words", "three", spacing=0.5)
    lefts = [current_bounds(item)[0, 0] for item in items.items]
    assert max(lefts) - min(lefts) < 1e-6
    baselines = [item.position.y.value + item.baseline for item in items.items]
    np.testing.assert_allclose(np.diff(baselines), [-0.5, -0.5])
    assert BulletedList().items == []
