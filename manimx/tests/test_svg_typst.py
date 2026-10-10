"""The SVG reader and Typst text (manimx.userland.svg, manimx.userland.typst)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import Shape2D
from manimx.userland import SVG, Typst, TypstMath, Write, read_svg
from manimx.userland.svg import path_points


def _svg(body: str, size: str = 'width="96" height="96" viewBox="0 0 96 96"') -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" {size}>{body}</svg>'


def test_path_commands_become_cubic_curves() -> None:
    points = path_points("M0 0 L10 0 H20 V10 C20 20 10 20 10 10 S0 0 0 10 Q5 15 10 10 T20 10 Z")
    assert points.shape[1] == 2 and len(points) % 4 == 0
    curves = points.reshape(-1, 4, 2)
    assert np.allclose(curves[0], [[0, 0], [10 / 3, 0], [20 / 3, 0], [10, 0]])
    # Every curve starts where the previous ended (one subpath), and Z closes it.
    assert np.allclose(curves[1:, 0], curves[:-1, 3])
    assert np.allclose(curves[-1, 3], [0, 0])
    # S reflects the previous handle.
    assert np.allclose(curves[4, 1], 2 * curves[3, 3] - curves[3, 2])


def test_relative_commands_and_implicit_lines() -> None:
    absolute = path_points("M1 1 L4 1 L4 5")
    relative = path_points("m1 1 3 0 0 4")
    assert np.allclose(absolute, relative)


def test_arc_flags_without_separators() -> None:
    compact = path_points("M0 0 a5 5 0 105 5")
    spaced = path_points("M0 0 a 5 5 0 1 0 5 5")
    assert np.allclose(compact, spaced)
    # A large arc of a radius-5 circle: three quarters, three curves.
    assert len(compact) == 12
    center = np.array([0.0, 5.0])
    anchors = compact.reshape(-1, 4, 2)[:, [0, 3]].reshape(-1, 2)
    assert np.allclose(np.linalg.norm(anchors - center, axis=1), 5.0)


def test_basic_shapes_size_and_centering() -> None:
    # 96 px is one inch, one scene unit; y points up after reading.
    paths = read_svg(_svg('<rect x="0" y="0" width="96" height="48" fill="#ff0000"/>'))
    (rect,) = paths
    low, high = rect.points.min(axis=0), rect.points.max(axis=0)
    assert np.allclose(low, [-0.5, -0.25]) and np.allclose(high, [0.5, 0.25])
    assert rect.fill_color == (1.0, 0.0, 0.0) and rect.stroke_color is None

    circle, line, polygon, ellipse = read_svg(
        _svg('<circle cx="48" cy="48" r="48"/><line x1="0" y1="0" x2="96" y2="96" stroke="blue"/>'
             '<polygon points="0,0 96,0 48,96"/><ellipse cx="48" cy="48" rx="48" ry="24"/>')
    )
    assert np.allclose(np.linalg.norm(circle.points.reshape(-1, 4, 2)[:, 0], axis=1), 0.5)
    assert line.fill_color == (0.0, 0.0, 0.0) and line.stroke_color == (0.0, 0.0, 1.0)  # SVG's default fill is black
    assert len(polygon.points) == 12
    assert np.isclose(ellipse.points[:, 1].max(), 0.25)


def test_size_from_units_and_requested_height() -> None:
    (rect,) = read_svg(_svg('<rect width="72" height="72"/>', 'width="72pt" height="72pt" viewBox="0 0 72 72"'))
    assert np.isclose(np.ptp(rect.points[:, 0]), 1.0)
    (rect,) = read_svg(_svg('<rect width="10" height="20"/>'), height=3.0)
    assert np.isclose(np.ptp(rect.points[:, 1]), 3.0) and np.isclose(np.ptp(rect.points[:, 0]), 1.5)


def test_transforms_use_and_inherited_paint() -> None:
    body = (
        '<defs><symbol id="s"><path d="M0 0 h10 v10 h-10 Z"/></symbol>'
        '<linearGradient id="g"><stop stop-color="#00ff00"/></linearGradient></defs>'
        '<g fill="#0000ff" opacity="0.5" transform="translate(10 0)">'
        '<use xlink:href="#s" x="5"/><rect width="10" height="10" style="fill:url(#g); stroke:white; stroke-width:2" transform="rotate(90) scale(2)"/></g>'
    )
    used, rect = read_svg(_svg(body))
    assert used.fill_color == (0.0, 0.0, 1.0) and used.fill_opacity == 0.5
    assert rect.fill_color == (0.0, 1.0, 0.0) and rect.stroke_color == (1.0, 1.0, 1.0)
    # The rotated, scaled rect spans x in [-20, 0] + 10, the used square [15, 25].
    width = np.ptp(np.vstack([used.points, rect.points])[:, 0])
    assert np.isclose(width, 35 / 96)
    # Stroke width 2 user units scaled by 2: 4 px of an inch.
    assert np.isclose(rect.stroke_width, 4 / 96 / 0.01)


def test_hidden_and_unknown() -> None:
    assert read_svg(_svg('<rect width="10" height="10" display="none"/><text>hi</text>')) == []
    with pytest.raises(ValueError, match="color"):
        read_svg(_svg('<rect width="10" height="10" fill="papayawhip"/>'))
    with pytest.raises(ValueError, match="needs"):
        path_points("M 0")


def test_svg_group_overrides_style(tmp_path) -> None:
    file = tmp_path / "a.svg"
    file.write_text(_svg('<rect width="10" height="10" fill="#ff0000"/><circle cx="50" cy="50" r="5"/>'))
    svg = SVG(file, width=2.0, fill_opacity=0.5)
    assert len(svg.shapes) == 2 and all(isinstance(shape, Shape2D) for shape in svg.shapes)
    assert np.isclose(np.ptp(np.vstack([s.points.value for s in svg.shapes])[:, 0]), 2.0)
    assert np.allclose(svg.shapes[0].fill_color.value, [[1, 0, 0]])
    assert all(np.allclose(s.fill_opacity.value, 0.5) for s in svg.shapes)
    assert len(SVG(str(file)).shapes) == len(SVG(file.read_text()).shapes) == 2


def test_typst_glyphs_and_math() -> None:
    text = Typst("Hi", '#set text(size: 72pt, fill: rgb("#ffffff"))')
    assert text.markup == "Hi" and len(text.shapes) == 2
    assert all(np.allclose(shape.fill_color.value, [[1, 1, 1]]) for shape in text.shapes)
    assert all(np.allclose(shape.stroke_width.value, 0.0) for shape in text.shapes)
    # 72pt text: the capital H is a bit under one inch tall.
    height = np.ptp(text.shapes[0].points.value[:, 1])
    assert 0.4 < height < 1.0
    formula = TypstMath("a/b", height=2.0, color="#e07a5f")
    assert len(formula.shapes) == 3  # a, b and the fraction rule
    assert np.isclose(np.ptp(np.vstack([s.points.value for s in formula.shapes])[:, 1]), 2.0)
    assert not math.isnan(Write(formula).run_time)
