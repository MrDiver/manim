"""The core's stroke records (core/src/geometry/strokes.rs) match the Python reference."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.renderer.core_callbacks import stroke_records
from manimx.robjects import CapStyle, JointType
from manimx.userland.geometry import corner_points

from manimx import _core

SQUARE = corner_points([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
OPEN_AND_CLOSED = np.concatenate([corner_points([(0, 0), (2, 1), (3, 0)]), SQUARE + 5])


@pytest.mark.parametrize(
    ("points", "colors", "opacities", "widths", "joints"),
    [
        (SQUARE, [[1.0, 0.5, 0.25]], [1.0], [4.0], [JointType.AUTO]),
        (OPEN_AND_CLOSED, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.2, 1.4], [1.0, 3.0, -2.0, 7.0], [JointType.ROUND, JointType.BEVEL]),
        (np.column_stack([OPEN_AND_CLOSED, np.linspace(0, 1, len(OPEN_AND_CLOSED))]), [[0.3, 0.3, 0.3]], [0.5], [2.0], [JointType.BEVEL]),
        (SQUARE, [[1.0, 1.0, 1.0]], [0.0], [4.0], [JointType.AUTO]),  # invisible
    ],
)
def test_core_stroke_records_match_python(points, colors, opacities, widths, joints) -> None:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    expected = stroke_records(points, colors[0] if len(colors) == 1 else colors, np.array(opacities), np.array(widths), tuple(joints), CapStyle.ROUND)
    core = _core.stroke_records(points, colors, opacities, widths, [joint.value for joint in joints], CapStyle.ROUND.value)
    assert core == expected
