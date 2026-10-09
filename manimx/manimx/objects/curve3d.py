from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .path import BezierPath


@dataclass(slots=True, eq=False)
class Curve3D(BezierPath):
    """A stroke-only path of cubic Bezier curves in 3D space.

    ``points`` is ``(N, 3)`` in the ``VMobject`` layout (see
    :class:`BezierPath`), so the outline of a :class:`Shape2D` lifts to a
    curve exactly. The stroke supports everything a ``Shape2D`` stroke does
    (per-point color, opacity and width, joints, caps), but its ribbon always
    faces the camera. Its width is in world units, so under a perspective
    camera it gets thinner with distance.
    """

    DIMENSION: ClassVar[int] = 3

    @classmethod
    def parametric(
        cls,
        function: Callable[[float], Any],
        t_range: tuple[float, float],
        samples: int = 64,
        **kwargs: Any,
    ) -> Curve3D:
        """Smooth curve through ``function(t)`` at ``samples`` evenly spaced ``t``.

        Handles come from the derivative estimated by central differences,
        which makes the curve C1-continuous (like a Catmull-Rom spline).
        """
        ts = np.linspace(t_range[0], t_range[1], samples + 1)
        anchors = np.array([function(t) for t in ts], dtype=np.float64)
        step = ts[1] - ts[0]
        derivative = np.gradient(anchors, step, axis=0)
        handles = derivative * step / 3
        curves = np.stack(
            [anchors[:-1], anchors[:-1] + handles[:-1], anchors[1:] - handles[1:], anchors[1:]],
            axis=1,
        )
        return cls(points=curves.reshape(-1, 3), **kwargs)
