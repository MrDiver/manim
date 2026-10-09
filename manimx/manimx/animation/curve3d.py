"""Curve3D: a stroke-only path in 3D space, and the curves built on it.

A :class:`Curve3D` snapshots to an :class:`~manimx.robjects.RCurve3D`: cubic
Bezier curves whose stroke ribbon always faces the camera. It is a leaf.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from ..robjects import CapStyle, JointType, RCurve3D, RNode
from ..utils.color import WHITE
from . import style
from .kinds import COLORS, DISCRETE, PATH, SCALARS
from .node import (
    ABS,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    ScalableMixin,
    animatable,
)
from .signals import RootSignal, Signal
from .snapshot import version_token
from .style import Colors, Scalars


class Curve3D(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Points ``(N, 3)`` in the RBezierPath layout (four per cubic curve)
    with a stroke. The stroke width is in world units.

    Every field of its render counterpart is a signal:

    ==================  ========  ==========================================
    signal              kind      RCurve3D field
    ==================  ========  ==========================================
    ``points``          PATH      ``points``
    ``stroke_color``    COLORS    ``stroke.color`` (one color or a gradient)
    ``stroke_opacity``  SCALARS   ``stroke.opacity`` (times every ``alpha``)
    ``stroke_width``    SCALARS   ``stroke.width``
    ``stroke_joint``    DISCRETE  ``stroke.joint`` (one, or one per curve)
    ``stroke_cap``      DISCRETE  ``stroke.cap``
    ``cast_shadows``    DISCRETE  ``cast_shadows``
    ==================  ========  ==========================================
    """

    def __init__(
        self,
        points: Any = (),
        name: str | None = None,
        *,
        color: Colors = WHITE,
        stroke_width: Scalars = 0.04,
        stroke_opacity: Scalars = 1.0,
        stroke_joint: JointType | Sequence[JointType] = JointType.AUTO,
        stroke_cap: CapStyle = CapStyle.AUTO,
        cast_shadows: bool = True,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        prefix = self.name
        self.points = RootSignal(np.zeros((0, 3)) if len(points) == 0 else points, PATH, name=f"{prefix}.points")
        self.stroke_color = RootSignal(style.colors(color), COLORS, name=f"{prefix}.stroke_color")
        self.stroke_width = RootSignal(stroke_width, SCALARS, name=f"{prefix}.stroke_width")
        self.stroke_opacity = RootSignal(stroke_opacity, SCALARS, name=f"{prefix}.stroke_opacity")
        self.stroke_joint = RootSignal(style.joints(stroke_joint), DISCRETE, name=f"{prefix}.stroke_joint")
        self.stroke_cap = RootSignal(style.cap(stroke_cap), DISCRETE, name=f"{prefix}.stroke_cap")
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (
            self.points, self.stroke_color, self.stroke_width, self.stroke_opacity,
            self.stroke_joint, self.stroke_cap, self.cast_shadows,
        )  # fmt: skip

    @animatable
    def set_stroke(
        self,
        color: Colors | None = None,
        width: Scalars | None = None,
        opacity: Scalars | None = None,
        joint: JointType | Sequence[JointType] | None = None,
        cap: CapStyle | None = None,
    ) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.stroke_color, ABS, style.colors(color)))
        if width is not None:
            changes.append((self.stroke_width, ABS, width))
        if opacity is not None:
            changes.append((self.stroke_opacity, ABS, opacity))
        if joint is not None:
            changes.append((self.stroke_joint, ABS, style.joints(joint)))
        if cap is not None:
            changes.append((self.stroke_cap, ABS, style.cap(cap)))
        return changes

    @animatable
    def set_color(self, color: Colors) -> ChangeList:
        return [(self.stroke_color, ABS, style.colors(color))]

    @animatable
    def set_cast_shadows(self, cast_shadows: bool) -> ChangeList:
        return [(self.cast_shadows, ABS, cast_shadows)]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alphas = (*alphas, self.alpha)
        alpha = math.prod(signal.value for signal in alphas)
        return RCurve3D(
            uid=self.uid,
            points=self.points.value,
            stroke=style.stroke(
                self.stroke_color.value, self.stroke_opacity.value, self.stroke_width.value,
                self.stroke_joint.value, self.stroke_cap.value, alpha,
            ),  # fmt: skip
            cast_shadows=bool(self.cast_shadows.value),
            transform=self._transform(),
            versions={
                "points": version_token(self.points),
                "stroke": version_token(
                    self.stroke_color, self.stroke_width, self.stroke_opacity, self.stroke_joint, self.stroke_cap, *alphas
                ),
            },
        )


class ParametricCurve3D(Curve3D):
    """A smooth curve through ``function(t)`` (a 3D point) at ``samples``
    evenly spaced ``t`` in ``t_range`` (see RCurve3D.parametric)."""

    def __init__(
        self,
        function: Callable[[float], Any],
        t_range: tuple[float, float] = (0.0, 1.0),
        samples: int = 64,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(RCurve3D.parametric(function, t_range, samples).points, name, **style)


__all__ = ["Curve3D", "ParametricCurve3D"]
