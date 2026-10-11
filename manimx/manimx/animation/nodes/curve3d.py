"""Curve3D: a stroke-only path in 3D space.

A :class:`Curve3D` snapshots to an :class:`~manimx.robjects.RCurve3D`: cubic
Bezier curves whose stroke ribbon always faces the camera. It is a leaf.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from ...robjects import CapStyle, JointType
from ...utils.color import WHITE
from ..signals.kinds import COLORS, DISCRETE, PATH, SCALARS
from ..signals.paths import PointSignal, drawn_part, point_signal
from ..signals.signal import RootSignal, Signal
from . import style
from .node import (
    ABS,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    ScalableMixin,
    animatable,
)
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
    ``stroke_opacity``  SCALARS   ``stroke.opacity``
    ``stroke_width``    SCALARS   ``stroke.width``
    ``stroke_joint``    DISCRETE  ``stroke.joint`` (one, or one per curve)
    ``stroke_cap``      DISCRETE  ``stroke.cap``
    ``cast_shadows``    DISCRETE  ``cast_shadows``
    ==================  ========  ==========================================

    ``draw_start`` and ``draw_end`` say which part of the path is drawn and
    ``drawn_points`` is that part, as for Shape2D.
    """

    render_kind = "curve3d"

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
        self.draw_start, self.draw_end, self.drawn_points = drawn_part(self.points, prefix)
        self.stroke_color = RootSignal(style.colors(color), COLORS, name=f"{prefix}.stroke_color")
        self.stroke_width = RootSignal(stroke_width, SCALARS, name=f"{prefix}.stroke_width")
        self.stroke_opacity = RootSignal(stroke_opacity, SCALARS, name=f"{prefix}.stroke_opacity")
        self.stroke_joint = RootSignal(style.joints(stroke_joint), DISCRETE, name=f"{prefix}.stroke_joint")
        self.stroke_cap = RootSignal(style.cap(stroke_cap), DISCRETE, name=f"{prefix}.stroke_cap")
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")

    def point_at(self, proportion: float | Signal, by_length: bool = True) -> PointSignal:
        """The point ``proportion`` (0 to 1, or a SCALAR signal) of the way
        along ``points``, in the parent's space, following the path, its
        position, scale and rotation: ``dot.follow(shape.point_at(0.25))``.
        With ``by_length`` equal steps are equally far apart along the
        path; else each curve gets an equal share (see paths.point_at).
        It is of the whole path, not only the part drawn."""
        return point_signal(self, proportion, by_length)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (
            self.points, self.draw_start, self.draw_end, self.drawn_points,
            self.stroke_color, self.stroke_width, self.stroke_opacity,
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

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        yield from super().render_fields()
        # The render object draws the part of the points from draw_start to draw_end.
        yield from (
            ("points", self.drawn_points), ("stroke_color", self.stroke_color),
            ("stroke_opacity", self.stroke_opacity), ("stroke_width", self.stroke_width),
            ("stroke_joint", self.stroke_joint), ("stroke_cap", self.stroke_cap), ("cast_shadows", self.cast_shadows),
        )  # fmt: skip


__all__ = ["Curve3D"]
