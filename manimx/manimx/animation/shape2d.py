"""Shape2D: a vector shape in its local plane, and the shapes built on it.

A :class:`Shape2D` snapshots to an :class:`~manimx.robjects.RShape2D`: a
path with fill and stroke. It is a leaf; to attach other nodes to it, put
them in a :class:`~manimx.animation.Group` together.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from ..robjects import CapStyle, Fill, JointType, RNode, RShape2D, Stroke
from ..utils.color import WHITE
from . import style
from .geometry import arc_points, rectangle_points, regular_polygon_points
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


class Shape2D(PositionMixin, RotationMixin, ScalableMixin, Node):
    """A path in the local XY plane with fill and stroke.

    Every field of its render counterpart is a signal:

    ==================  ========  ==========================================
    signal              kind      RShape2D field
    ==================  ========  ==========================================
    ``points``          PATH      ``points``
    ``fill_color``      COLORS    ``fill.color`` (one color or a gradient)
    ``fill_opacity``    SCALARS   ``fill.opacity`` (times every ``alpha``)
    ``stroke_color``    COLORS    ``stroke.color``
    ``stroke_opacity``  SCALARS   ``stroke.opacity`` (times every ``alpha``)
    ``stroke_width``    SCALARS   ``stroke.width``
    ``stroke_joint``    DISCRETE  ``stroke.joint`` (one, or one per curve)
    ``stroke_cap``      DISCRETE  ``stroke.cap``
    ``cast_shadows``    DISCRETE  ``cast_shadows``
    ==================  ========  ==========================================

    Lists (gradients, tapering widths) are spread over the path's points
    by the renderer. ``color`` sets fill and stroke color together. The
    fill is transparent unless ``fill_opacity`` is given.
    """

    def __init__(
        self,
        points: Any = (),
        name: str | None = None,
        *,
        color: Colors = WHITE,
        fill_color: Colors | None = None,
        fill_opacity: Scalars = 0.0,
        stroke_color: Colors | None = None,
        stroke_width: Scalars = 4.0,
        stroke_opacity: Scalars = 1.0,
        stroke_joint: JointType | Sequence[JointType] = JointType.AUTO,
        stroke_cap: CapStyle = CapStyle.AUTO,
        cast_shadows: bool = True,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        prefix = self.name
        self.points = RootSignal(np.zeros((0, 2)) if len(points) == 0 else points, PATH, name=f"{prefix}.points")
        self.fill_color = RootSignal(style.colors(color if fill_color is None else fill_color), COLORS, name=f"{prefix}.fill_color")
        self.fill_opacity = RootSignal(fill_opacity, SCALARS, name=f"{prefix}.fill_opacity")
        self.stroke_color = RootSignal(style.colors(color if stroke_color is None else stroke_color), COLORS, name=f"{prefix}.stroke_color")
        self.stroke_width = RootSignal(stroke_width, SCALARS, name=f"{prefix}.stroke_width")
        self.stroke_opacity = RootSignal(stroke_opacity, SCALARS, name=f"{prefix}.stroke_opacity")
        self.stroke_joint = RootSignal(style.joints(stroke_joint), DISCRETE, name=f"{prefix}.stroke_joint")
        self.stroke_cap = RootSignal(style.cap(stroke_cap), DISCRETE, name=f"{prefix}.stroke_cap")
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")
        self._styles: tuple[tuple[object, Fill | None], tuple[object, Stroke | None]] = ((None, None), (None, None))
        """The last snapshot's (token, Fill) and (token, Stroke)."""

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (
            self.points, self.fill_color, self.fill_opacity, self.stroke_color, self.stroke_width,
            self.stroke_opacity, self.stroke_joint, self.stroke_cap, self.cast_shadows,
        )  # fmt: skip

    @animatable
    def set_fill(self, color: Colors | None = None, opacity: Scalars | None = None) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.fill_color, ABS, style.colors(color)))
        if opacity is not None:
            changes.append((self.fill_opacity, ABS, opacity))
        return changes

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
        return [(self.fill_color, ABS, style.colors(color)), (self.stroke_color, ABS, style.colors(color))]

    @animatable
    def set_cast_shadows(self, cast_shadows: bool) -> ChangeList:
        """Whether this shape blocks light for lit meshes behind it."""
        return [(self.cast_shadows, ABS, cast_shadows)]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alphas = (*alphas, self.alpha)
        fill_token = version_token(self.fill_color, self.fill_opacity, *alphas)
        stroke_token = version_token(
            self.stroke_color, self.stroke_width, self.stroke_opacity, self.stroke_joint, self.stroke_cap, *alphas
        )
        # Fill and Stroke are immutable values of the signals in their
        # token: while the token holds, the previous value is still right.
        cached_fill, cached_stroke = self._styles
        if cached_fill[0] != fill_token or cached_stroke[0] != stroke_token:
            alpha = math.prod(signal.value for signal in alphas)
            if cached_fill[0] != fill_token:
                cached_fill = (fill_token, style.fill(self.fill_color.value, self.fill_opacity.value, alpha))
            if cached_stroke[0] != stroke_token:
                cached_stroke = (
                    stroke_token,
                    style.stroke(
                        self.stroke_color.value, self.stroke_opacity.value, self.stroke_width.value,
                        self.stroke_joint.value, self.stroke_cap.value, alpha,
                    ),  # fmt: skip
                )
            self._styles = (cached_fill, cached_stroke)
        return RShape2D(
            uid=self.uid,
            points=self.points.value,
            fill=cached_fill[1],
            stroke=cached_stroke[1],
            cast_shadows=bool(self.cast_shadows.value),
            transform=self._transform(),
            versions={"points": version_token(self.points), "fill": fill_token, "stroke": stroke_token},
        )


class Circle(Shape2D):
    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(arc_points(radius), name, **style)


class Dot(Shape2D):
    def __init__(self, radius: float = 0.08, name: str | None = None, **style: Any) -> None:
        style.setdefault("fill_opacity", 1.0)
        style.setdefault("stroke_width", 0.0)
        super().__init__(arc_points(radius), name, **style)


class Rectangle(Shape2D):
    def __init__(self, width: float = 4.0, height: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(rectangle_points(width, height), name, **style)


class Square(Shape2D):
    def __init__(self, side_length: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(rectangle_points(side_length, side_length), name, **style)


class RegularPolygon(Shape2D):
    def __init__(self, n: int = 6, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(regular_polygon_points(n, radius), name, **style)


class Polyline(Shape2D):
    """Straight segments through ``corners`` (closed into a polygon if asked)."""

    def __init__(self, corners: Sequence[Sequence[float]], closed: bool = False, name: str | None = None, **style: Any) -> None:
        super().__init__(RShape2D.from_corners(corners, closed=closed).points, name, **style)


class Triangle(RegularPolygon):
    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(3, radius, name, **style)


__all__ = ["Circle", "Dot", "Polyline", "Rectangle", "RegularPolygon", "Shape2D", "Square", "Triangle"]
