"""Shape2D: a vector shape in its local plane.

A :class:`Shape2D` snapshots to an :class:`~manimx.robjects.RShape2D`: a
path with fill and stroke. It is a leaf; to attach other nodes to it, put
them in a :class:`~manimx.animation.Group` together. Shapes built on it
(Circle, Square, Polyline, ...) are userland (manimx.userland.shapes2d).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from ...robjects import CapStyle, JointType, Texture
from ...utils.color import WHITE
from .. import authoring
from ..signals.kinds import COLORS, DISCRETE, PATH, SCALARS
from ..signals.paths import PointSignal, drawn_part, point_signal
from ..signals.signal import DerivedSignal, RootSignal, Signal
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


class Shape2D(PositionMixin, RotationMixin, ScalableMixin, Node):
    """A path in the local XY plane with fill and stroke.

    Every field of its render counterpart is a signal:

    ==================  ========  ==========================================
    signal              kind      RShape2D field
    ==================  ========  ==========================================
    ``points``          PATH      ``points`` (the part ``drawn_points``)
    ``fill_color``      COLORS    ``fill.color`` (one color or a gradient)
    ``fill_opacity``    SCALARS   ``fill.opacity``
    ``fill_image``      DISCRETE  ``fill.image`` (a Texture, or None)
    ``stroke_color``    COLORS    ``stroke.color``
    ``stroke_opacity``  SCALARS   ``stroke.opacity``
    ``stroke_width``    SCALARS   ``stroke.width``
    ``stroke_joint``    DISCRETE  ``stroke.joint`` (one, or one per curve)
    ``stroke_cap``      DISCRETE  ``stroke.cap``
    ``cast_shadows``    DISCRETE  ``cast_shadows``
    ==================  ========  ==========================================

    Lists (gradients, tapering widths) are spread over the path's points
    by the renderer. ``color`` sets fill and stroke color together. The
    fill is transparent unless ``fill_opacity`` is given. A ``fill_image``
    is tinted by the fill color and opacity (white and 1 show it as it
    is) and stretched over the shape's local box around its origin, as far
    as the points reach either way (a centered shape's bounds).

    ``draw_start`` and ``draw_end`` (SCALAR, 0 and 1) say which part of the
    path is drawn, as fractions of its curves: Create and Write animate
    them, so they never change ``points`` (or a shape's parameters).
    ``drawn_points`` is derived from the three and is what the snapshot
    draws.
    """

    render_kind = "shape2d"

    def __init__(
        self,
        points: Any = (),
        name: str | None = None,
        *,
        color: Colors = WHITE,
        fill_color: Colors | None = None,
        fill_opacity: Scalars = 0.0,
        fill_image: Texture | None = None,
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
        self.draw_start, self.draw_end, self.drawn_points = drawn_part(self.points, prefix)
        self.fill_color = RootSignal(style.colors(color if fill_color is None else fill_color), COLORS, name=f"{prefix}.fill_color")
        self.fill_opacity = RootSignal(fill_opacity, SCALARS, name=f"{prefix}.fill_opacity")
        self.fill_image = RootSignal(fill_image, DISCRETE, name=f"{prefix}.fill_image")
        self.stroke_color = RootSignal(style.colors(color if stroke_color is None else stroke_color), COLORS, name=f"{prefix}.stroke_color")
        self.stroke_width = RootSignal(stroke_width, SCALARS, name=f"{prefix}.stroke_width")
        self.stroke_opacity = RootSignal(stroke_opacity, SCALARS, name=f"{prefix}.stroke_opacity")
        self.stroke_joint = RootSignal(style.joints(stroke_joint), DISCRETE, name=f"{prefix}.stroke_joint")
        self.stroke_cap = RootSignal(style.cap(stroke_cap), DISCRETE, name=f"{prefix}.stroke_cap")
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")
        """The last snapshot's (token, Fill) and (token, Stroke)."""

    def unpack(self) -> Shape2D:
        """A plain Shape2D like this one is now: its points (free, no longer
        derived from parameters such as a Circle's radius), style, transform
        and alpha, as authoring code sees them. It is a new node, not in the
        scene: put it where this one was (``self.remove(circle)``,
        ``self.add(plain)``) to bend or morph its points freely."""
        plain = Shape2D(np.array(authoring.current(self.points)), self.name, tags=self.tags)
        for name in _UNPACKED:
            getattr(plain, name).set(authoring.current(getattr(self, name)))
        for mine, its in zip((*plain.position.axes, *plain.scale.axes), (*self.position.axes, *self.scale.axes), strict=True):
            mine.set(authoring.current(its))
        return plain

    def _derive_points(self, function: Callable[..., Any], *parameters: Signal) -> None:
        """Make ``points`` a signal derived from ``parameters`` by
        ``function`` (a pure function of their values): what a shape with
        parameters (a Circle's radius) calls in its constructor. Animating
        a parameter then changes the points at every frame; ``points``
        itself can no longer be changed directly."""
        self.points = DerivedSignal(function, parameters, PATH, name=f"{self.name}.points")
        self.draw_start, self.draw_end, self.drawn_points = drawn_part(self.points, self.name)

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
            self.fill_color, self.fill_opacity, self.fill_image, self.stroke_color, self.stroke_width,
            self.stroke_opacity, self.stroke_joint, self.stroke_cap, self.cast_shadows,
        )  # fmt: skip

    @animatable
    def set_fill(self, color: Colors | None = None, opacity: Scalars | None = None, image: Texture | None = None) -> ChangeList:
        """Change the fill; an ``image`` replaces the fill image (it is
        switched, not blended: cross-fade two shapes to blend images)."""
        changes: ChangeList = []
        if color is not None:
            changes.append((self.fill_color, ABS, style.colors(color)))
        if opacity is not None:
            changes.append((self.fill_opacity, ABS, opacity))
        if image is not None:
            changes.append((self.fill_image, ABS, image))
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

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        yield from super().render_fields()
        # The render object draws the part of the points from draw_start to draw_end.
        yield from (
            ("points", self.drawn_points), ("fill_color", self.fill_color), ("fill_opacity", self.fill_opacity),
            ("fill_image", self.fill_image),
            ("stroke_color", self.stroke_color),
            ("stroke_opacity", self.stroke_opacity), ("stroke_width", self.stroke_width),
            ("stroke_joint", self.stroke_joint), ("stroke_cap", self.stroke_cap), ("cast_shadows", self.cast_shadows),
        )  # fmt: skip


_UNPACKED = (
    "fill_color", "fill_opacity", "fill_image", "stroke_color", "stroke_width", "stroke_opacity", "stroke_joint", "stroke_cap",
    "cast_shadows", "draw_start", "draw_end", "rotation", "alpha",
)  # fmt: skip
"""The signals besides points and the transform axes that unpack() copies."""


__all__ = ["Shape2D"]
