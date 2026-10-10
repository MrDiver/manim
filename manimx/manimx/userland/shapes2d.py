"""Shapes built on Shape2D, each with its parameters as signals.

A shape here is a proxy for its parameters: ``Circle.radius``,
``Rectangle.width`` and so on are signals, and the shape's ``points`` are
derived from them (Shape2D._derive_points), so animating a parameter
reshapes the path at every frame::

    circle = Circle(1.0)
    self.play(circle.animate.set_radius(2.0))

They are userland: a project defines its own shapes the same way, with
the point generators of manimx.userland.geometry. No shape picks a color.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..animation import (
    ARRAY,
    DISCRETE,
    SCALAR,
    DerivedSignal,
    Node,
    PositionMixin,
    RootSignal,
    Shape2D,
    Signal,
    SignalKind,
    animatable,
)
from ..animation.authoring import current
from ..animation.node import ABS, ChangeList, require
from .geometry import (
    arc_points,
    brace_points,
    corner_points,
    rectangle_points,
    regular_polygon_points,
)


def parameter(node: Shape2D, field: str, value: Any, kind: SignalKind = SCALAR) -> RootSignal:
    """A parameter signal of ``node`` named ``<node>.<field>``."""
    return RootSignal(value, kind, name=f"{node.name}.{field}")


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


class Circle(Shape2D):
    """A circle of ``radius`` around the origin."""

    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.radius = parameter(self, "radius", _positive("radius", radius))
        self._derive_points(arc_points, self.radius)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.radius

    @animatable
    def set_radius(self, radius: float) -> ChangeList:
        return [(self.radius, ABS, _positive("radius", radius))]


class Dot(Circle):
    """A small filled circle without an outline."""

    def __init__(self, radius: float = 0.08, name: str | None = None, **style: Any) -> None:
        style.setdefault("fill_opacity", 1.0)
        style.setdefault("stroke_width", 0.0)
        super().__init__(radius, name, **style)


class Rectangle(Shape2D):
    """A ``width`` by ``height`` rectangle centered on the origin."""

    def __init__(self, width: float = 4.0, height: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.width = parameter(self, "width", _positive("width", width))
        self.height = parameter(self, "height", _positive("height", height))
        self._derive_points(rectangle_points, self.width, self.height)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.width, self.height)

    @animatable
    def set_size(self, width: float | None = None, height: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if width is not None:
            changes.append((self.width, ABS, _positive("width", width)))
        if height is not None:
            changes.append((self.height, ABS, _positive("height", height)))
        return changes

    def edge(self, direction: Sequence[float]) -> DerivedSignal:
        """The side facing ``direction`` (UP, DOWN, LEFT or RIGHT) as a
        segment signal ``[start, end]`` that follows ``width`` and
        ``height``, in the rectangle's own frame. The sides run
        counterclockwise, so a Brace on one points away from the
        rectangle: ``Brace(rectangle.edge(DOWN))``."""
        side = _Side(direction)
        return DerivedSignal(side, (self.width, self.height), ARRAY, name=f"{self.name}.edge{side.direction}")


class Square(Shape2D):
    """A square with sides ``side_length`` long, centered on the origin."""

    def __init__(self, side_length: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.side_length = parameter(self, "side_length", _positive("side_length", side_length))
        self._derive_points(_square_points, self.side_length)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.side_length

    @animatable
    def set_side_length(self, side_length: float) -> ChangeList:
        return [(self.side_length, ABS, _positive("side_length", side_length))]


class RegularPolygon(Shape2D):
    """A regular polygon with ``n`` corners on a circle of ``radius``;
    odd ones have a corner pointing up."""

    def __init__(self, n: int = 6, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.n = parameter(self, "n", _corners(n), DISCRETE)
        self.radius = parameter(self, "radius", _positive("radius", radius))
        self._derive_points(regular_polygon_points, self.n, self.radius)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.n, self.radius)

    @animatable
    def set_radius(self, radius: float) -> ChangeList:
        return [(self.radius, ABS, _positive("radius", radius))]


class Triangle(RegularPolygon):
    """An equilateral triangle on a circle of ``radius``, a corner up."""

    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(3, radius, name, **style)


class Polyline(Shape2D):
    """Straight segments through ``corners`` (2D or 3D), closed into a
    polygon if asked. ``corners`` is a signal: moving corners animates
    them one by one (the count stays)."""

    def __init__(self, corners: Sequence[Sequence[float]], closed: bool = False, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.corners = parameter(self, "corners", _corner_array(corners), ARRAY)
        self.closed = parameter(self, "closed", bool(closed), DISCRETE)
        self._derive_points(_polyline_points, self.corners, self.closed)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.corners, self.closed)

    @animatable
    def set_corners(self, corners: Sequence[Sequence[float]]) -> ChangeList:
        array = _corner_array(corners)
        if array.shape != np.shape(self.corners.value):
            raise ValueError(f"{self.name} has {len(self.corners.value)} corners; set_corners keeps the count")
        return [(self.corners, ABS, array)]


class Brace(Shape2D):
    """A curly brace along ``segment`` (``[start, end]``), ``buff`` away
    from it and ``depth`` deep, its tip on the right of the way from
    start to end (below a segment running to the right).

    ``segment`` is a point pair or a segment signal to follow, such as
    :meth:`Rectangle.edge`; following one, the brace is bound to it (it
    has no segment of its own to change). The points are a path for a
    stroke; a tapering ``stroke_width`` list gives the classic look.
    """

    def __init__(
        self,
        segment: Sequence[Sequence[float]] | Signal,
        depth: float = 0.25,
        buff: float = 0.1,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__((), name, **style)
        self.segment = segment if isinstance(segment, Signal) else parameter(self, "segment", _segment(segment), ARRAY)
        self.depth = _positive("depth", depth)
        self.buff = float(buff)
        self._derive_points(_BracePoints(self.depth, self.buff), self.segment)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        if isinstance(self.segment, RootSignal):
            yield self.segment

    @animatable
    def set_segment(self, segment: Sequence[Sequence[float]]) -> ChangeList:
        if not isinstance(self.segment, RootSignal):
            raise TypeError(f"{self.name} follows {self.segment.name}; change that instead")
        return [(self.segment, ABS, _segment(segment))]

    def attach(self, node: Node, buff: float = 0.15) -> Node:
        """Keep ``node`` (centered on its position) ``buff`` beyond the
        brace's tip while the segment moves: its position follows the
        segment, the node's current size keeps it clear of the tip."""
        require(node, PositionMixin, "Brace.attach()")
        reach = self.buff + self.depth + buff
        extent = _half_size(node)
        for axis in range(2):
            label = _LabelAxis(axis, reach, extent)
            setattr(node.position, "xy"[axis], DerivedSignal(label, (self.segment,), SCALAR, name=f"{node.name}.position.{'xy'[axis]}"))  # type: ignore[attr-defined]
        return node


class _Side:
    """The segment of a rectangle's side from its width and height (for
    Rectangle.edge; a class so it compares and prints well)."""

    __slots__ = ("direction",)

    def __init__(self, direction: Sequence[float]) -> None:
        x, y = (float(component) for component in tuple(direction)[:2])
        if (abs(x), abs(y)) not in ((1.0, 0.0), (0.0, 1.0)):
            raise ValueError(f"an edge is UP, DOWN, LEFT or RIGHT, got {tuple(direction)}")
        self.direction = (x, y)

    def __call__(self, width: float, height: float) -> np.ndarray:
        x, y = self.direction
        normal = np.array([x * width / 2, y * height / 2])
        along = np.array([-y * width / 2, x * height / 2])  # counterclockwise
        return np.array([normal - along, normal + along])


class _BracePoints:
    """The points of a Brace along a segment (its derivation)."""

    __slots__ = ("depth", "buff")

    def __init__(self, depth: float, buff: float) -> None:
        self.depth, self.buff = depth, buff

    def __call__(self, segment: np.ndarray) -> np.ndarray:
        start, end = segment[0, :2], segment[1, :2]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return np.zeros((0, 2))
        along = direction / length
        right = np.array([along[1], -along[0]])
        local = brace_points(length, self.depth) - [0.0, self.buff]
        return (start + end) / 2 + local[:, :1] * along + local[:, 1:] * -right


class _LabelAxis:
    """One coordinate of a node kept beyond a Brace's tip (Brace.attach)."""

    __slots__ = ("axis", "reach", "extent")

    def __init__(self, axis: int, reach: float, extent: np.ndarray) -> None:
        self.axis, self.reach, self.extent = axis, reach, extent

    def __call__(self, segment: np.ndarray) -> float:
        start, end = segment[0, :2], segment[1, :2]
        direction = end - start
        length = float(np.linalg.norm(direction)) or 1.0
        right = np.array([direction[1], -direction[0]]) / length
        clearance = float(np.abs(right) @ self.extent)  # the node's half size towards the tip
        return float((start + end)[self.axis] / 2 + right[self.axis] * (self.reach + clearance))


def _half_size(node: Node) -> np.ndarray:
    """Half the width and height of the Shape2D points under ``node`` as
    authoring code sees them (positions of nested nodes count, scales and
    rotations do not)."""
    boxes = []
    for leaf in node.family():
        if isinstance(leaf, Shape2D):
            points = np.asarray(current(leaf.points))
            if len(points):
                offset = np.array([current(axis) for axis in leaf.position.axes[:2]]) if leaf is not node else 0.0
                boxes.append(points[:, :2] + offset)
    if not boxes:
        return np.zeros(2)
    everything = np.vstack(boxes)
    return (everything.max(axis=0) - everything.min(axis=0)) / 2


def _segment(segment: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(segment, dtype=np.float64)
    if array.shape not in ((2, 2), (2, 3)):
        raise ValueError(f"a segment is two 2D or 3D points, got shape {array.shape}")
    return array


def _square_points(side_length: float) -> np.ndarray:
    return rectangle_points(side_length, side_length)


def _polyline_points(corners: np.ndarray, closed: bool) -> np.ndarray:
    return corner_points(corners, closed=closed)


def _corners(n: int) -> int:
    if int(n) != n or n < 3:
        raise ValueError(f"a polygon needs at least 3 corners, got {n}")
    return int(n)


def _corner_array(corners: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(corners, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3) or len(array) < 2:
        raise ValueError(f"corners must be two or more 2D or 3D points, got shape {array.shape}")
    return array


__all__ = ["Brace", "Circle", "Dot", "Polyline", "Rectangle", "RegularPolygon", "Square", "Triangle", "parameter"]
