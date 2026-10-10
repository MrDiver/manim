"""Marks around other nodes: rectangles, underlines, crosses and braces.

Each follows the node it marks through :func:`bounds`, a signal of the
node's bounding box derived from the points of its paths and every
transform on the way to them. So a rectangle around a formula stays
around it while the formula moves, grows or changes shape.

Bounds are in the space of the marked node's parent: put a mark next to
the node it marks (in the same group, or both in the scene). The family
is the one when the mark is made; nodes added to a group later are not
counted. No mark picks a color.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import quaternion

from ..animation import (
    ARRAY,
    ClippingGroup,
    Curve3D,
    DerivedSignal,
    Group,
    Node,
    PositionMixin,
    RotationMixin,
    ScalableMixin,
    Shape2D,
    Signal,
)
from .geometry import corner_points, rectangle_points, rounded_rectangle_points
from .shapes2d import Brace, _Side


def bounds(node: Node) -> DerivedSignal:
    """The bounding box ``[[min x, min y, min z], [max x, max y, max z]]``
    of the paths under ``node`` (Shape2D and Curve3D), in its parent's
    space, as an ARRAY signal that follows them (all zeros if none)."""
    layout: list[tuple[int, tuple[str, ...], bool]] = []
    inputs: list[Signal] = []
    index: dict[int, int] = {}

    def visit(member: Node, parent: int) -> None:
        parts = tuple(part for part, mixin in (("position", PositionMixin), ("rotation", RotationMixin), ("scale", ScalableMixin)) if isinstance(member, mixin))
        for part in parts:
            inputs.extend(getattr(member, part).axes if part != "rotation" else (member.rotation,))  # type: ignore[attr-defined]
        is_path = isinstance(member, (Shape2D, Curve3D))
        if is_path:
            inputs.append(member.points)  # type: ignore[attr-defined]
        index[id(member)] = len(layout)
        layout.append((parent, parts, is_path))
        for child in _children(member):
            visit(child, index[id(member)])

    visit(node, -1)
    return DerivedSignal(_Bounds(tuple(layout)), tuple(inputs), ARRAY, name=f"{node.name}.bounds")


def edge(node: Node, direction: Sequence[float]) -> DerivedSignal:
    """The side of ``node``'s bounds facing ``direction`` (UP, DOWN, LEFT
    or RIGHT) as a segment signal, counterclockwise like
    :meth:`Rectangle.edge`: ``Brace(edge(formula, DOWN))``."""
    side = _Side(direction)
    return DerivedSignal(_BoundsSide(side), (bounds(node),), ARRAY, name=f"{node.name}.edge{side.direction}")


class SurroundingRectangle(Shape2D):
    """A rectangle around ``node``, ``buff`` away from its bounds, with
    corners rounded by ``corner_radius``."""

    def __init__(self, node: Node, buff: float = 0.1, corner_radius: float = 0.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.node, self.buff, self.corner_radius = node, float(buff), float(corner_radius)
        self._derive_points(_Around(self.buff, self.corner_radius), bounds(node))


class BackgroundRectangle(SurroundingRectangle):
    """A filled rectangle behind ``node`` (add it before the node): no
    outline and ``fill_opacity`` 0.75 unless given; the color is yours."""

    def __init__(self, node: Node, buff: float = 0.0, corner_radius: float = 0.0, name: str | None = None, **style: Any) -> None:
        style.setdefault("fill_opacity", 0.75)
        style.setdefault("stroke_width", 0.0)
        super().__init__(node, buff, corner_radius, name, **style)


class Underline(Shape2D):
    """A line under ``node``, ``buff`` below its bounds, as wide as they are."""

    def __init__(self, node: Node, buff: float = 0.1, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.node, self.buff = node, float(buff)
        self._derive_points(_Under(self.buff), bounds(node))


class Cross(Shape2D):
    """Two lines from corner to corner of ``node``'s bounds, ``buff``
    beyond them (one subpath each)."""

    def __init__(self, node: Node, buff: float = 0.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.node, self.buff = node, float(buff)
        self._derive_points(_Cross(self.buff), bounds(node))


class BraceLabel(Group):
    """A Brace along the side of ``node`` facing ``direction`` with
    ``label`` (any node, e.g. Typst or a DecimalNumber) beyond its tip:
    a Group of ``brace`` and ``label``. ``brace_style`` styles the brace."""

    def __init__(
        self,
        node: Node,
        label: Node,
        direction: Sequence[float] = (0.0, -1.0, 0.0),
        buff: float = 0.1,
        label_buff: float = 0.15,
        name: str | None = None,
        **brace_style: Any,
    ) -> None:
        super().__init__(name=name)
        self.brace = Brace(edge(node, direction), buff=buff, **brace_style)
        self.label = self.brace.attach(label, label_buff)
        self.add(self.brace, self.label)


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------


class _Bounds:
    """The bounding box of a family from its transforms and points; the
    layout lists, per node in family order, its parent's index, its
    transform parts and whether it is a path (see bounds())."""

    __slots__ = ("layout",)

    def __init__(self, layout: tuple[tuple[int, tuple[str, ...], bool], ...]) -> None:
        self.layout = layout

    def __call__(self, *values: Any) -> np.ndarray:
        matrices: list[np.ndarray] = []
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        cursor = 0
        for parent, parts, is_path in self.layout:
            position, rotation, scale = np.zeros(3), quaternion.one, np.ones(3)
            for part in parts:
                if part == "position":
                    position, cursor = np.asarray(values[cursor : cursor + 3], dtype=np.float64), cursor + 3
                elif part == "rotation":
                    rotation, cursor = values[cursor], cursor + 1
                else:
                    scale, cursor = np.asarray(values[cursor : cursor + 3], dtype=np.float64), cursor + 3
            local = np.eye(4)
            local[:3, :3] = quaternion.as_rotation_matrix(rotation) * scale
            local[:3, 3] = position
            matrix = matrices[parent] @ local if parent >= 0 else local
            matrices.append(matrix)
            if is_path:
                points, cursor = np.asarray(values[cursor], dtype=np.float64), cursor + 1
                if len(points):
                    full = np.zeros((len(points), 3))
                    full[:, : points.shape[1]] = points
                    moved = full @ matrix[:3, :3].T + matrix[:3, 3]
                    low, high = np.minimum(low, moved.min(axis=0)), np.maximum(high, moved.max(axis=0))
        return np.array([low, high]) if np.isfinite(low).all() else np.zeros((2, 3))


class _BoundsSide:
    __slots__ = ("side",)

    def __init__(self, side: _Side) -> None:
        self.side = side

    def __call__(self, box: np.ndarray) -> np.ndarray:
        low, high = box[0, :2], box[1, :2]
        width, height = high - low
        return self.side(width, height) + (low + high) / 2


class _Around:
    __slots__ = ("buff", "corner_radius")

    def __init__(self, buff: float, corner_radius: float) -> None:
        self.buff, self.corner_radius = buff, corner_radius

    def __call__(self, box: np.ndarray) -> np.ndarray:
        low, high = box[0, :2], box[1, :2]
        width, height = high - low + 2 * self.buff
        outline = rounded_rectangle_points(width, height, self.corner_radius) if self.corner_radius > 0 else rectangle_points(width, height)
        return outline + (low + high) / 2


class _Under:
    __slots__ = ("buff",)

    def __init__(self, buff: float) -> None:
        self.buff = buff

    def __call__(self, box: np.ndarray) -> np.ndarray:
        y = box[0, 1] - self.buff
        return corner_points([(box[0, 0], y), (box[1, 0], y)])


class _Cross:
    __slots__ = ("buff",)

    def __init__(self, buff: float) -> None:
        self.buff = buff

    def __call__(self, box: np.ndarray) -> np.ndarray:
        (x0, y0), (x1, y1) = box[0, :2] - self.buff, box[1, :2] + self.buff
        return np.concatenate([corner_points([(x0, y1), (x1, y0)]), corner_points([(x0, y0), (x1, y1)])])


def _children(node: Node) -> list[Node]:
    if isinstance(node, Group):
        return list(node.children)
    if isinstance(node, ClippingGroup):
        return [node.content]
    return []


__all__ = ["BackgroundRectangle", "BraceLabel", "Cross", "SurroundingRectangle", "Underline", "bounds", "edge"]
