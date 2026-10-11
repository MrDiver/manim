"""Lines, arrows and angle marks, each with its parameters as signals.

A Line's ``start`` and ``end`` are points with one signal per axis, like
a position: ``line.animate.put_start_and_end_on(a, b)`` moves them, and
``line.connect(start=dot)`` keeps an end on a node or on a point of a
path (``circle.point_at(0.25)``) at every frame. Ends are in the line's
own frame; Shape2D draws in its XY plane, so z is ignored.

Arrows are Groups of a line and tips (Shape2D proxies whose points are
derived from the arrow's ends), so ``arrow.line`` and ``arrow.tip`` can be
styled and animated on their own. Angle and RightAngle are derived from
two lines and follow them.

No class picks a color: the tips take the arrow's ``color`` (or
``stroke_color``) as their fill.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..animation import (
    SCALAR,
    DerivedSignal,
    Group,
    Node,
    RootSignal,
    Shape2D,
    Signal,
    Vector3Signal,
)
from ..animation.authoring import current
from ..animation.nodes.node import ABS, ChangeList, animatable, follow_point
from ..animation.signals.paths import PointSignal, partial_path
from .geometry import arc_between_points, corner_points, dashes_points, tip_points

TIP_LENGTH = 0.35
"""The default length of an arrow's tip (Manim's)."""

Point = Sequence[float]


class _Ends:
    """``start`` and ``end`` as points with a signal per axis, and the
    changes that move them (shared by Line and the arrows)."""

    start: Vector3Signal
    end: Vector3Signal
    name: str

    def _init_ends(self, start: Point, end: Point) -> None:
        self.start = Vector3Signal(_point(start), name=f"{self.name}.start")
        self.end = Vector3Signal(_point(end), name=f"{self.name}.end")

    @property
    def ends(self) -> tuple[Signal, ...]:
        """The six axis signals, start first (the inputs of the points)."""
        return (*self.start.axes, *self.end.axes)

    @animatable
    def put_start_and_end_on(self, start: Point, end: Point) -> ChangeList:
        return _moves(self.start, start) + _moves(self.end, end)

    @animatable
    def set_start(self, start: Point) -> ChangeList:
        return _moves(self.start, start)

    @animatable
    def set_end(self, end: Point) -> ChangeList:
        return _moves(self.end, end)

    def connect(self, start: Node | PointSignal | None = None, end: Node | PointSignal | None = None) -> Any:
        """Keep the ends on nodes' positions or on points (such as
        ``shape.point_at(t)``) from now on; they can no longer be moved
        by changes."""
        if start is not None:
            follow_point(self.start, start, name=f"{self.name}.start", caller=f"{type(self).__name__}.connect()")
        if end is not None:
            follow_point(self.end, end, name=f"{self.name}.end", caller=f"{type(self).__name__}.connect()")
        return self

    def get_start(self) -> np.ndarray:
        """The start after everything recorded so far."""
        return np.array([current(axis) for axis in self.start.axes])

    def get_end(self) -> np.ndarray:
        """The end after everything recorded so far."""
        return np.array([current(axis) for axis in self.end.axes])

    def get_length(self) -> float:
        return float(np.linalg.norm(self.get_end() - self.get_start()))

    def get_angle(self) -> float:
        """The direction from start to end, counterclockwise from +x."""
        x, y, _ = self.get_end() - self.get_start()
        return math.atan2(y, x)


class Line(_Ends, Shape2D):
    """A straight line from ``start`` to ``end``, ``buff`` short of both."""

    def __init__(self, start: Point = (-1.0, 0.0, 0.0), end: Point = (1.0, 0.0, 0.0), buff: float = 0.0, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self._init_ends(start, end)
        self.buff = float(buff)
        self._derive_points(_LinePoints(self.buff), *self.ends)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from self.ends


class DashedLine(Line):
    """A Line drawn in dashes ``dash_length`` long; ``dashed_ratio`` of
    its length is drawn. The dashes stay whole as the line changes."""

    def __init__(
        self,
        start: Point = (-1.0, 0.0, 0.0),
        end: Point = (1.0, 0.0, 0.0),
        dash_length: float = 0.05,
        dashed_ratio: float = 0.5,
        buff: float = 0.0,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(start, end, buff, name, **style)
        self.dash_length, self.dashed_ratio = float(dash_length), float(dashed_ratio)
        self._derive_points(_DashedPoints(self.buff, self.dash_length, self.dashed_ratio), *self.ends)


class Arrow(_Ends, Group):
    """A line from ``start`` to ``end`` with a tip at the end, ``buff``
    short of both: a Group of ``line`` and ``tip`` (``tips``: all of them).

    The tip is ``tip_length`` long and ``tip_width`` wide (default: as
    wide as long), but at most ``max_tip_length_to_length_ratio`` of the
    arrow, so short arrows get small tips. ``style`` is the line's; the
    tips are filled with its color.
    """

    def __init__(
        self,
        start: Point = (-1.0, 0.0, 0.0),
        end: Point = (1.0, 0.0, 0.0),
        buff: float = 0.25,
        tip_length: float = TIP_LENGTH,
        tip_width: float | None = None,
        max_tip_length_to_length_ratio: float = 0.25,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        self._init_ends(start, end)
        tip = _Tip(float(buff), float(tip_length), float(tip_length if tip_width is None else tip_width), float(max_tip_length_to_length_ratio))
        self.line = Shape2D((), f"{self.name}.line", **style)
        self.line._derive_points(_ArrowLine(tip, self._tip_count() == 2), *self.ends)
        self.tips = [Shape2D((), f"{self.name}.tip", **_tip_style(style))]
        self.tips[0]._derive_points(_TipPoints(tip, at_start=False), *self.ends)
        if self._tip_count() == 2:
            start_tip = Shape2D((), f"{self.name}.start_tip", **_tip_style(style))
            start_tip._derive_points(_TipPoints(tip, at_start=True), *self.ends)
            self.tips.append(start_tip)
        self.add(self.line, *self.tips)

    @property
    def tip(self) -> Shape2D:
        """The tip at the end."""
        return self.tips[0]

    def _tip_count(self) -> int:
        return 1

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from self.ends


class DoubleArrow(Arrow):
    """An Arrow with a tip at both ends (``tips[1]`` at the start)."""

    def _tip_count(self) -> int:
        return 2


class Vector(Arrow):
    """An Arrow from the origin to ``direction``, with no buff."""

    def __init__(self, direction: Point = (1.0, 0.0, 0.0), name: str | None = None, **options: Any) -> None:
        options.setdefault("buff", 0.0)
        super().__init__((0.0, 0.0, 0.0), direction, name=name, **options)


class CurvedArrow(_Ends, Group):
    """An arrow along the arc from ``start`` to ``end`` turning by
    ``angle`` (counterclockwise if positive), its tip along the arc: a
    Group of ``line`` (the arc) and ``tip``."""

    def __init__(
        self,
        start: Point = (-1.0, 0.0, 0.0),
        end: Point = (1.0, 0.0, 0.0),
        angle: float = math.pi / 4,
        tip_length: float = TIP_LENGTH,
        tip_width: float | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(name=name)
        self._init_ends(start, end)
        self.angle = float(angle)
        tip = _Tip(0.0, float(tip_length), float(tip_length if tip_width is None else tip_width), 0.25)
        self.line = Shape2D((), f"{self.name}.line", **style)
        self.line._derive_points(_ArcArrowLine(tip, self.angle), *self.ends)
        self.tip = Shape2D((), f"{self.name}.tip", **_tip_style(style))
        self.tip._derive_points(_ArcTipPoints(tip, self.angle), *self.ends)
        self.add(self.line, self.tip)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from self.ends


class ArcBetweenPoints(_Ends, Shape2D):
    """The arc from ``start`` to ``end`` turning by ``angle``
    (counterclockwise if positive, a straight line at 0); the ends move
    and connect like a Line's."""

    def __init__(self, start: Point = (-1.0, 0.0, 0.0), end: Point = (1.0, 0.0, 0.0), angle: float = math.pi / 2, name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self._init_ends(start, end)
        self.angle = RootSignal(float(angle), SCALAR, name=f"{self.name}.angle")
        self._derive_points(_ArcBetween(), *self.ends, self.angle)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (*self.ends, self.angle)

    @animatable
    def set_angle(self, angle: float) -> ChangeList:
        return [(self.angle, ABS, float(angle))]


class Angle(Shape2D):
    """The arc marking the angle between two Lines, ``radius`` from where
    they cross, counterclockwise from ``first`` to ``second`` (the other
    way round with ``other_angle``). ``quadrant`` picks, per line, the
    side of the crossing: 1 towards its end, -1 towards its start.

    It follows the lines; ``radians`` is the angle as a signal (to show
    with a DecimalNumber). Positions of the lines count, their rotations
    and scales do not.
    """

    def __init__(
        self,
        first: Line,
        second: Line,
        radius: float = 0.4,
        other_angle: bool = False,
        quadrant: tuple[int, int] = (1, 1),
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__((), name, **style)
        self.first, self.second = first, second
        corner = _Corner(tuple(quadrant), bool(other_angle))
        inputs = _corner_inputs(first, second)
        self.radians = DerivedSignal(corner.radians, inputs, SCALAR, name=f"{self.name}.radians")
        self._derive_points(_AnglePoints(corner, float(radius)), *inputs)


class RightAngle(Shape2D):
    """The square mark of a right angle between two Lines, sides
    ``length`` long, where they cross (``quadrant`` as in Angle)."""

    def __init__(self, first: Line, second: Line, length: float = 0.3, quadrant: tuple[int, int] = (1, 1), name: str | None = None, **style: Any) -> None:
        super().__init__((), name, **style)
        self.first, self.second = first, second
        self._derive_points(_ElbowPoints(_Corner(tuple(quadrant), False), float(length)), *_corner_inputs(first, second))


class Elbow(Shape2D):
    """A corner mark: two sides ``width`` long meeting at a right angle,
    turned by ``angle`` (opening towards +x and +y at 0)."""

    def __init__(self, width: float = 0.2, angle: float = 0.0, name: str | None = None, **style: Any) -> None:
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        corners = np.array([(width, 0.0), (width, width), (0.0, width)]) @ rotation.T
        super().__init__(corner_points(corners), name, **style)


# --------------------------------------------------------------------------
# Derivations: points from the ends (classes, so they print and compare)
# --------------------------------------------------------------------------


def _ends(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """``(start, end)`` in the plane from the six axis values."""
    return np.array(values[0:2], dtype=np.float64), np.array(values[3:5], dtype=np.float64)


def _shortened(start: np.ndarray, end: np.ndarray, at_start: float, at_end: float) -> tuple[np.ndarray, np.ndarray]:
    """The segment cut short by the given lengths (never past its middle)."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return start, end
    unit = direction / length
    at_start, at_end = min(at_start, length / 2), min(at_end, length / 2)
    return start + unit * at_start, end - unit * at_end


class _LinePoints:
    __slots__ = ("buff",)

    def __init__(self, buff: float) -> None:
        self.buff = buff

    def __call__(self, *values: float) -> np.ndarray:
        return corner_points(_shortened(*_ends(values), self.buff, self.buff))


class _DashedPoints:
    __slots__ = ("buff", "dash_length", "dashed_ratio")

    def __init__(self, buff: float, dash_length: float, dashed_ratio: float) -> None:
        self.buff, self.dash_length, self.dashed_ratio = buff, dash_length, dashed_ratio

    def __call__(self, *values: float) -> np.ndarray:
        start, end = _shortened(*_ends(values), self.buff, self.buff)
        return dashes_points(start, end, self.dash_length, self.dashed_ratio)


class _ArcBetween:
    __slots__ = ()

    def __call__(self, *values: float) -> np.ndarray:
        return arc_between_points(*_ends(values[:6]), values[6])


class _Tip:
    """An arrow's buff and tip size, and the tip length for a length."""

    __slots__ = ("buff", "length", "width", "max_ratio")

    def __init__(self, buff: float, length: float, width: float, max_ratio: float) -> None:
        self.buff, self.length, self.width, self.max_ratio = buff, length, width, max_ratio

    def size(self, arrow_length: float) -> tuple[float, float]:
        """``(length, width)`` of the tip on an arrow ``arrow_length`` long."""
        length = min(self.length, self.max_ratio * arrow_length)
        return length, self.width * (length / self.length if self.length else 0.0)


class _ArrowLine:
    """The line of an arrow: from the start (or its tip) to the base of the
    tip, half a tip into it so no gap shows between them."""

    __slots__ = ("tip", "double")

    def __init__(self, tip: _Tip, double: bool) -> None:
        self.tip, self.double = tip, double

    def __call__(self, *values: float) -> np.ndarray:
        start, end = _shortened(*_ends(values), self.tip.buff, self.tip.buff)
        tip_length, _ = self.tip.size(float(np.linalg.norm(end - start)))
        return corner_points(_shortened(start, end, tip_length / 2 if self.double else 0.0, tip_length / 2))


class _TipPoints:
    __slots__ = ("tip", "at_start")

    def __init__(self, tip: _Tip, at_start: bool) -> None:
        self.tip, self.at_start = tip, at_start

    def __call__(self, *values: float) -> np.ndarray:
        start, end = _shortened(*_ends(values), self.tip.buff, self.tip.buff)
        if self.at_start:
            start, end = end, start
        length, width = self.tip.size(float(np.linalg.norm(end - start)))
        return tip_points(end, end - start, length, width)


class _ArcArrowLine:
    """The arc of a CurvedArrow, short of the tip's base by half a tip."""

    __slots__ = ("tip", "angle")

    def __init__(self, tip: _Tip, angle: float) -> None:
        self.tip, self.angle = tip, angle

    def __call__(self, *values: float) -> np.ndarray:
        arc = arc_between_points(*_ends(values), self.angle)
        length = _arc_length(arc)
        tip_length, _ = self.tip.size(length)
        return partial_path(arc, 0.0, 1.0 - tip_length / 2 / length) if length > 0 else arc


class _ArcTipPoints:
    """The tip of a CurvedArrow: at the end, along the arc's last chord of
    a tip's length."""

    __slots__ = ("tip", "angle")

    def __init__(self, tip: _Tip, angle: float) -> None:
        self.tip, self.angle = tip, angle

    def __call__(self, *values: float) -> np.ndarray:
        arc = arc_between_points(*_ends(values), self.angle)
        length = _arc_length(arc)
        tip_length, width = self.tip.size(length)
        base = partial_path(arc, 0.0, 1.0 - tip_length / length)[-1] if length > 0 else arc[-1]
        return tip_points(arc[-1], arc[-1] - base, tip_length, width)


def _arc_length(arc: np.ndarray) -> float:
    """The length of an arc of equal curves, from its anchors' chords
    (exact enough for the tip's place)."""
    from ..animation.signals.paths import point_at

    samples = np.array([point_at(arc, t, by_length=False) for t in np.linspace(0.0, 1.0, 33)])
    return float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())


class _Corner:
    """Where two lines cross and the directions of the chosen sides."""

    __slots__ = ("quadrant", "other_angle")

    def __init__(self, quadrant: tuple[int, int], other_angle: bool) -> None:
        self.quadrant, self.other_angle = quadrant, other_angle

    def solve(self, values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(crossing, first direction, second direction)`` from the two
        lines' ends and positions (18 values, see _corner_inputs)."""
        v = np.asarray(values, dtype=np.float64)
        offset_1, offset_2 = v[6:8], v[15:17]
        a1, b1 = v[0:2] + offset_1, v[3:5] + offset_1
        a2, b2 = v[9:11] + offset_2, v[12:14] + offset_2
        d1, d2 = (b1 - a1) * self.quadrant[0], (b2 - a2) * self.quadrant[1]
        matrix = np.column_stack([b1 - a1, -(b2 - a2)])
        if abs(np.linalg.det(matrix)) < 1e-12:
            crossing = a1  # parallel: no crossing
        else:
            t, _ = np.linalg.solve(matrix, a2 - a1)
            crossing = a1 + t * (b1 - a1)
        return crossing, d1 / (np.linalg.norm(d1) or 1.0), d2 / (np.linalg.norm(d2) or 1.0)

    def angles(self, values: Sequence[float]) -> tuple[np.ndarray, float, float]:
        """``(crossing, start angle, turn)`` of the arc from the first side
        to the second."""
        crossing, d1, d2 = self.solve(values)
        start = math.atan2(d1[1], d1[0])
        turn = (math.atan2(d2[1], d2[0]) - start) % math.tau
        if self.other_angle:
            turn -= math.tau
        return crossing, start, turn

    def radians(self, *values: float) -> float:
        return abs(self.angles(values)[2])


class _AnglePoints:
    __slots__ = ("corner", "radius")

    def __init__(self, corner: _Corner, radius: float) -> None:
        self.corner, self.radius = corner, radius

    def __call__(self, *values: float) -> np.ndarray:
        from .geometry import arc_points

        crossing, start, turn = self.corner.angles(values)
        return arc_points(self.radius, start, turn, max(1, math.ceil(abs(turn) / (math.tau / 8) - 1e-9))) + crossing


class _ElbowPoints:
    __slots__ = ("corner", "length")

    def __init__(self, corner: _Corner, length: float) -> None:
        self.corner, self.length = corner, length

    def __call__(self, *values: float) -> np.ndarray:
        crossing, d1, d2 = self.corner.solve(values)
        return corner_points([crossing + d1 * self.length, crossing + (d1 + d2) * self.length, crossing + d2 * self.length])


def _corner_inputs(first: Line, second: Line) -> tuple[Signal, ...]:
    """The signals an angle mark depends on: each line's ends and position."""
    return (*first.ends, *first.position.axes, *second.ends, *second.position.axes)


def _tip_style(style: dict[str, Any]) -> dict[str, Any]:
    """A tip's style from its arrow's: filled with the line's color."""
    color = style.get("stroke_color", style.get("color"))
    tip: dict[str, Any] = {"fill_opacity": 1.0, "stroke_width": 0.0}
    if color is not None:
        tip["color"] = color
    if "stroke_opacity" in style:
        tip["fill_opacity"] = style["stroke_opacity"]
    return tip


def _point(value: Point) -> tuple[float, float, float]:
    array = np.zeros(3)
    given = np.asarray(value, dtype=np.float64).reshape(-1)
    array[: len(given)] = given
    return (float(array[0]), float(array[1]), float(array[2]))


def _moves(point: Vector3Signal, value: Point) -> ChangeList:
    return [(axis, ABS, component) for axis, component in zip(point.axes, _point(value), strict=True)]


__all__ = ["TIP_LENGTH", "Angle", "ArcBetweenPoints", "Arrow", "CurvedArrow", "DashedLine", "DoubleArrow", "Elbow", "Line", "RightAngle", "Vector"]
