"""The built-in animations: fades, turns, pulses, growing and shrinking.

They are userland: each one only implements the Animation contract
(``targets()``, ``contribution()``, ``nodes()``, ``removes()``) on the
public signals of a node, exactly as a project's own animation would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import quaternion

from ..animation import (
    Animation,
    Change,
    ClippingGroup,
    Curve3D,
    Group,
    Node,
    PositionMixin,
    RootSignal,
    RotationMixin,
    ScalableMixin,
    Shape2D,
    Target,
)
from ..animation.authoring import active_compiler
from ..animation.authoring import current as authoring_current
from ..animation.contributions import (
    Between,
    Contribution,
    Partial,
    Power,
    Ramp,
    Turn,
    evaluate,
)
from ..animation.node import require
from ..animation.rates import linear, smooth
from ..animation.signal_tracks import RateFunction
from ..robjects import Transform
from ..robjects.transform import quaternion_from_axis_angle
from .geometry import align_paths
from .rate_functions import double_smooth


@dataclass(eq=False)
class FadeIn(Animation):
    """Fade a node in: its opacity multiplier goes from 0 to 1, optionally
    while it shifts in from ``shift`` away."""

    node: Node
    shift: Sequence[float] | None = None
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        if self.shift is not None:
            require(self.node, PositionMixin, "FadeIn(shift=...)")

    def targets(self) -> Sequence[Target]:
        targets = [Target(self.node.alpha, Change.ABSOLUTE)]
        if self.shift is not None:
            targets += [Target(axis, Change.RELATIVE) for axis in self.node.position.axes]
        return targets

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        if signal is self.node.alpha:
            return alpha
        offset = self.shift[self.node.position.axes.index(signal)]
        return -offset * (1.0 - alpha)  # arrive at the authored position

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        if signal is self.node.alpha:
            return Ramp(1.0)
        return Ramp(float(-self.shift[self.node.position.axes.index(signal)]), remaining=True)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class FadeOut(Animation):
    """Fade a node out (opacity multiplier from 1 to 0) and remove it."""

    node: Node
    shift: Sequence[float] | None = None
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        if self.shift is not None:
            require(self.node, PositionMixin, "FadeOut(shift=...)")

    def targets(self) -> Sequence[Target]:
        targets = [Target(self.node.alpha, Change.ABSOLUTE)]
        if self.shift is not None:
            targets += [Target(axis, Change.RELATIVE) for axis in self.node.position.axes]
        return targets

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        if signal is self.node.alpha:
            return 1.0 - alpha
        return self.shift[self.node.position.axes.index(signal)] * alpha

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        if signal is self.node.alpha:
            return Ramp(1.0, remaining=True)
        return Ramp(float(self.shift[self.node.position.axes.index(signal)]))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)

    def removes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Rotate(Animation):
    """Rotate by ``angle`` about ``axis``. Unlike ``animate.rotate`` (which
    slerps along the shortest way), this sweeps the full angle, so a turn by
    2 pi spins once around."""

    node: Node
    angle: float
    axis: Sequence[float] = (0.0, 0.0, 1.0)
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        require(self.node, RotationMixin, "Rotate")

    def targets(self) -> Sequence[Target]:
        return [Target(self.node.rotation, Change.RELATIVE)]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> quaternion.quaternion:
        return quaternion_from_axis_angle(self.axis, self.angle * alpha)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        x, y, z = (float(c) for c in self.axis)
        return Turn((x, y, z), float(self.angle))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Indicate(Animation):
    """Briefly scale a node up and back (Manim's Indicate, without the
    color flash)."""

    node: Node
    scale_factor: float = 1.2
    run_time: float = 1.0
    rate: RateFunction = staticmethod(lambda t: float(np.sin(np.pi * t)))  # type: ignore[assignment]

    def __post_init__(self) -> None:
        require(self.node, ScalableMixin, "Indicate")

    def targets(self) -> Sequence[Target]:
        return [Target(axis, Change.RELATIVE) for axis in self.node.scale.axes]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        return float(self.scale_factor**alpha)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return Power(float(self.scale_factor))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class GrowFromCenter(Animation):
    """Grow a node from nothing (scale 0) to its authored scale."""

    node: Node
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        require(self.node, ScalableMixin, "GrowFromCenter")
        # The authored scale at the moment the animation is created.
        self._target = tuple(authoring_current(axis) for axis in self.node.scale.axes)

    def targets(self) -> Sequence[Target]:
        return [Target(axis, Change.ABSOLUTE) for axis in self.node.scale.axes]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        return self._target[self.node.scale.axes.index(signal)] * alpha

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return Ramp(float(self._target[self.node.scale.axes.index(signal)]))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class ShrinkToCenter(Animation):
    """Shrink a node to nothing and remove it."""

    node: Node
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        require(self.node, ScalableMixin, "ShrinkToCenter")

    def targets(self) -> Sequence[Target]:
        return [Target(axis, Change.ABSOLUTE) for axis in self.node.scale.axes]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        return start_value * (1.0 - alpha)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return Ramp(None, remaining=True)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)

    def removes(self) -> Sequence[Node]:
        return (self.node,)




# --------------------------------------------------------------------------
# Morphing, creating and writing paths
# --------------------------------------------------------------------------


class _PlannedAnimation(Animation):
    """An animation defined entirely by data forms (``Between``,
    ``Partial``): one per signal it changes, all absolute. Its
    ``contribution()`` evaluates them in Python; the core evaluates the
    same data in Rust.
    """

    def _plan(self) -> None:
        self._planned: dict[int, tuple[RootSignal, Between | Partial]] = {}

    def _set(self, signal: RootSignal, planned: Between | Partial) -> None:
        self._planned[signal.id] = (signal, planned)

    def _between(self, signal: RootSignal, start: Any, end: Any, lower: float = 0.0, upper: float = 1.0) -> None:
        kind = signal.kind
        self._set(signal, Between(kind.validate(start), kind.validate(end), lower, upper))

    def targets(self) -> Sequence[Target]:
        return [Target(signal, Change.ABSOLUTE) for signal, _ in self._planned.values()]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        return evaluate(self._planned[signal.id][1], signal.kind, alpha)

    def planned_contribution(self, signal: RootSignal) -> Contribution:
        return self._planned[signal.id][1]


@dataclass(eq=False)
class Morph(_PlannedAnimation):
    """Turn ``source`` into ``target`` (Manim's ``ReplacementTransform``).

    The paths of their shapes (``Shape2D`` and ``Curve3D``, in tree order)
    are paired evenly, as Manim pairs submobjects: with more shapes on one
    side, each shape of the other side stands for several in a row. A
    pair's paths are aligned once, when the animation is created, and then
    interpolated point by point together with their colors, opacities and
    widths. A target shape that shares its source with an earlier one
    starts as a copy of that source; a source shape that shares its
    target with an earlier one ends as a copy of that target. When the
    morph ends, ``target`` takes ``source``'s place in the scene; it looks
    exactly like the morphed source, so the switch does not show.
    """

    source: Node
    target: Node
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        self._plan()
        sources, targets = _drawn_paths(self.source), _drawn_paths(self.target)
        if not sources or not targets:
            raise ValueError("Morph needs a Shape2D or Curve3D (or a group of them) on both sides")
        worlds = {**_world_matrices(self.source), **_world_matrices(self.target)}
        paired: set[int] = set()
        if len(targets) >= len(sources):
            for index, target in enumerate(targets):
                source = sources[index * len(sources) // len(targets)]
                if id(source) in paired:
                    self._move(target, source, target, worlds)  # a copy of the source turns into it
                else:
                    paired.add(id(source))
                    self._morph_pair(source, target, worlds)
        else:
            for index, source in enumerate(sources):
                target = targets[index * len(targets) // len(sources)]
                if id(target) in paired:
                    self._move(source, source, target, worlds)  # it ends as a copy of the target
                else:
                    paired.add(id(target))
                    self._morph_pair(source, target, worlds)

    def _morph_pair(self, source: Node, target: Node, worlds: dict[int, np.ndarray]) -> None:
        self._move(source, source, target, worlds)
        # The target stays hidden until it replaces the source at the end.
        self._between(target.alpha, 0.0, authoring_current(target.alpha), 1.0, 1.0)

    def _move(self, moving: Node, start: Node, end: Node, worlds: dict[int, np.ndarray]) -> None:
        """Animate ``moving``'s path and style from those of ``start`` to
        those of ``end`` (``moving`` is one of them)."""
        here = _in_space_of(_points(start), worlds[id(start)], worlds[id(moving)], _points(moving).shape[1])
        there = _in_space_of(_points(end), worlds[id(end)], worlds[id(moving)], _points(moving).shape[1])
        first, last = align_paths(here, there)
        self._between(moving.points, first, last)
        for name in _STYLE_SIGNALS:
            if hasattr(start, name) and hasattr(end, name) and hasattr(moving, name):
                self._between(getattr(moving, name), authoring_current(getattr(start, name)), authoring_current(getattr(end, name)))

    def nodes(self) -> Sequence[Node]:
        return (self.source, self.target)

    def removes(self) -> Sequence[Node]:
        return (self.source,)


@dataclass(eq=False)
class MorphMatching(Morph):
    """Turn ``source`` into ``target`` keeping what they share (Manim's
    ``TransformMatchingShapes``): shapes that look the same on both sides
    (the same path up to where it is, e.g. the same glyph of a formula)
    move to their new place; the others fade out in the first half and in
    in the second, so the two never pile up. Pairs are found in tree
    order, each shape used once.
    """

    tolerance: float = 1e-3
    """How far (world units) points may differ for shapes to match."""

    def __post_init__(self) -> None:
        self._plan()
        sources, targets = _drawn_paths(self.source), _drawn_paths(self.target)
        if not sources or not targets:
            raise ValueError("MorphMatching needs a Shape2D or Curve3D (or a group of them) on both sides")
        worlds = {**_world_matrices(self.source), **_world_matrices(self.target)}
        free = [(target, self._relative_points(target, worlds)) for target in targets]
        matched: set[int] = set()
        for source in sources:
            points = self._relative_points(source, worlds)
            target = next((t for t, other in free if id(t) not in matched and self._same(points, other)), None)
            if target is not None:
                matched.add(id(target))
                self._morph_pair(source, target, worlds)
            else:
                self._between(source.alpha, authoring_current(source.alpha), 0.0, 0.0, 0.5)
        for target in targets:
            if id(target) not in matched:
                self._between(target.alpha, 0.0, authoring_current(target.alpha), 0.5, 1.0)

    @staticmethod
    def _relative_points(path: Node, worlds: dict[int, np.ndarray]) -> np.ndarray:
        """The path's points in world space, relative to their lowest corner."""
        points = _in_space_of(_points(path), worlds[id(path)], np.eye(4), 3)
        return points - points.min(axis=0) if len(points) else points

    def _same(self, points: np.ndarray, other: np.ndarray) -> bool:
        return points.shape == other.shape and len(points) > 0 and bool(np.abs(points - other).max() <= self.tolerance)


@dataclass(eq=False)
class Create(_PlannedAnimation):
    """Draw a node's paths from their start (Manim's ``Create``): the
    shapes of a group one after another, each overlapping the previous by
    ``1 - lag_ratio`` of its time.
    """

    node: Node
    lag_ratio: float = 1.0
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        self._plan()
        paths = _drawn_paths(self.node)
        for path, (lower, upper) in zip(paths, _lagged(len(paths), self.lag_ratio), strict=True):
            self._set(path.points, Partial(_points(path), lower, upper))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Uncreate(_PlannedAnimation):
    """Undraw a node's paths back to their start, the last shape first,
    and remove the node (Manim's ``Uncreate``).
    """

    node: Node
    lag_ratio: float = 1.0
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        self._plan()
        paths = _drawn_paths(self.node)
        windows = _lagged(len(paths), self.lag_ratio)[::-1]
        for path, (lower, upper) in zip(paths, windows, strict=True):
            self._set(path.points, Partial(_points(path), lower, upper, reverse=True))

    def nodes(self) -> Sequence[Node]:
        return (self.node,)

    def removes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class DrawBorderThenFill(_PlannedAnimation):
    """Draw each shape's outline, then fade its fill and stroke in (Manim's
    ``DrawBorderThenFill``). The outline has width ``stroke_width`` and the
    shape's stroke color, or its fill color when it has no stroke.
    """

    node: Node
    run_time: float = 2.0
    rate: RateFunction = double_smooth
    lag_ratio: float = 0.0
    stroke_width: float = 2.0

    def __post_init__(self) -> None:
        self._plan()
        paths = _drawn_paths(self.node)
        for path, window in zip(paths, _lagged(len(paths), self.lag_ratio), strict=True):
            self._draw_border_then_fill(path, window)

    def _draw_border_then_fill(self, path: Node, window: tuple[float, float]) -> None:
        lower, upper = window
        middle = lower + (upper - lower) / 2
        width = authoring_current(path.stroke_width)
        has_stroke = bool(np.any(np.asarray(width) > 0.0))
        outline_color = authoring_current(path.stroke_color if has_stroke or not hasattr(path, "fill_color") else path.fill_color)
        self._set(path.points, Partial(_points(path), lower, middle))
        self._between(path.stroke_color, outline_color, authoring_current(path.stroke_color), middle, upper)
        self._between(path.stroke_width, self.stroke_width, width, middle, upper)
        if hasattr(path, "fill_opacity"):
            self._between(path.fill_opacity, 0.0, authoring_current(path.fill_opacity), middle, upper)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Write(DrawBorderThenFill):
    """Write a node like a hand would (Manim's ``Write``):
    :class:`DrawBorderThenFill` shape after shape, at a constant speed. The
    lag and the run time follow from the number of shapes, as in Manim:
    1 second below 15 shapes, else 2.
    """

    run_time: float | None = None  # type: ignore[assignment]
    rate: RateFunction = linear
    lag_ratio: float | None = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        count = len(_drawn_paths(self.node))
        if self.run_time is None:
            self.run_time = 1.0 if count < 15 else 2.0
        if self.lag_ratio is None:
            self.lag_ratio = min(4.0 / max(1.0, count), 0.2)
        super().__post_init__()


_STYLE_SIGNALS = ("fill_color", "fill_opacity", "stroke_color", "stroke_width", "stroke_opacity", "stroke_joint", "stroke_cap")


def _drawn_paths(node: Node) -> list[Node]:
    """The Shape2D and Curve3D nodes of ``node``'s family, in tree order."""
    return [member for member in node.family() if isinstance(member, (Shape2D, Curve3D))]


def _points(path: Node) -> np.ndarray:
    return np.asarray(authoring_current(path.points))


def _lagged(count: int, lag_ratio: float) -> list[tuple[float, float]]:
    """The stretch of the progress each of ``count`` parts plays in, as
    Manim lags them: part ``i`` starts ``i * lag_ratio`` of a part's time
    after the first.
    """
    length = (count - 1) * lag_ratio + 1
    return [(index * lag_ratio / length, (index * lag_ratio + 1) / length) for index in range(count)]


def _children(node: Node) -> list[Node]:
    if isinstance(node, Group):
        return list(node.children)
    if isinstance(node, ClippingGroup):
        return [node.mask, node.content]
    return []


def _local_matrix(node: Node) -> np.ndarray:
    """``node``'s transform as authoring code sees it right now."""
    parts: dict[str, Any] = {}
    if isinstance(node, PositionMixin):
        parts["position"] = tuple(authoring_current(axis) for axis in node.position.axes)
    if isinstance(node, RotationMixin):
        parts["rotation"] = authoring_current(node.rotation)
    if isinstance(node, ScalableMixin):
        parts["scale"] = tuple(authoring_current(axis) for axis in node.scale.axes)
    return Transform(**parts).matrix()


def _world_matrices(root: Node) -> dict[int, np.ndarray]:
    """World matrices of ``root`` and everything below it, by ``id``:
    through the groups that hold ``root`` in the scene, if any.
    """
    world = np.eye(4)
    for ancestor in _ancestors(root):
        world = world @ _local_matrix(ancestor)
    worlds: dict[int, np.ndarray] = {}

    def visit(node: Node, parent: np.ndarray) -> None:
        worlds[id(node)] = parent @ _local_matrix(node)
        for child in _children(node):
            visit(child, worlds[id(node)])

    visit(root, world)
    return worlds


def _ancestors(node: Node) -> list[Node]:
    """The groups above ``node`` in the scene being composed, top first."""
    compiler = active_compiler()

    def search(current: Node, path: list[Node]) -> list[Node] | None:
        if current is node:
            return path
        for child in _children(current):
            found = search(child, [*path, current])
            if found is not None:
                return found
        return None

    for root in compiler.roots if compiler is not None else ():
        found = search(root, [])
        if found is not None:
            return found
    return []


def _in_space_of(points: np.ndarray, from_world: np.ndarray, to_world: np.ndarray, dimension: int) -> np.ndarray:
    """``points`` of a node with world matrix ``from_world`` in the local
    space of a node with ``to_world`` (2D paths lie in z = 0).
    """
    points_3d = np.column_stack([points, np.zeros(len(points))]) if points.shape[1] == 2 else points
    matrix = np.linalg.inv(to_world) @ from_world
    moved = points_3d @ matrix[:3, :3].T + matrix[:3, 3]
    return moved[:, :dimension]

__all__ = [
    "Create",
    "DrawBorderThenFill",
    "FadeIn",
    "FadeOut",
    "GrowFromCenter",
    "Indicate",
    "Morph",
    "MorphMatching",
    "Rotate",
    "ShrinkToCenter",
    "Uncreate",
    "Write",
]
