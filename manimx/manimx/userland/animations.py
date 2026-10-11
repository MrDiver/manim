"""The built-in animations: fades, turns, pulses, growing and shrinking.

They are userland: each one only implements the Animation contract
(``targets()``, ``contribution()``, ``nodes()``, ``removes()``) on the
public signals of a node, exactly as a project's own animation would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

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
from ..animation.nodes.node import require
from ..animation.signals.contributions import (
    Between,
    Contribution,
    Partial,
    Power,
    Ramp,
    Turn,
    evaluate,
)
from ..animation.signals.paths import point_at
from ..animation.signals.rates import linear, smooth
from ..animation.signals.signal_tracks import RateFunction
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


MorphTarget = TypeVar("MorphTarget", bound=Node)


@dataclass(eq=False)
class Morph(_PlannedAnimation, Generic[MorphTarget]):
    """Turn ``source`` into ``target`` (Manim's ``ReplacementTransform``).

    Neither side is changed: the morph is drawn by in-between shapes
    (plain Shape2D or Curve3D with free points), one per pair of paths,
    which exist only while it plays. When it ends, ``target`` takes
    ``source``'s place in the scene, so the morph's result is ``target``
    itself, of its own class and with its own parameters::

        square = Square(2.0)
        self.play(Morph(circle, square))
        self.play(square.animate.set_side_length(3.0))

    Paths (``Shape2D`` and ``Curve3D``, in tree order) are paired evenly,
    as Manim pairs submobjects: with more on one side, each path of the
    other side stands for several in a row. A pair's paths are aligned
    once, when the animation is created; its in-between shape goes from
    the source path (where it is, how it looks) to the target path.
    """

    source: Node
    target: MorphTarget
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        self._plan()
        sources, targets = _drawn_paths(self.source), _drawn_paths(self.target)
        if not sources or not targets:
            raise ValueError(f"{type(self).__name__} needs a Shape2D or Curve3D (or a group of them) on both sides")
        self._worlds = {**_world_matrices(self.source), **_world_matrices(self.target)}
        self._inbetweens: list[Node] = []
        self._pair(sources, targets)
        # The in-between shapes show the morph: the source is hidden from
        # its start, the target until its end.
        self._between(self.source.alpha, authoring_current(self.source.alpha), 0.0, 0.0, 0.0)
        self._between(self.target.alpha, 0.0, authoring_current(self.target.alpha), 1.0, 1.0)

    def _pair(self, sources: list[Node], targets: list[Node]) -> None:
        count = max(len(sources), len(targets))
        for index in range(count):
            source = sources[index * len(sources) // count]
            target = targets[index * len(targets) // count]
            self._inbetween(source, target)

    def _inbetween(
        self,
        start: Node,
        end: Node,
        alpha: tuple[float, float] | None = None,
        window: tuple[float, float] = (0.0, 1.0),
        shift: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """An in-between shape going from path ``start`` to path ``end``,
        moved by ``shift`` (world units) at the end; its alpha goes from
        ``alpha[0]`` to ``alpha[1]`` within ``window`` (default: from
        ``start``'s to ``end``'s over the whole morph)."""
        flat = isinstance(start, Shape2D) and isinstance(end, Shape2D)
        dimension = 2 if flat else 3
        here = _resized(_points(start), dimension)
        there = _resized(_points(end), dimension)
        first, last = align_paths(here, there)
        shape: Shape2D | Curve3D = Shape2D(first, f"{start.name}-morph") if flat else Curve3D(first, f"{start.name}-morph")
        self._between(shape.points, first, last)
        for name in _STYLE_SIGNALS:
            if hasattr(start, name) and hasattr(end, name) and hasattr(shape, name):
                begin, finish = authoring_current(getattr(start, name)), authoring_current(getattr(end, name))
                getattr(shape, name).set(begin)
                self._between(getattr(shape, name), begin, finish)
        (start_position, start_rotation, start_scale), (end_position, end_rotation, end_scale) = (
            _decomposed(self._worlds[start.uid]), _decomposed(self._worlds[end.uid])
        )  # fmt: skip
        for axis, begin, finish, offset in zip(shape.position.axes, start_position, end_position, shift, strict=True):
            axis.set(begin)
            self._between(axis, begin, finish + offset)
        for axis, begin, finish in zip(shape.scale.axes, start_scale, end_scale, strict=True):
            axis.set(begin)
            self._between(axis, begin, finish)
        shape.rotation.set(start_rotation)
        self._between(shape.rotation, start_rotation, end_rotation)
        if alpha is None:
            alpha = (_world_alpha(start), _world_alpha(end))
        shape.alpha.set(alpha[0])
        self._between(shape.alpha, alpha[0], alpha[1], *window)
        self._inbetweens.append(shape)

    @property
    def inbetweens(self) -> list[Node]:
        """The shapes that show the morph while it plays."""
        return list(self._inbetweens)

    def nodes(self) -> Sequence[Node]:
        return (self.source, self.target, *self._inbetweens)

    def removes(self) -> Sequence[Node]:
        return (self.source, *self._inbetweens)


@dataclass(eq=False)
class MorphMatching(Morph[MorphTarget]):
    """Turn ``source`` into ``target`` keeping what they share (Manim's
    ``TransformMatchingShapes``): paths of the same shape on both sides
    (up to where they are and how large, e.g. the same glyph of two
    formulas, also as an exponent) move to their new place and size. The
    others cross-fade over the whole morph, as in Manim: the source's fade
    out while drifting towards where the target's appear. Pairs are found
    in tree order, each path used once. Like :class:`Morph`, it changes
    neither side and leaves ``target`` in the scene.

    ``pairs`` says which paths become which, before any matching: each
    ``(source, target)`` names a path of each side by its index among the
    side's paths (in tree order; the preview's Indices toggle shows them
    for a formula) or by its name, if only one path has it (a Typst glyph
    is named by its character). A source may become several targets, and
    several sources one target::

        MorphMatching(two, three, pairs=[("2", "3"), (0, 0), (0, 2)])  # one x becomes two
    """

    tolerance: float = 1e-3
    """How far points may differ for paths to match, as a fraction of
    the path's size."""
    pairs: Sequence[tuple[int | str, int | str]] = ()
    """Paths to morph into each other, before the matching (see above)."""

    def _pair(self, sources: list[Node], targets: list[Node]) -> None:
        chosen = [(_path_of(sources, source, "source"), _path_of(targets, target, "target")) for source, target in self.pairs]
        for source, target in chosen:
            self._inbetween(source, target)
        taken = {path.uid for pair in chosen for path in pair}
        sources = [source for source in sources if source.uid not in taken]
        targets = [target for target in targets if target.uid not in taken]
        free = [(target, self._normalized_points(target)) for target in targets]
        matched: dict[int, Node] = {}
        for source in sources:
            points = self._normalized_points(source)
            target = next((t for t, other in free if t.uid not in {m.uid for m in matched.values()} and self._same(points, other)), None)
            if target is not None:
                matched[source.uid] = target
        gone = [source for source in sources if source.uid not in matched]
        new = [target for target in targets if target not in matched.values()]
        drift = self._center(new) - self._center(gone) if gone and new else np.zeros(3)
        for source in sources:
            if source.uid in matched:
                self._inbetween(source, matched[source.uid])
            else:
                self._inbetween(source, source, (_world_alpha(source), 0.0), shift=drift)
        for target in new:
            self._inbetween(target, target, (0.0, _world_alpha(target)))

    def _world_points(self, path: Node) -> np.ndarray:
        return _in_space_of(_points(path), self._worlds[path.uid], np.eye(4), 3)

    def _normalized_points(self, path: Node) -> np.ndarray:
        """The path's points in world space, relative to their lowest
        corner and divided by their largest extent."""
        points = self._world_points(path)
        if not len(points):
            return points
        points = points - points.min(axis=0)
        size = points.max()
        return points / size if size > 0 else points

    def _center(self, paths: list[Node]) -> np.ndarray:
        """The center of the world bounding box of ``paths``."""
        points = np.vstack([self._world_points(path) for path in paths])
        return (points.min(axis=0) + points.max(axis=0)) / 2 if len(points) else np.zeros(3)

    def _same(self, points: np.ndarray, other: np.ndarray) -> bool:
        return points.shape == other.shape and len(points) > 0 and bool(np.abs(points - other).max() <= self.tolerance)


@dataclass(eq=False)
class MoveAlongPath(Animation):
    """Move a node along ``path`` from its start to its end (Manim's
    ``MoveAlongPath``): the node's position goes through the path's
    points. ``by_length`` as in :meth:`Shape2D.point_at`. The path is
    taken where and as it is when the animation is created; to keep to a
    path that changes, follow ``path.point_at(proportion)`` instead.
    """

    node: Node
    path: Node
    by_length: bool = True
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        require(self.node, PositionMixin, "MoveAlongPath")
        if not isinstance(self.path, (Shape2D, Curve3D)):
            raise TypeError(f"MoveAlongPath needs a Shape2D or Curve3D path, got {type(self.path).__name__}")
        node_world = _world_matrices(self.node)[self.node.uid]
        parent = node_world @ np.linalg.inv(_local_matrix(self.node))
        self._points = _in_space_of(_points(self.path), _world_matrices(self.path)[self.path.uid], parent, 3)
        self._last: tuple[float, np.ndarray] | None = None

    def targets(self) -> Sequence[Target]:
        return [Target(axis, Change.ABSOLUTE) for axis in self.node.position.axes]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        if self._last is None or self._last[0] != alpha:  # the three axes ask for the same point
            self._last = (alpha, point_at(self._points, alpha, self.by_length))
        return float(self._last[1][self.node.position.axes.index(signal)])

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Create(_PlannedAnimation):
    """Draw a node's paths from their start (Manim's ``Create``): the
    shapes of a group one after another, each overlapping the previous by
    ``1 - lag_ratio`` of its time. It animates each path's ``draw_end``,
    never its points, so the shape may change meanwhile.
    """

    node: Node
    lag_ratio: float = 1.0
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        self._plan()
        paths = _drawn_paths(self.node)
        for path, (lower, upper) in zip(paths, _lagged(len(paths), self.lag_ratio), strict=True):
            self._between(path.draw_end, 0.0, 1.0, lower, upper)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Uncreate(_PlannedAnimation):
    """Undraw a node's paths back to their start, the last shape first,
    and remove the node (Manim's ``Uncreate``), through ``draw_end``.
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
            self._between(path.draw_end, authoring_current(path.draw_end), 0.0, lower, upper)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)

    def removes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class DrawBorderThenFill(_PlannedAnimation):
    """Draw each shape's outline, then fade its fill and stroke in (Manim's
    ``DrawBorderThenFill``). The outline is opaque, has width
    ``stroke_width`` and the shape's stroke color, or its fill color when
    it has no stroke.
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
        self._between(path.draw_end, 0.0, 1.0, lower, middle)
        self._between(path.stroke_color, outline_color, authoring_current(path.stroke_color), middle, upper)
        self._between(path.stroke_width, self.stroke_width, width, middle, upper)
        # The outline is opaque even where the shape has no stroke (glyphs).
        self._between(path.stroke_opacity, 1.0, authoring_current(path.stroke_opacity), middle, upper)
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


@dataclass(eq=False)
class Unwrite(Write):
    """Undo :class:`Write` and remove the node (Manim's ``Unwrite``): shape
    after shape from the last one, each fill and stroke fading back to
    its outline, then the outline undrawn. Run time and lag as for Write.
    """

    def _draw_border_then_fill(self, path: Node, window: tuple[float, float]) -> None:
        lower, upper = 1.0 - window[1], 1.0 - window[0]  # the last shape first
        middle = lower + (upper - lower) / 2
        width = authoring_current(path.stroke_width)
        has_stroke = bool(np.any(np.asarray(width) > 0.0))
        outline_color = authoring_current(path.stroke_color if has_stroke or not hasattr(path, "fill_color") else path.fill_color)
        self._between(path.stroke_color, authoring_current(path.stroke_color), outline_color, lower, middle)
        self._between(path.stroke_width, width, self.stroke_width, lower, middle)
        self._between(path.stroke_opacity, authoring_current(path.stroke_opacity), 1.0, lower, middle)
        if hasattr(path, "fill_opacity"):
            self._between(path.fill_opacity, authoring_current(path.fill_opacity), 0.0, lower, middle)
        self._between(path.draw_end, authoring_current(path.draw_end), 0.0, middle, upper)

    def removes(self) -> Sequence[Node]:
        return (self.node,)


_STYLE_SIGNALS = ("fill_color", "fill_opacity", "stroke_color", "stroke_width", "stroke_opacity", "stroke_joint", "stroke_cap")


def _path_of(paths: list[Node], key: int | str, side: str) -> Node:
    """The path ``key`` (an index or a unique name) of a MorphMatching side."""
    if isinstance(key, int):
        if not -len(paths) <= key < len(paths):
            raise IndexError(f"the {side} has {len(paths)} paths, no index {key}")
        return paths[key]
    found = [index for index, path in enumerate(paths) if path.name == key]
    if len(found) != 1:
        where = f"at the indices {found}; give one of them" if found else "none"
        raise ValueError(f"the {side} has {len(found)} paths named {key!r} ({where})")
    return paths[found[0]]


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


def _resized(points: np.ndarray, dimension: int) -> np.ndarray:
    """``points`` with ``dimension`` columns (2D paths lie in z = 0)."""
    if points.shape[1] == dimension:
        return points
    if dimension == 3:
        return np.column_stack([points, np.zeros(len(points))])
    return points[:, :2]


def _decomposed(matrix: np.ndarray) -> tuple[tuple[float, ...], quaternion.quaternion, tuple[float, ...]]:
    """Position, rotation and scale of a world matrix without shear."""
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    rotation = matrix[:3, :3] / np.where(scale > 1e-12, scale, 1.0)
    if np.linalg.det(rotation) < 0:  # a mirror: put it in the scale
        scale[0], rotation[:, 0] = -scale[0], -rotation[:, 0]
    return tuple(matrix[:3, 3]), quaternion.from_rotation_matrix(rotation), tuple(scale)


def _world_alpha(node: Node) -> float:
    """``node``'s alpha times that of every group above it."""
    alpha = authoring_current(node.alpha)
    for ancestor in _ancestors(node):
        alpha *= authoring_current(ancestor.alpha)
    return float(alpha)


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
    """World matrices of ``root`` and everything below it, by node uid:
    through the groups that hold ``root`` in the scene, if any.
    """
    world = np.eye(4)
    for ancestor in _ancestors(root):
        world = world @ _local_matrix(ancestor)
    worlds: dict[int, np.ndarray] = {}

    def visit(node: Node, parent: np.ndarray) -> None:
        worlds[node.uid] = parent @ _local_matrix(node)
        for child in _children(node):
            visit(child, worlds[node.uid])

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
    "MoveAlongPath",
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
    "Unwrite",
    "Write",
]
