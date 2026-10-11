"""Animations that point at something: flashes, frames drawn around it,
wiggles and waves.

Like the built-in animations (animations.py) they are userland: each
implements the Animation contract on public signals. The ones that draw
a mark of their own (Flash, Circumscribe) make it when they are created,
where the marked node is then, and remove it when they end. None picks a
color: ``style`` reaches the marks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..animation import (
    Animation,
    Change,
    Group,
    Node,
    PositionMixin,
    RootSignal,
    RotationMixin,
    ScalableMixin,
    Shape2D,
    Target,
)
from ..animation.authoring import current
from ..animation.nodes.node import require
from ..animation.signals.signal_tracks import RateFunction
from ..robjects.transform import quaternion_from_axis_angle
from .animations import _drawn_paths, _PlannedAnimation, _points
from .constants import UP
from .markers import current_bounds
from .rate_functions import linear, smooth, there_and_back
from .shapes2d import Circle, Rectangle


class _PassingFlash(_PlannedAnimation):
    """A stretch ``time_width`` long (as a fraction of each path) running
    along paths from start to end: the drawn part's end runs ahead and its
    start follows, as Manim's ShowPassingFlash cuts them."""

    def _flash(self, paths: Sequence[Node], time_width: float) -> None:
        width = max(float(time_width), 1e-6)
        for path in paths:
            # Drawn from start = clip(upper - width) to end = clip(upper),
            # upper = alpha (1 + width): both move linearly in alpha.
            self._between(path.draw_end, 0.0, 1.0, 0.0, 1.0 / (1.0 + width))
            self._between(path.draw_start, 0.0, 1.0, width / (1.0 + width), 1.0)


@dataclass(eq=False)
class ShowPassingFlash(_PassingFlash):
    """Draw only a stretch ``time_width`` of each of a node's paths,
    running from their starts to their ends, and remove the node. Play it
    on a copy or a mark (Circumscribe does) to keep the original."""

    node: Node
    time_width: float = 0.1
    run_time: float = 1.0
    rate: RateFunction = smooth

    def __post_init__(self) -> None:
        self._plan()
        self._flash(_drawn_paths(self.node), self.time_width)

    def nodes(self) -> Sequence[Node]:
        return (self.node,)

    def removes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class Flash(_PassingFlash):
    """Lines bursting out of ``point`` and gone: ``num_lines`` of them,
    ``line_length`` long, starting ``flash_radius`` from the point. The
    lines are ``lines`` (a Group) while it plays; ``style`` styles them."""

    point: Sequence[float] = (0.0, 0.0, 0.0)
    line_length: float = 0.2
    num_lines: int = 12
    flash_radius: float = 0.1
    time_width: float = 1.0
    run_time: float = 1.0
    rate: RateFunction = smooth
    style: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._plan()
        center = np.zeros(3)
        center[: len(self.point)] = self.point
        rays = []
        for index in range(self.num_lines):
            angle = math.tau * index / self.num_lines
            direction = np.array([math.cos(angle), math.sin(angle)])
            start, end = direction * self.flash_radius, direction * (self.flash_radius + self.line_length)
            rays.append(Shape2D(np.array([start, (2 * start + end) / 3, (start + 2 * end) / 3, end]), f"flash.ray{index}", **self.style))
        self.lines = Group(*rays, name="flash").move_to(center)
        self._flash(rays, self.time_width)

    def nodes(self) -> Sequence[Node]:
        return (self.lines,)

    def removes(self) -> Sequence[Node]:
        return (self.lines,)


@dataclass(eq=False)
class Circumscribe(_PassingFlash):
    """Run a stroke once around a node: a rectangle (``shape="rectangle"``)
    or a circle (``"circle"``) ``buff`` outside its bounding box, drawn as
    a passing flash ``time_width`` of it long. The mark is ``frame`` while
    it plays; ``style`` styles it."""

    node: Node
    shape: str = "rectangle"
    buff: float = 0.1
    time_width: float = 0.3
    run_time: float = 1.0
    rate: RateFunction = smooth
    style: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shape not in ("rectangle", "circle"):
            raise ValueError(f"Circumscribe draws a 'rectangle' or a 'circle', not {self.shape!r}")
        self._plan()
        low, high = current_bounds(self.node)
        size, center = high - low, (low + high) / 2
        if self.shape == "rectangle":
            self.frame: Shape2D = Rectangle(size[0] + 2 * self.buff, size[1] + 2 * self.buff, name=f"{self.node.name}.circumscribe", **self.style)
        else:
            self.frame = Circle(float(np.linalg.norm(size[:2])) / 2 + self.buff, name=f"{self.node.name}.circumscribe", **self.style)
        self.frame.move_to(center)
        self._flash([self.frame], self.time_width)

    def nodes(self) -> Sequence[Node]:
        return (self.frame,)

    def removes(self) -> Sequence[Node]:
        return (self.frame,)


@dataclass(eq=False)
class Wiggle(Animation):
    """Wiggle a node: it grows by ``scale_value`` and back while turning
    to and fro by ``rotation_angle`` ``n_wiggles`` times about ``axis``."""

    node: Node
    scale_value: float = 1.1
    rotation_angle: float = 0.01 * math.tau
    n_wiggles: int = 6
    axis: Sequence[float] = (0.0, 0.0, 1.0)
    run_time: float = 2.0
    rate: RateFunction = linear

    def __post_init__(self) -> None:
        require(self.node, ScalableMixin, "Wiggle")
        require(self.node, RotationMixin, "Wiggle")

    def targets(self) -> Sequence[Target]:
        return [*(Target(axis, Change.RELATIVE) for axis in self.node.scale.axes), Target(self.node.rotation, Change.RELATIVE)]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        swell = there_and_back(alpha)
        if signal is self.node.rotation:
            return quaternion_from_axis_angle(self.axis, self.rotation_angle * swell * math.sin(self.n_wiggles * math.tau * alpha))
        return 1.0 + (self.scale_value - 1.0) * swell

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


@dataclass(eq=False)
class ApplyWave(Animation):
    """Send a wave through a node from left to right: points are lifted
    along ``direction`` by up to ``amplitude`` where the wave is, a bump
    ``time_width`` of the node wide, and all are back at the end. Paths
    whose points follow other signals are lifted as a whole instead."""

    node: Node
    direction: Sequence[float] = tuple(UP)
    amplitude: float = 0.2
    time_width: float = 0.5
    run_time: float = 2.0
    rate: RateFunction = linear

    def __post_init__(self) -> None:
        self._paths = _drawn_paths(self.node)
        low, high = current_bounds(self.node)
        self._left, self._width = float(low[0]), float(high[0] - low[0]) or 1.0
        self._lift = np.zeros(3)
        self._lift[: len(self.direction)] = self.direction
        self._targets: dict[int, tuple[str, Node, Any]] = {}
        for path in self._paths:
            offset = np.array([current(axis) for axis in path.position.axes]) if isinstance(path, PositionMixin) else np.zeros(3)
            if isinstance(path.points, RootSignal):
                points = _points(path)
                self._targets[path.points.id] = ("points", path, (points, points[:, 0] + offset[0]))
            elif isinstance(path, PositionMixin):
                box = current_bounds(path)
                for index, axis in enumerate(path.position.axes):
                    self._targets[axis.id] = ("position", path, (index, float((box[0, 0] + box[1, 0]) / 2)))
            else:
                require(path, PositionMixin, "ApplyWave")

    def _height(self, x: np.ndarray | float, alpha: float) -> np.ndarray | float:
        """The wave's lift at ``x`` (in the node's parent's space) at ``alpha``."""
        across = (np.asarray(x) - self._left) / self._width
        phase = np.clip((alpha * (1.0 + self.time_width) - across) / self.time_width, 0.0, 1.0)
        return self.amplitude * np.sin(np.pi * phase)

    def targets(self) -> Sequence[Target]:
        changes = {"points": Change.ABSOLUTE, "position": Change.RELATIVE}
        return [Target(self._signal(kind, path, data), changes[kind]) for kind, path, data in self._targets.values()]

    @staticmethod
    def _signal(kind: str, path: Node, data: Any) -> RootSignal:
        return path.points if kind == "points" else path.position.axes[data[0]]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        kind, path, data = self._targets[signal.id]
        if kind == "points":
            points, xs = data
            lift = np.asarray(self._height(xs, alpha))[:, None] * self._lift[: points.shape[1]]
            return points + lift
        index, x = data
        return float(self._height(x, alpha)) * self._lift[index]

    def nodes(self) -> Sequence[Node]:
        return (self.node,)


__all__ = ["ApplyWave", "Circumscribe", "Flash", "ShowPassingFlash", "Wiggle"]
