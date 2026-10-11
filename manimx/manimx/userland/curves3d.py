"""Curves built on Curve3D (userland, like manimx.userland.shapes2d)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..animation import (
    ARRAY,
    PATH,
    Curve3D,
    DerivedSignal,
    Node,
    PositionMixin,
    PreparedScene,
    RootSignal,
)
from ..animation.authoring import active_compiler
from ..animation.signals.paths import PointSignal, drawn_part
from .geometry import parametric_points


class ParametricCurve3D(Curve3D):
    """A smooth curve through ``function(t)`` (a 3D point) at ``samples``
    evenly spaced ``t`` in ``t_range`` (see geometry.parametric_points)."""

    def __init__(
        self,
        function: Callable[[float], Any],
        t_range: tuple[float, float] = (0.0, 1.0),
        samples: int = 64,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(parametric_points(function, t_range, samples), name, **style)


class TracedPath(Curve3D):
    """The trail of a moving point: where ``point`` (a node's position or a
    PointSignal such as ``shape.point_at(0.5)``) has been since the trail
    was made, or over the last ``dissipating_time`` seconds.

    Every frame is independent of the frames before it (scenes are
    evaluated at any time in any order), so the trail cannot be collected
    while playing. Instead, once the scene is prepared, the point is
    sampled ``samples_per_second`` times per second of the whole scene
    (:meth:`after_prepare`); the trail at a time is those samples up to
    it, ending exactly at the point. Between samples it is straight, so
    fast curvy motion needs more samples.

    The points are in the space the point is given in (a node's parent):
    add the trail to the same parent. Make it inside ``construct``: it
    starts at the scene time it is made.
    """

    def __init__(
        self,
        point: PointSignal | Node,
        dissipating_time: float | None = None,
        samples_per_second: float = 60.0,
        name: str | None = None,
        **style: Any,
    ) -> None:
        compiler = active_compiler()
        if compiler is None:
            raise RuntimeError("make a TracedPath inside construct(): it starts at the scene time it is made")
        if samples_per_second <= 0.0:
            raise ValueError("a TracedPath needs a positive sample rate")
        if dissipating_time is not None and dissipating_time <= 0.0:
            raise ValueError("dissipating_time must be positive (None: keep the whole trail)")
        if isinstance(point, PositionMixin):
            axes = point.position.axes
        elif isinstance(point, PointSignal):
            axes = point.axes
        else:
            raise TypeError(f"TracedPath traces a node's position or a PointSignal, not {point!r}")
        super().__init__((), name, **style)
        self.axes = axes
        self.start = compiler.scene_time()
        self.dissipating_time = dissipating_time
        self.samples_per_second = float(samples_per_second)
        self.samples = RootSignal(np.zeros((0, 4)), ARRAY, name=f"{self.name}.samples")
        """``(scene time, x, y, z)`` rows of the point after the start,
        set by :meth:`after_prepare`."""
        trail = _Trail(self.start, dissipating_time)
        self.points = DerivedSignal(trail, (compiler.time.scene_time, self.samples, *axes), PATH, name=f"{self.name}.points")
        self.draw_start, self.draw_end, self.drawn_points = drawn_part(self.points, self.name)

    def signals(self) -> Any:
        yield from super().signals()
        yield self.samples

    def after_prepare(self, scene: PreparedScene) -> bool:
        """Sample the point over the whole scene into :attr:`samples`."""
        plan = scene.plan
        rows: list[list[float]] = []
        count = max(2, round(scene.duration * self.samples_per_second) + 1)
        for t in np.linspace(0.0, scene.duration, count):
            plan.evaluate(float(t))
            time = scene.playing_time_at(float(t))
            if time < self.start or (rows and time <= rows[-1][0]):
                continue  # before the trail, or holding at an event
            rows.append([time, *(float(plan.value(axis)) for axis in self.axes)])
        self.samples.set(np.array(rows) if rows else np.zeros((0, 4)))
        return True


class _Trail:
    """A TracedPath's points from the scene time, its samples and where
    the point is now."""

    __slots__ = ("dissipating", "start")

    def __init__(self, start: float, dissipating: float | None) -> None:
        self.start, self.dissipating = start, dissipating

    def __call__(self, time: float, samples: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
        if time < self.start:
            return np.zeros((0, 3))
        earliest = self.start if self.dissipating is None else max(self.start, time - self.dissipating)
        times = samples[:, 0] if len(samples) else np.zeros(0)
        corners = np.concatenate([samples[(times >= earliest) & (times < time), 1:], [[x, y, z]]])
        if len(corners) < 2:
            return np.zeros((0, 3))
        starts, ends = corners[:-1], corners[1:]
        curves = np.stack([starts, (2 * starts + ends) / 3, (starts + 2 * ends) / 3, ends], axis=1)
        return curves.reshape(-1, 3)


__all__ = ["ParametricCurve3D", "TracedPath"]
