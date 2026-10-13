"""The light show's animations: what a lighting console plays.

Every one is an ordinary animation: a function of its progress ``u``
(0 at its start, 1 at its end, rate ``linear``), never of seconds. When
it plays and for how long is the timeline's business: the scenes place
them on the music's beat grid with ``wait_until`` and ``play``, so a
``Strobe(light, flashes=32)`` played over four bars flashes on every
eighth note because 32 flashes fit four bars, not because it knows the
tempo.

* :class:`Aim`: a moving head follows ``path(u)``, a world point.
* :class:`Flash`: a light snaps to ``peak`` and settles to ``rest``.
* :class:`Pulse`: ``beats`` flashes, each decaying to ``rest``.
* :class:`Strobe`: ``flashes`` on/off cycles, optionally speeding up.
* :class:`Chase`: one light after another on ``steps`` steps.
* :class:`Fade`: a signal from ``start`` to ``end``.
* :class:`Recolor`: a light through ``colors``, one after another.
* :class:`Rainbow`: a light around the color wheel ``turns`` times.

Paths for :class:`Aim` are functions of progress too: :func:`circle`,
:func:`still`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from manimx.animation import ROTATION, Animation, Change, RootSignal, SpotLight, Target, linear, smooth

from .rig import MovingHead, hue, rgb

Path = Callable[[float], Sequence[float]]
"""A world point for every progress ``u`` in [0, 1]."""


@dataclass(eq=False)
class Aim(Animation):
    """Point ``head`` at ``path(u)``. During the first ``blend`` of its
    progress it turns smoothly from wherever it pointed before."""

    head: MovingHead
    path: Path
    blend: float = 0.15
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def targets(self) -> Sequence[Target]:
        return [Target(self.head.yoke.rotation, Change.ABSOLUTE), Target(self.head.head.rotation, Change.ABSOLUTE)]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        pan, tilt = self.head.pan_tilt(self.path(alpha))
        if alpha < self.blend:
            blend = smooth(alpha / self.blend)
            return [ROTATION.interpolate(start, value, blend) for start, value in zip(start_values, (pan, tilt), strict=True)]
        return [pan, tilt]


@dataclass(eq=False)
class _Level(Animation):
    """An animation of one light's intensity (absolute)."""

    light: SpotLight

    def targets(self) -> Sequence[Target]:
        return [Target(self.light.intensity, Change.ABSOLUTE)]


@dataclass(eq=False)
class Flash(_Level):
    """Snap to ``peak``, settle exponentially to ``rest`` (``decay``: the
    fraction of the run time to fall by e)."""

    peak: float = 1.0
    rest: float = 0.0
    decay: float = 0.3
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[float]:
        return [self.rest + (self.peak - self.rest) * math.exp(-alpha / self.decay)]


@dataclass(eq=False)
class Pulse(_Level):
    """``beats`` flashes of ``peak``, evenly spaced, each decaying to
    ``rest`` (``decay``: the fraction of one beat to fall by e)."""

    beats: int = 4
    peak: float = 1.0
    rest: float = 0.0
    decay: float = 0.3
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[float]:
        within = (alpha * self.beats) % 1.0
        return [self.rest + (self.peak - self.rest) * math.exp(-within / self.decay)]


@dataclass(eq=False)
class Strobe(_Level):
    """``flashes`` cycles of on (``level``) and off (``rest``), on for
    ``duty`` of each. With ``speedup`` > 1 the flashes come faster and
    faster: the last ones ``speedup`` times as fast as the first."""

    flashes: float = 8.0
    level: float = 1.0
    rest: float = 0.0
    duty: float = 0.4
    speedup: float = 1.0
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[float]:
        # The frequency grows linearly from f0 to speedup * f0; the phase
        # is its integral, scaled so that `flashes` fit.
        f0 = self.flashes / (1.0 + (self.speedup - 1.0) / 2.0)
        phase = f0 * (alpha + (self.speedup - 1.0) * alpha**2 / 2.0)
        return [self.level if phase % 1.0 < self.duty else self.rest]


@dataclass(eq=False)
class Chase(Animation):
    """``lights`` light up one after another, one per step, ``steps``
    times in all; the others stay at ``rest``."""

    lights: Sequence[SpotLight]
    steps: int = 16
    level: float = 1.0
    rest: float = 0.0
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def targets(self) -> Sequence[Target]:
        return [Target(light.intensity, Change.ABSOLUTE) for light in self.lights]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[float]:
        lead = min(int(alpha * self.steps), self.steps - 1) % len(self.lights)
        return [self.level if index == lead else self.rest for index in range(len(self.lights))]


@dataclass(eq=False)
class Fade(Animation):
    """``signal`` from ``start`` to ``end`` (``start`` None: from where it is)."""

    signal: RootSignal
    end: float
    start: float | None = None
    run_time: float = 1.0
    rate: Any = staticmethod(smooth)

    def targets(self) -> Sequence[Target]:
        return [Target(self.signal, Change.ABSOLUTE)]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[float]:
        start = start_values[0] if self.start is None else self.start
        return [start + (self.end - start) * alpha]


@dataclass(eq=False)
class Recolor(Animation):
    """``light`` through ``colors``, each for an equal share of the run
    time, switching instantly (like a color wheel in a fixture)."""

    light: SpotLight
    colors: Sequence[str]
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def targets(self) -> Sequence[Target]:
        return [Target(self.light.color, Change.ABSOLUTE)]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        return [rgb(self.colors[min(int(alpha * len(self.colors)), len(self.colors) - 1)])]


@dataclass(eq=False)
class Rainbow(Animation):
    """``light`` around the color wheel ``turns`` times, starting at hue
    ``offset``."""

    light: SpotLight
    turns: float = 1.0
    offset: float = 0.0
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def targets(self) -> Sequence[Target]:
        return [Target(self.light.color, Change.ABSOLUTE)]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        return [hue(self.offset + self.turns * alpha)]


# -- paths -------------------------------------------------------------------------


def circle(center: Sequence[float], radius: float, turns: float, phase: float = 0.0) -> Path:
    """``turns`` times around a circle on the floor (negative: clockwise)."""
    center = np.asarray(center, dtype=np.float64)

    def at(u: float) -> np.ndarray:
        angle = phase + math.tau * turns * u
        return center + radius * np.array([math.cos(angle), math.sin(angle), 0.0])

    return at


def still(point: Sequence[float]) -> Path:
    return lambda u: point


__all__ = ["Aim", "Chase", "Fade", "Flash", "Path", "Pulse", "Rainbow", "Recolor", "Strobe", "circle", "still"]
