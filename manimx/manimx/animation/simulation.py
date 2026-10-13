"""Simulations: user-defined step functions, baked into keyframes once.

Section 7 of docs/animation_engine_design.md. manimx ships no physics
engine. A :class:`Simulation` is a function ``step(dt)`` that changes
nodes however it likes (a hand-written integrator, springs,
particles, a wrapped engine), plus a step size::

    ball = Dot()
    velocity = np.array([2.0, 0.0, 0.0])

    def fall(dt):
        velocity[1] -= 9.81 * dt
        ball.shift(velocity * dt)
        if ball.position.y.value < -3:          # bounce
            velocity[1] = abs(velocity[1]) * 0.8

    self.play(paddle.animate.shift(RIGHT), Simulation(fall, dt=1 / 240, duration=4))

Simulating is sequential by nature, the timeline is random access. They
meet by **baking**: when ``play`` places a simulation (while
``construct()`` runs, so once per prepare), the step function runs from
start to end at the fixed step, and the values of every signal it changed
are stored as keyframes, ``keyframe_rate`` per second. Afterwards the
simulation is an ordinary animation that interpolates its keyframes, so
``state_at(t)``, worker processes, retiming and cuts work unchanged and
never re-simulate.

What a step sees:

* **Animated inputs:** before each step, every signal that other
  animations drive is set to its value at that time, so a paddle animated
  in the same ``play`` call pushes the ball. A simulation sees everything
  recorded before it (and the other animations of its ``play`` call),
  never what scene code records later.
* **Its own outputs:** a signal belongs to the simulation from the first
  step that changes it; from then on only the step function sets it.
* **Time:** :attr:`Simulation.time` is the time since the simulation
  started; ``scene_time`` and derived signals of it follow too. The global
  time is not known while preparing, so it equals the scene time here.

Reactions run the other way through signals: derived signals of simulated
ones (``label.follow(ball)``) follow every keyframe, and ``emit`` inside
the step function records an event at the simulated time, which other
clips (a camera scene) can wait for.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .animations import Animation, Target
from .audio import DEFAULT_SAMPLE_RATE, Sound
from .authoring import _active
from .signals.contributions import Contribution, Keyframes
from .signals.signal import RootSignal
from .signals.signal_tracks import Change, linear

if TYPE_CHECKING:
    from .nodes.node import Node
    from .scene import SceneCompiler

DEFAULT_KEYFRAME_RATE = 60.0


class Simulation(Animation):
    """A step function baked into keyframes when it is played.

    It ends after ``duration`` seconds, or at the first step after which
    ``until()`` returns True (then within ``max_time``, or an error). The
    last step is shortened so the simulation ends exactly at
    ``duration``. ``adds`` are nodes the step function changes that
    are not in the scene yet (they are added); every node in the scene
    is watched anyway.
    """

    def __init__(
        self,
        step: Callable[[float], Any],
        dt: float = 1 / 240,
        duration: float | None = None,
        until: Callable[[], bool] | None = None,
        max_time: float = 60.0,
        keyframe_rate: float = DEFAULT_KEYFRAME_RATE,
        adds: Sequence[Node] = (),
    ) -> None:
        if dt <= 0.0 or keyframe_rate <= 0.0:
            raise ValueError("a simulation needs a positive step and keyframe rate")
        if (duration is None) == (until is None):
            raise ValueError("a simulation ends after a duration or when until() is true: give exactly one")
        self.step = step
        self.dt = dt
        self.duration = duration
        self.until = until
        self.max_time = max_time
        self.keyframe_rate = keyframe_rate
        self.adds = tuple(adds)
        self.run_time = duration if duration is not None else max_time
        self.rate = linear
        self.time = 0.0
        """Simulated time since the start (read it inside the step function)."""
        self.events: list[tuple[str, float]] = []
        """Events emitted while baking, with their simulated times."""
        self.sounds: list[tuple[Sound, float, float]] = []
        """Sounds started while baking: ``(sound, volume, simulated time)``."""
        self.times: np.ndarray = np.zeros(0)
        """Keyframe times (simulated time) once baked."""
        self._keyframes: dict[int, tuple[RootSignal, list[Any]]] = {}
        self._baked = False

    # -- used inside the step function ---------------------------------------

    def emit(self, event: str) -> None:
        """Fire ``event`` at the current simulated time (e.g. a collision)."""
        self.events.append((event, self.time))

    def play_sound(self, sound: str | Path | Sound, volume: float = 1.0) -> Sound:
        """Start a sound at the current simulated time (e.g. on a bounce)."""
        sound = sound if isinstance(sound, Sound) else Sound.load(sound)
        self.sounds.append((sound, volume, self.time))
        return sound

    def play_wave(
        self, samples: np.ndarray | str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE, volume: float = 1.0
    ) -> Sound:
        """Start a wave at the current simulated time: an array (mono
        ``(count,)`` or stereo ``(count, 2)``, values in [-1, 1], at
        ``sample_rate``) or a sound file (its own sample rate applies).
        ``volume`` scales the samples (1: as recorded)."""
        sound = Sound.load(samples) if isinstance(samples, (str, Path)) else Sound(samples, sample_rate)
        return self.play_sound(sound, volume)

    # -- the animation contract -----------------------------------------------

    def nodes(self) -> Sequence[Node]:
        return self.adds

    def targets(self) -> Sequence[Target]:
        if not self._baked:
            raise RuntimeError("a simulation is baked when play() places it")
        return [Target(signal, Change.ABSOLUTE) for signal, _ in self._keyframes.values()]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        index, fraction = self._keyframe_at(alpha)
        return [_between(signal, values, index, fraction) for signal, values in self._keyframes.values()]

    def planned_contributions(self) -> Sequence[Contribution]:
        times = [float(t) for t in self.times]
        return [Keyframes(self.run_time, times, values) for _, values in self._keyframes.values()]

    def _keyframe_at(self, alpha: float) -> tuple[int, float]:
        """The keyframe before progress ``alpha`` and how far past it."""
        t = alpha * self.run_time
        index = min(max(bisect.bisect_right(self.times, t) - 1, 0), len(self.times) - 2)
        t0, t1 = self.times[index], self.times[index + 1]
        return index, min(max((t - t0) / (t1 - t0), 0.0), 1.0)

    # -- baking ---------------------------------------------------------------

    def bake(self, compiler: SceneCompiler, begin: float) -> float:
        """Run the step function from compiled time ``begin``; returns the
        simulated duration. Called by ``play``."""
        if self._baked:
            raise RuntimeError("a simulation can only be played once")
        watched = _watched_signals([*compiler.roots, *self.adds])
        initial = {key: signal.value for key, signal in watched.items()}
        inputs = {key: track for key, track in compiler.tracks.items() if key in watched}
        owned: set[int] = set()
        recorded: dict[int, list[Any]] = {key: [] for key in watched}
        times: list[float] = []

        def feed(t: float) -> None:
            compiler.set_time(begin + t)
            for key, track in inputs.items():
                if key not in owned:
                    track.signal.set(track.value_at(begin + t))

        def record(t: float) -> None:
            times.append(t)
            for key, signal in watched.items():
                recorded[key].append(signal.value)

        token = _active.set(None)  # inside a step, changes set signals directly
        try:
            feed(0.0)
            record(0.0)
            interval = 1.0 / self.keyframe_rate
            next_keyframe = interval
            self.time = 0.0
            while not self._finished():
                dt = self.dt if self.duration is None else min(self.dt, self.duration - self.time)
                feed(self.time)
                before = {key: signal.version for key, signal in watched.items()}
                self.step(dt)
                owned.update(key for key, signal in watched.items() if signal.version != before[key])
                self.time += dt
                if self.time >= next_keyframe - 1e-9 or self._finished():
                    record(self.time)
                    next_keyframe = (math.floor(self.time / interval + 1e-9) + 1) * interval
        finally:
            _active.reset(token)
            for key, signal in watched.items():
                signal.set(initial[key])  # tracks capture a signal's value before the simulation as their base
            compiler.set_time(begin)

        self.times = np.array(times)
        self._keyframes = {key: (watched[key], recorded[key]) for key in owned}
        self.run_time = self.time
        self._baked = True
        return self.time

    def _finished(self) -> bool:
        if self.duration is not None:
            return self.time >= self.duration - 1e-12
        if self.time > 0.0 and self.until is not None and self.until():
            return True
        if self.time >= self.max_time:
            raise RuntimeError(f"simulation did not finish within max_time={self.max_time} s")
        return False


def _between(signal: RootSignal, values: list[Any], index: int, fraction: float) -> Any:
    """``signal``'s recorded ``values`` between keyframe ``index`` and the next."""
    start, end = values[index], values[index + 1]
    if start is end:
        return start  # unchanged between the keyframes (most signals, most of the time)
    if isinstance(start, np.ndarray) and start.shape != end.shape:
        return start if fraction < 1.0 else end  # the path changed its point count: no blending
    return signal.kind.interpolate(start, end, fraction)


def _watched_signals(nodes: Iterable[Node]) -> dict[int, RootSignal]:
    signals: dict[int, RootSignal] = {}
    for root in nodes:
        for member in root.family():
            for signal in member.signals():
                if isinstance(signal, RootSignal):
                    signals[signal.id] = signal
    return signals
