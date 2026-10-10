"""Physics: a simulation baked into keyframes, with sound.

A Simulation steps a function while the scene is prepared; its results
are stored like any animation, so rendering never simulates again. It
emits an event at every bounce and plays a synthesized plop, panned to
where the ball lands.
"""

from __future__ import annotations

import numpy as np
from manimx.animation import Scene, Simulation
from manimx.animation.audio import DEFAULT_SAMPLE_RATE
from manimx.userland import DOWN, LEFT, UP, Dot, FadeIn, FadeOut, Rectangle

from .style import NAVY, PALETTE, caption

GRAVITY = 9.81
FLOOR = -2.6
RADIUS = 0.22


def plop(pitch: float) -> np.ndarray:
    """A short drop-like sound: a sine gliding down, decaying fast."""
    t = np.arange(int(0.12 * DEFAULT_SAMPLE_RATE)) / DEFAULT_SAMPLE_RATE
    frequency = pitch * (180 + 700 * np.exp(-t / 0.018))
    phase = 2 * np.pi * np.cumsum(frequency) / DEFAULT_SAMPLE_RATE
    return (0.5 * np.sin(phase) * np.minimum(t / 0.002, 1.0) * np.exp(-t / 0.03)).astype(np.float32)


def panned(sound: np.ndarray, position: float) -> np.ndarray:
    angle = (min(max(position, -1.0), 1.0) + 1.0) * np.pi / 4
    return np.sqrt(2) * np.outer(sound, [np.cos(angle), np.sin(angle)]).astype(np.float32)


class Physics(Scene):
    def construct(self) -> None:
        title = caption("Physics", "Simulation: a step function baked into keyframes, with sound")
        floor = Rectangle(12.0, 0.1, color=NAVY, fill_opacity=1.0).move_to(DOWN * (-FLOOR + RADIUS + 0.05))
        balls = [Dot(RADIUS, color=color).move_to(LEFT * (4.0 - 2.0 * i) + UP * (0.5 + 0.4 * i)) for i, color in enumerate(PALETTE)]
        velocities = np.array([[0.6 * (i - 2), 0.0, 0.0] for i in range(len(balls))])
        sounds = [plop(1.0 + 0.15 * i) for i in range(len(balls))]

        def step(dt: float) -> None:
            for ball, velocity, sound in zip(balls, velocities, sounds, strict=True):
                velocity[1] -= GRAVITY * dt
                ball.shift(velocity * dt)
                x, y = ball.position.x.value, ball.position.y.value
                if y < FLOOR and velocity[1] < 0:
                    ball.shift(UP * (FLOOR - y))
                    impact = -velocity[1]
                    velocity[1] = impact * 0.8
                    simulation.emit("bounce")
                    if impact > 0.6:
                        simulation.play_wave(panned(sound, x / 5.0), volume=min(1.0, impact / 9.0) ** 1.5)
                if abs(x) > 5.6:
                    velocity[0] = -velocity[0]

        simulation = Simulation(step, dt=1 / 480, duration=5.0)
        self.play(FadeIn(title), FadeIn(floor), *(FadeIn(ball) for ball in balls), run_time=0.6)
        self.play(simulation, run_time=5.0)
        self.play(*(FadeOut(node) for node in (title, floor, *balls)), run_time=0.6)
