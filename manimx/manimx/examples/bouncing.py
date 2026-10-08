"""Bouncing balls: a hand-written simulation next to keyframed animation.

    uv run python scripts/render_scene.py manimx.examples.bouncing:Bouncing bouncing.mp4 --background "#0b1020"

No physics engine: the step function below is the whole simulation
(gravity, a floor, bounces, and a wall that an ordinary animation moves).
The wall is animated in the same ``play`` call, so the balls feel it; the
ring around each ball is a derived signal that follows its keyframes; a
bounce emits an event and plays a plop (a synthesized wave array, pitched
per ball, as loud as the impact is hard).
"""

from __future__ import annotations

import numpy as np

from ..animation import DOWN, LEFT, RIGHT, UP, Circle, Dot, FadeIn, FadeOut, Rectangle, Scene, Simulation
from ..animation.audio import DEFAULT_SAMPLE_RATE
from ..animation.rate_functions import there_and_back

GRAVITY = 9.81
FLOOR = -3.0
RADIUS = 0.25
COLORS = ("#e07a5f", "#f2cc8f", "#81b29a", "#5ea3d0", "#c38fd6")


def plop(pitch: float = 1.0, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """A short drop-like "plop": a sine gliding down in pitch, decaying fast."""
    t = np.arange(int(0.12 * sample_rate)) / sample_rate
    frequency = pitch * (180 + 700 * np.exp(-t / 0.018))  # a fast downward glide
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    envelope = np.minimum(t / 0.002, 1.0) * np.exp(-t / 0.03)
    return (0.6 * np.sin(phase) * envelope).astype(np.float32)


class Bouncing(Scene):
    def construct(self) -> None:
        floor = Rectangle(13, 0.1, color="#3d4a6b", fill_opacity=1).move_to(DOWN * (-FLOOR + 0.05 + RADIUS))
        wall = Rectangle(0.2, 3, color="#e8e6e3", fill_opacity=1).move_to(LEFT * 6 + DOWN * 1.6)
        balls = [Dot(RADIUS, color=color).move_to(LEFT * (4 - 1.2 * i) + UP * (1 + 0.5 * i)) for i, color in enumerate(COLORS)]
        rings = [Circle(RADIUS * 1.6, color=color, stroke_width=2).follow(ball) for ball, color in zip(balls, COLORS)]
        velocities = np.array([[0.4 * i - 0.8, 0.0, 0.0] for i in range(len(balls))])
        plops = [plop(1.0 + 0.12 * i) for i in range(len(balls))]

        def step(dt: float) -> None:
            wall_x = wall.position.x.value + 0.1  # the wall's right face, wherever its animation put it
            for ball, velocity, sound in zip(balls, velocities, plops):
                velocity[1] -= GRAVITY * dt
                ball.shift(velocity * dt)
                x, y = ball.position.x.value, ball.position.y.value
                if y < FLOOR and velocity[1] < 0:
                    ball.shift(UP * (FLOOR - y))
                    impact = -velocity[1]
                    velocity[1] = impact * 0.82
                    simulation.emit("bounce")
                    if impact > 0.5:  # resting balls stay quiet
                        simulation.play_wave(sound, gain=min(1.0, impact / 8.0))
                if x - RADIUS < wall_x:
                    ball.shift(RIGHT * (wall_x - x + RADIUS))
                    velocity[0] = max(velocity[0], 0.0) + 6.0 * dt * 60  # pushed by the moving wall
                velocity[0] *= 1.0 - 0.15 * dt

        simulation = Simulation(step, dt=1 / 480, duration=6.0)

        self.play(FadeIn(floor), FadeIn(wall), *(FadeIn(ball) for ball in balls), *(FadeIn(r) for r in rings), run_time=0.5)
        self.play(simulation, wall.animate.shift(RIGHT * 4), rate=there_and_back, run_time=6.0)
        self.play(*(FadeOut(mobject) for mobject in (floor, wall, *balls, *rings)), run_time=0.5)
