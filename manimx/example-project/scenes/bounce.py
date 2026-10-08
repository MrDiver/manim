"""Bounce: balls under gravity, pushed by an animated wall, with stereo plops.

A simulation scene. The step function is the whole physics (gravity, a
floor, bounces, the wall's push); it runs once while the scene is
prepared and is stored as keyframes. The wall is an ordinary animation in
the same ``play`` call, so the balls feel it. Every bounce plays a plop:
a synthesized wave, panned to where the ball lands, with a small stereo
reverb, as loud as the impact is hard.
"""

from __future__ import annotations

import numpy as np

from manimx.animation import DOWN, LEFT, RIGHT, UP, Circle, Dot, FadeIn, FadeOut, Rectangle, Scene, Simulation
from manimx.animation.audio import DEFAULT_SAMPLE_RATE
from manimx.animation.rate_functions import there_and_back

GRAVITY = 9.81
FLOOR = -3.0
RADIUS = 0.25
COLORS = ("#e07a5f", "#f2cc8f", "#81b29a", "#5ea3d0", "#c38fd6")


MAX_IMPACT = 10.0  # m/s, about the first impact from the highest ball
PAN_WIDTH = 4.0  # world units from the center at which a sound is fully on one side


def plop(pitch: float = 1.0, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """A short drop-like "plop": a sine gliding down in pitch, decaying fast."""
    t = np.arange(int(0.12 * sample_rate)) / sample_rate
    frequency = pitch * (180 + 700 * np.exp(-t / 0.018))  # a fast downward glide
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    envelope = np.minimum(t / 0.002, 1.0) * np.exp(-t / 0.03)
    return (0.6 * np.sin(phase) * envelope).astype(np.float32)


def reverb_tail(dry: np.ndarray, seconds: float = 1.2, wet: float = 0.2, seed: int = 0, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """The stereo reverb of mono ``dry``, shape ``(count, 2)``.

    A room's impulse response is roughly noise that dies away (-60 dB
    after ``seconds``, the RT60). Each side gets its own noise, so the two
    tails are uncorrelated and the room sounds wide. Convolution through
    the FFT keeps it fast.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    decay = np.exp(-6.9 * t / seconds)  # 6.9 = ln(1000): -60 dB at t = seconds
    predelay = np.zeros(int(0.015 * sample_rate))  # the first reflections arrive a little later
    size = len(dry) + len(predelay) + len(t) - 1
    spectrum = np.fft.rfft(dry, size)
    tails = []
    for _ in range(2):
        impulse = np.concatenate([predelay, rng.standard_normal(len(t)) * decay])
        impulse *= wet / np.sqrt(np.sum(impulse**2))  # the same energy for every room size
        tails.append(np.fft.irfft(spectrum * np.fft.rfft(impulse, size), size))
    return np.stack(tails, axis=1).astype(np.float32)


def panned(dry: np.ndarray, tail: np.ndarray, position: float) -> np.ndarray:
    """Stereo ``dry`` placed at ``position`` (-1 left .. 1 right) with
    constant power (cos/sin, scaled so the center keeps full level), plus
    the room's ``tail``, which does not move with the source. The
    position is pushed outwards (a power below one), so sounds off the
    center already lean clearly to their side."""
    position = min(max(position, -1.0), 1.0)
    angle = (np.sign(position) * abs(position) ** 0.6 + 1.0) * np.pi / 4
    stereo = tail.copy()
    stereo[: len(dry)] += np.sqrt(2) * np.outer(dry, [np.cos(angle), np.sin(angle)])
    return stereo


class Bounce(Scene):
    def construct(self) -> None:
        floor = Rectangle(13, 0.1, color="#3d4a6b", fill_opacity=1).move_to(DOWN * (-FLOOR + 0.05 + RADIUS))
        wall = Rectangle(0.2, 3, color="#e8e6e3", fill_opacity=1).move_to(LEFT * 6 + DOWN * 1.6)
        balls = [Dot(RADIUS, color=color).move_to(LEFT * (4 - 1.2 * i) + UP * (1 + 0.5 * i)) for i, color in enumerate(COLORS)]
        rings = [Circle(RADIUS * 1.6, color=color, stroke_width=2).follow(ball) for ball, color in zip(balls, COLORS)]
        velocities = np.array([[0.4 * i - 0.8, 0.0, 0.0] for i in range(len(balls))])
        plops = [plop(1.0 + 0.12 * i) for i in range(len(balls))]
        rooms = [reverb_tail(sound, seed=i) for i, sound in enumerate(plops)]

        def step(dt: float) -> None:
            wall_x = wall.position.x.value + 0.1  # the wall's right face, wherever its animation put it
            for ball, velocity, sound, room in zip(balls, velocities, plops, rooms):
                velocity[1] -= GRAVITY * dt
                ball.shift(velocity * dt)
                x, y = ball.position.x.value, ball.position.y.value
                if y < FLOOR and velocity[1] < 0:
                    ball.shift(UP * (FLOOR - y))
                    impact = -velocity[1]
                    velocity[1] = impact * 0.82
                    simulation.emit("bounce")
                    if impact > 0.5:  # resting balls stay quiet
                        stereo = panned(sound, room, ball.position.x.value / PAN_WIDTH)
                        # Loudness follows the impact speed (squared-ish: soft bounces fade fast).
                        simulation.play_wave(stereo, volume=min(1.0, impact / MAX_IMPACT) ** 1.5)
                if x - RADIUS < wall_x:
                    ball.shift(RIGHT * (wall_x - x + RADIUS))
                    velocity[0] = max(velocity[0], 0.0) + 6.0 * dt * 60  # pushed by the moving wall
                velocity[0] *= 1.0 - 0.15 * dt

        simulation = Simulation(step, dt=1 / 480, duration=6.0)

        self.play(FadeIn(floor), FadeIn(wall), *(FadeIn(ball) for ball in balls), *(FadeIn(r) for r in rings), run_time=0.5)
        self.play(simulation, wall.animate.shift(RIGHT * 4), rate=there_and_back, run_time=6.0)
        self.play(*(FadeOut(mobject) for mobject in (floor, wall, *balls, *rings)), run_time=0.5)
