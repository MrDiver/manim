"""Finale: the ManimX logo arrives on the beat, then fireworks.

It continues where ``LightShow`` ends: the same stage, in blackout, the
camera facing the stage. On four hits the parts of the Manim logo slam
in, each with a moving head snapping on in its color; on the fifth the X
(two glowing slabs) grows in and the whole rig flashes white.

The fireworks are a :class:`~manimx.animation.Simulation`: a step
function launches shells from behind the stage, lets them climb against
gravity and bursts each at its apex into a shell of sparks with drag,
gravity and a fading glow. Every burst flashes a point light over the
stage and plays a bang at the simulated time. The simulation is baked
once while the scene is prepared, so rendering (in any order, in worker
processes) never simulates again. The sparks are emissive meshes, so
the bloom makes them glow, and the motion blur draws their trails.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from manimx.animation import PointLight, Scene, Shape3D, Simulation
from manimx.examples.manim_logo import LOGO_BLUE, LOGO_GREEN, LOGO_RED
from manimx.robjects.transform import rotation_towards
from manimx.userland import Box, FadeIn
from manimx.userland.geometry import sphere_mesh

from .lights import Aim, Fade, Flash, Recolor
from .music import FINALE_BEATS, FINALE_LENGTH, FINALE_X, boom, finale_track, whoosh
from .rig import AMBER, CYAN, LOGO_CENTER, LOGO_LIGHT, MAGENTA, WHITE, X_GLOW, build_stage, fly, manimx_logo, on_floor, rgb, stand_up

GRAVITY = 9.81
SPARKS_PER_SHELL = 64
FIREWORKS_START = FINALE_X + 0.6
LAUNCH_Y = 14.0
"""Shells go up from behind the stage."""


@dataclass(frozen=True)
class Shell:
    launch: float  # seconds after the fireworks start
    x: float
    height: float  # of the burst
    color: str
    speed: float = 6.0  # of the sparks right after the burst

    @property
    def rise(self) -> float:
        """Seconds from the launch to the apex."""
        return math.sqrt(2 * (self.height - 1.0) / GRAVITY)


SHELLS = (
    Shell(0.0, -4.5, 12.5, "#ffc94a"),
    Shell(0.6, 4.0, 13.5, MAGENTA),
    Shell(1.3, 0.5, 14.5, X_GLOW),
    Shell(1.9, -6.5, 12.0, LOGO_GREEN),
    Shell(2.4, 6.5, 12.5, LOGO_RED),
    Shell(3.3, -3.5, 14.0, WHITE, 7.5),
    Shell(3.3, 3.5, 14.0, WHITE, 7.5),
    # The grand finale: a salvo of five, the biggest in the middle.
    Shell(4.8, -7.0, 13.0, MAGENTA, 7.0),
    Shell(4.8, 7.0, 13.0, X_GLOW, 7.0),
    Shell(5.0, -3.0, 15.0, "#ffc94a", 8.0),
    Shell(5.0, 3.0, 15.0, "#ffc94a", 8.0),
    Shell(5.2, 0.0, 16.5, WHITE, 10.0),
)
FIREWORKS_LENGTH = FINALE_LENGTH - FIREWORKS_START

CAMERA_PATH = [
    (0.0, (0.0, -13.0, 3.4), tuple(LOGO_CENTER)),
    (FINALE_X, (0.0, -7.0, 2.6), (0.0, 8.6, 3.9)),
    (FIREWORKS_START + 1.5, (0.0, -10.0, 2.0), (0.0, 9.0, 6.4)),
    (FINALE_LENGTH, (0.0, -12.5, 1.8), (0.0, 9.0, 7.4)),
]


X_GLOW_REST = 3.0
"""The X's glow between bursts."""


class Finale(Scene):
    def construct(self) -> None:
        stage = build_stage(self.time, beat=finale_beat, centerpiece=False)
        logo = stand_up(manimx_logo(), scale=1.35)
        parts = [(logo.circle, LOGO_GREEN), (logo.square, LOGO_BLUE), (logo.triangle, LOGO_RED), (logo.m, LOGO_LIGHT)]
        for (part, color), head in zip(parts, stage.back, strict=True):
            part.alpha.set(0.0)
            head.point_at(on_floor(0.55 * head.mount[0], 7.0, 0.8))
            head.light.color.set(rgb(color))
        for i, head in enumerate(stage.front):
            head.point_at(on_floor(-6.0 + 3.0 * i, 3.0))
            head.light.color.set(rgb(WHITE))
        logo.x.set_scale(0.0)
        self.add(stage.root, logo.root)
        self.camera.set_field_of_view(math.radians(48))
        self.camera.near, self.camera.far = 0.1, 160.0
        fly(self.camera, self.time, CAMERA_PATH)
        self.play_wave(finale_track())

        # The logo, part by part on the beat, each part with its own light.
        for (part, _), head, at in zip(parts, stage.back, FINALE_BEATS, strict=True):
            self.wait_until(at)
            self.play(*slam_in(part), Flash(head.light, peak=500.0, rest=300.0, run_time=0.4))

        # The X grows in on the big kick, glowing; the front rig flashes white.
        self.wait_until(FINALE_X)
        self.play(
            logo.x.animate.set_scale(1.0).set_run_time(0.3),
            *(Fade(bar.emissive_strength, X_GLOW_REST, start=12.0, run_time=0.6) for bar in logo.bars),
            *(Flash(head.light, peak=900.0, rest=0.0, decay=0.5, run_time=0.6) for head in stage.front),
        )

        # The fireworks, with the rig swaying slowly in warm colors underneath.
        self.wait_until(FIREWORKS_START)
        self.play(
            make_fireworks(logo.bars),
            *(Aim(head, sway(i), blend=0.15, run_time=FIREWORKS_LENGTH) for i, head in enumerate(stage.heads)),
            *(Fade(head.light.intensity, 260.0, run_time=1.5) for head in stage.heads),
            *(Recolor(head.light, [(AMBER, MAGENTA, CYAN)[i % 3]], run_time=0.1) for i, head in enumerate(stage.heads)),
        )


def slam_in(part):
    """Fade ``part`` in while it shrinks from 1.3 times its size."""
    size = part.scale.value
    part.set_scale((1.3 * size[0], 1.3 * size[1], size[2]))
    return [FadeIn(part, run_time=0.3), part.animate.set_scale(size).set_run_time(0.3)]


def sway(i: int):
    """A slow wander over the floor in front of the stage."""
    return lambda u: on_floor(7.0 * math.sin(7.5 * u + 0.7 * i), 4.0 + 2.5 * math.cos(5.6 * u + i))


def finale_beat(t: float) -> float:
    """What the floor grid, the arches and the speakers pulse with here."""
    hits = [*FINALE_BEATS, FINALE_X] + [FIREWORKS_START + s.launch + s.rise for s in SHELLS]
    past = [t - at for at in hits if at <= t]
    return math.exp(-min(past) / 0.2) if past else 0.0


def make_fireworks(glowing: Sequence[Box]) -> Simulation:
    """The fireworks as a Simulation, with its rockets and sparks (all
    hidden, scale 0, until the step function lights them) and two point
    lights for the flashes. Every burst also lights up ``glowing`` (the
    X), which settles back to its rest glow."""
    rng = np.random.default_rng(7)
    mesh = sphere_mesh(1.0, 10, 6)  # one mesh for every spark

    def particle(
        name: str,
    ) -> Shape3D:  # a sphere around z, stretched along z when it flies
        node = Shape3D(
            mesh,
            name,
            color="#000000",
            emissive=WHITE,
            emissive_strength=0.0,
            cast_shadows=False,
        )
        return node.set_scale(0.0)

    rockets = [particle(f"rocket-{i}") for i in range(len(SHELLS))]
    sparks = [particle(f"spark-{i}") for i in range(len(SHELLS) * SPARKS_PER_SHELL)]
    flashes = [PointLight(f"flash-{i}", intensity=0.0, cast_shadows=False) for i in range(2)]

    count = len(sparks)
    position = np.zeros((count, 3))
    velocity = np.zeros((count, 3))
    age = np.zeros(count)
    life = np.zeros(count)
    alive = np.zeros(count, dtype=bool)
    flicker = rng.uniform(0.0, math.tau, count)
    rocket_state: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    flash_state = [(0.0, -10.0) for _ in flashes]  # (peak intensity, burst time)

    def launch(index: int, shell: Shell) -> None:
        speed = math.sqrt(2 * GRAVITY * (shell.height - 1.0))
        rocket_state[index] = (
            np.array([shell.x, LAUNCH_Y, 1.0]),
            np.array([rng.normal(0, 0.3), 0.0, speed]),
        )
        rockets[index].set_emissive(shell.color, 8.0)
        simulation.play_wave(whoosh(index), volume=0.7)

    def burst(index: int, shell: Shell, where: np.ndarray) -> None:
        rockets[index].set_scale(0.0)
        group = slice(index * SPARKS_PER_SHELL, (index + 1) * SPARKS_PER_SHELL)
        directions = rng.normal(size=(SPARKS_PER_SHELL, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        position[group] = where
        velocity[group] = directions * shell.speed * rng.uniform(0.85, 1.0, (SPARKS_PER_SHELL, 1))
        age[group] = 0.0
        life[group] = rng.uniform(1.4, 2.3, SPARKS_PER_SHELL)
        alive[group] = True
        for spark in sparks[group]:
            spark.set_emissive(shell.color)
        flash = flashes[index % 2]
        flash.move_to(where).set_color(shell.color)
        flash_state[index % 2] = (2500.0 * shell.speed / 6.0, simulation.time)
        for node in glowing:
            node.set_emissive(strength=X_GLOW_REST + 6.0)
        simulation.emit("burst")
        simulation.play_wave(boom(index), volume=0.3 * shell.speed / 6.0)  # the salvo's five bangs add up

    def step(dt: float) -> None:
        now = simulation.time
        for index, shell in enumerate(SHELLS):
            if index not in rocket_state and now >= shell.launch:
                launch(index, shell)
        for index, (where, speed) in list(rocket_state.items()):
            if speed[2] <= 0.0:
                continue  # burst already
            speed[2] -= GRAVITY * dt
            where += speed * dt
            if speed[2] <= 0.0:
                burst(index, SHELLS[index], where.copy())
            else:
                rockets[index].move_to(where).set_scale(0.1)

        moving = np.flatnonzero(alive)
        velocity[moving] *= math.exp(-1.4 * dt)  # air drag
        velocity[moving, 2] -= 0.45 * GRAVITY * dt
        position[moving] += velocity[moving] * dt
        age[moving] += dt
        for i in moving:
            left = 1.0 - age[i] / life[i]
            if left <= 0.0:
                alive[i] = False
                sparks[i].set_scale(0.0)
                continue
            twinkle = 0.75 + 0.25 * math.sin(40.0 * now + flicker[i]) if left < 0.5 else 1.0
            # Stretched along its velocity: fast sparks streak, slow ones are dots.
            size = 0.08 * (0.4 + 0.6 * left)
            speed = float(np.linalg.norm(velocity[i]))
            sparks[i].move_to(position[i]).set_scale((size, size, size + 0.07 * speed))
            if speed > 1e-6:
                sparks[i].rotation.set(rotation_towards(tuple(velocity[i])))
            sparks[i].set_emissive(strength=14.0 * left**1.5 * twinkle)

        for node in glowing:
            if node.emissive_strength.value > X_GLOW_REST:
                node.set_emissive(strength=X_GLOW_REST + (node.emissive_strength.value - X_GLOW_REST) * math.exp(-dt / 0.3))
        for flash, (peak, at) in zip(flashes, flash_state, strict=True):
            flash.set_intensity(peak * math.exp(-(now - at) / 0.25) if now >= at else 0.0)

    simulation = Simulation(step, dt=1 / 60, duration=FIREWORKS_LENGTH, adds=[*rockets, *sparks, *flashes])
    return simulation
