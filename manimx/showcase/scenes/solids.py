"""Solids: 3D shapes, materials, lights and shadows.

Box, Sphere, Torus and a ParametricSurface with metallic-roughness
materials; directional, point and spot lights casting soft shadows; an
emissive material glowing with bloom; a ParametricCurve3D and Particles.
"""

from __future__ import annotations

import math

import numpy as np
from manimx.animation import DirectionalLight, Particles, PointLight, Scene, SpotLight
from manimx.userland import (
    TAU,
    Box,
    FadeIn,
    FadeOut,
    ParametricCurve3D,
    ParametricSurface,
    Rotate,
    Sphere,
    Torus,
)
from manimx.userland.geometry import box_mesh

from .style import BLUE, GREEN, INK, NAVY, RED, SAND, VIOLET


def wave(u: float, v: float) -> tuple[float, float, float]:
    return (u, v, 0.25 * math.sin(2.0 * u) * math.cos(2.0 * v))


class Solids(Scene):
    def construct(self) -> None:
        self.camera.move_to((0.0, -11.0, 6.5)).look_at((0.0, 0.0, 0.5))
        self.camera.set_field_of_view(0.55)

        ground = Box(14.0, 9.0, 0.2, color=NAVY, roughness=0.9).move_to((0.0, 0.0, -0.1))
        box = Box(1.4, 1.4, 1.4, color=RED, roughness=0.5).move_to((-3.6, 0.6, 0.7))
        ball = Sphere(0.85, color=SAND, metallic=1.0, roughness=0.25).move_to((-1.2, -0.6, 0.85))
        ring = Torus(0.8, 0.25, color=GREEN, roughness=0.35).move_to((1.3, 0.8, 1.0)).rotate(math.pi / 2, axis=(1.0, 0.0, 0.0))
        lamp = Sphere(0.35, color="#000000", emissive="#ffd9a0", emissive_strength=6.0).move_to((3.6, -0.6, 1.6))
        surface = ParametricSurface(wave, (-1.2, 1.2), (-1.2, 1.2), color=BLUE, roughness=0.4).move_to((3.6, 1.6, 0.3))
        helix = ParametricCurve3D(lambda t: (0.6 * math.cos(t), 0.6 * math.sin(t), 0.12 * t), (0.0, 4 * TAU), color=VIOLET, stroke_width=4.0)
        helix.move_to((-1.2, 2.4, 1.5))

        sun = DirectionalLight(direction=(0.6, 0.8, -1.0), color="#fff2e0", intensity=2.0, size=0.08)
        glow = PointLight(position=(3.6, -0.6, 1.6), color="#ffd9a0", intensity=25.0, size=0.3)
        spot = SpotLight(position=(-4.5, -3.0, 5.0), target=(-3.6, 0.6, 0.0), color=INK, intensity=60.0, outer_cone_angle=0.5)
        self.add(sun, glow, spot, ground)

        sparks = Particles(40, box_mesh(0.12, 0.12, 0.12), color="#ffffff", emissive="#ffffff", emissive_strength=2.0)
        angles = np.linspace(0.0, TAU, 40, endpoint=False)
        sparks.set_positions(np.column_stack([3.6 + 1.0 * np.cos(angles), -0.6 + 1.0 * np.sin(angles), np.full(40, 1.6)]))

        self.play(*(FadeIn(node) for node in (box, ball, ring, lamp, surface, helix, sparks)), run_time=1.0)
        self.play(
            Rotate(box, math.pi / 2),
            Rotate(ring, TAU, axis=(0.0, 0.0, 1.0)),
            sparks.animate.set_positions(np.column_stack([3.6 + 1.6 * np.cos(angles + 1.0), -0.6 + 1.6 * np.sin(angles + 1.0), 1.6 + 0.6 * np.sin(3 * angles)])),
            spot.animate.shift((3.0, 0.0, 0.0)),
            run_time=3.0,
        )
        self.play(ball.animate.set_material(roughness=0.8, metallic=0.0, color=VIOLET), run_time=1.0)
        self.wait(0.5)
        self.play(*(FadeOut(node) for node in (box, ball, ring, lamp, surface, helix, sparks, ground)), run_time=0.6)
