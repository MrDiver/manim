"""Shapes3D: the round and straight solids, lines and arrows in space, and
three-dimensional axes with a surface and a curve on them.

Cube, Prism, Cylinder and Cone are Shape3D meshes standing along z;
Line3D, Arrow3D and Dot3D mark points and directions; ThreeDAxes places
coordinates, plot_surface draws z = f(x, y) and plot_curve a curve.
"""

from __future__ import annotations

import math

from manimx.animation import DirectionalLight, Scene
from manimx.userland import (
    TAU,
    Arrow3D,
    Box,
    Cone,
    Create,
    Cube,
    Cylinder,
    Dot3D,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line3D,
    Prism,
    Rotate,
    ThreeDAxes,
)

from .style import BLUE, GREEN, INK, MUTED, NAVY, RED, SAND, VIOLET


class Shapes3D(Scene):
    def construct(self) -> None:
        self.camera.move_to((0.0, -11.0, 6.0)).look_at((0.0, 0.3, 0.8))
        self.camera.set_field_of_view(0.55)
        sun = DirectionalLight(direction=(0.5, 0.9, -1.0), color="#fff2e0", intensity=2.2, size=0.08)
        ground = Box(16.0, 10.0, 0.2, color=NAVY, roughness=0.9).move_to((0.0, 0.0, -0.1))
        self.add(sun, ground)

        solids = [
            Cube(1.2, color=RED, roughness=0.5).move_to((-4.6, 1.8, 0.6)),
            Prism(6, 0.7, 1.4, color=SAND, roughness=0.4).move_to((-2.8, 1.8, 0.7)),
            Cylinder(0.6, 1.4, color=GREEN, roughness=0.35).move_to((-1.0, 1.8, 0.7)),
            Cone(0.7, 1.5, color=BLUE, roughness=0.4).move_to((0.8, 1.8, 0.75)),
        ]
        self.play(*(GrowFromCenter(solid) for solid in solids), run_time=1.0)
        self.play(*(Rotate(solid, TAU / 3) for solid in solids[:3]), run_time=1.5)

        origin = (-2.6, -1.4, 0.0)
        dots = [Dot3D(0.12, color=INK).move_to(point) for point in ((-4.2, -2.0, 1.6), (-1.2, -1.8, 2.2), (-2.0, -0.2, 1.2))]
        arrows = [Arrow3D(origin, dot.get_center(), color=color, stroke_width=5.0) for dot, color in zip(dots, (RED, SAND, VIOLET), strict=True)]
        links = [Line3D(dots[index].get_center(), dots[(index + 1) % 3].get_center(), color=MUTED, stroke_width=3.0) for index in range(3)]
        self.play(*(FadeIn(dot) for dot in dots), run_time=0.5)
        self.play(*(Create(arrow.line) for arrow in arrows), *(GrowFromCenter(arrow.tip) for arrow in arrows), run_time=0.8)
        self.add(*arrows)
        self.play(*(Create(link) for link in links), run_time=0.8)
        self.play(links[0].animate.set_ends(dots[0].get_center(), (-2.6, -1.4, 2.8)), run_time=0.8)

        axes = ThreeDAxes((-3, 3, 1), (-3, 3, 1), (-1, 2, 1), 3.4, 3.4, 2.0, color=INK, stroke_width=2.0).move_to((3.6, -0.2, 1.5))
        surface = axes.plot_surface(lambda x, y: 0.8 * math.sin(x) * math.cos(y), resolution=(40, 40), color=BLUE, roughness=0.5)
        curve = axes.plot_curve(lambda t: (2.5 * math.cos(t), 2.5 * math.sin(t), 1.5 + 0.4 * math.sin(3 * t)), (0.0, TAU), 96, color=SAND, stroke_width=4.0)
        self.play(Create(axes), run_time=1.0)
        self.play(FadeIn(surface), run_time=0.8)
        self.play(Create(curve), run_time=1.2)
        self.play(self.camera.animate.move_to((10.0, -9.0, 8.0)).look_at((1.0, 0.0, 0.8)), run_time=2.5)
        self.wait(0.5)
        self.play(*(FadeOut(node) for node in (*solids, *dots, *arrows, *links, axes, surface, curve, ground)), run_time=0.6)
