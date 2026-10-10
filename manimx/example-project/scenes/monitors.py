"""Monitors: two cameras' pictures on planes in the scene.

Shows ``CameraView``: a top camera and a side camera film a
small 3D set, and their pictures sit next to it, one of them masked by a
circle (a ClippingGroup). The cameras are animated like any node.
"""

from __future__ import annotations

from manimx.animation import Camera, CameraView, ClippingGroup, Group, Scene
from manimx.userland import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Box,
    Circle,
    FadeIn,
    Rectangle,
    Rotate,
    Sphere,
    Torus,
    Typst,
    Write,
)

INK = "#ece6e2"


class Monitors(Scene):
    def construct(self) -> None:
        box = Box(1.2, 1.2, 1.2, color="#5ea3d0").move_to((-1.2, 0.0, 0.6))
        ball = Sphere(0.6, color="#e07a5f", roughness=0.3).move_to((1.0, 0.6, 0.6))
        ring = Torus(0.7, 0.18, color="#87c2a5", metallic=0.5).move_to((0.2, -1.4, 0.3))
        stage = Group(box, ball, ring).shift(LEFT * 3.2)

        side = Camera("side", position=(-3.2, -6.0, 3.0), field_of_view=0.7)
        side.look_at((-3.2, 0.0, 0.5))
        top = Camera("top", position=(-3.2, 0.0, 9.0), frame_height=4.5)

        side_view = CameraView(side, 4.0, 2.25).move_to(RIGHT * 3.4 + UP * 1.6)
        frame = Rectangle(4.1, 2.35, color=INK, stroke_width=2.0).move_to(side_view.position.value)
        top_view = ClippingGroup(Circle(1.3, fill_opacity=1.0), CameraView(top, 3.0, 3.0), "porthole").move_to(RIGHT * 3.4 + DOWN * 1.8)
        label = Typst("side · top", f'#set text(size: 20pt, fill: rgb("{INK}"))').move_to(RIGHT * 3.4 + DOWN * 3.4)

        self.play(FadeIn(stage), run_time=1.0)
        self.play(FadeIn(side_view), FadeIn(frame), FadeIn(top_view), Write(label), run_time=1.0)
        self.play(
            side.animate.shift(RIGHT * 2.5),
            Rotate(ring, PI, axis=(1.0, 0.0, 0.0)),
            run_time=3.0,
        )
        self.play(top.animate.zoom(1.8), run_time=1.5)
        self.wait(0.5)
