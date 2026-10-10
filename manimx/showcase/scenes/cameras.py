"""Cameras: moving the camera, other cameras' pictures, and focus.

The scene's camera is a node: animate.move_to().look_at() describes
where it ends and what it looks at. CameraView shows another camera's
picture on a plane (here masked by a circle). The lens is a signal too:
focus_on() pulls the focus.
"""

from __future__ import annotations

from manimx.animation import Camera, CameraView, ClippingGroup, Scene
from manimx.userland import Box, Circle, FadeIn, Rectangle, Rotate, Sphere, Torus

from .style import GREEN, INK, NAVY, RED, SAND


class Cameras(Scene):
    def construct(self) -> None:
        self.camera.move_to((0.0, -12.0, 4.0)).look_at((0.0, 0.0, 0.5))
        self.camera.set_field_of_view(0.6)

        ground = Box(16.0, 10.0, 0.2, color=NAVY, roughness=0.9).move_to((0.0, 2.0, -0.1))
        box = Box(1.2, 1.2, 1.2, color=RED).move_to((-3.0, 0.0, 0.6))
        ball = Sphere(0.7, color=SAND, roughness=0.3).move_to((0.0, 2.5, 0.7))
        ring = Torus(0.7, 0.22, color=GREEN).move_to((3.0, 5.0, 0.7))

        # Another camera looks down from above; its picture hangs in the
        # air, seen through a round window.
        top = Camera("top", position=(0.0, 2.5, 12.0), frame_height=13.0)
        picture = CameraView(top, 3.2, 3.2)
        window = ClippingGroup(Circle(1.5, fill_opacity=1.0), picture, "window")
        rim = Circle(1.55, color=INK, stroke_width=3.0)
        monitor = ClippingGroup(Rectangle(20.0, 20.0, fill_opacity=1.0), window, "monitor")  # nested masks work too
        for node in (monitor, rim):
            # Upright, turned towards where the camera ends up.
            node.move_to((3.5, 0.5, 3.0)).rotate(1.45, axis=(1.0, 0.0, 0.0)).rotate(-0.75, axis=(0.0, 0.0, 1.0))

        self.add(ground, box, ball, ring)
        self.play(FadeIn(monitor), FadeIn(rim), run_time=0.8)
        # The chain is one end state: at the new place, looking at the ball.
        self.play(self.camera.animate.move_to((-8.0, -8.0, 5.0)).look_at((0.0, 2.5, 0.7)), Rotate(ring, 3.0), run_time=2.5)
        self.play(top.animate.zoom(1.6), run_time=1.0)

        # Depth of field: open the lens, then pull the focus front to back.
        self.camera.set_aperture(0.03)
        self.play(self.camera.animate.focus_on(box.position.value), run_time=0.8)
        self.play(self.camera.animate.focus_on(ring.position.value), run_time=1.5)
        self.play(self.camera.animate.focus_on(ball.position.value).set_aperture(0.0), run_time=1.5)
        self.wait(0.3)
