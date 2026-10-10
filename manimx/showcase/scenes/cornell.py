"""Cornell box: the classic test scene of light transport.

A white room with a red left and a green right wall, two blocks, and a
light in the ceiling: an emissive panel with a soft point light just
below it. The light comes up, the blocks turn, and the soft shadows
follow. Without global illumination the walls do not light each other;
a dim light without shadows stands in for that bounce.
"""

from __future__ import annotations

from manimx.animation import PointLight, Scene
from manimx.userland import Box, FadeIn, Rotate

WHITE = "#bab9b3"
RED = "#a12a1f"
GREEN = "#2f7a2a"


class Cornell(Scene):
    def construct(self) -> None:
        self.camera.move_to((0.0, -6.8, 2.0)).look_at((0.0, 2.0, 2.0))
        self.camera.set_field_of_view(0.62)

        wall = {"roughness": 0.95}
        room = [
            Box(4.2, 4.2, 0.1, color=WHITE, **wall).move_to((0.0, 2.0, -0.05)),  # floor
            Box(4.2, 4.2, 0.1, color=WHITE, **wall).move_to((0.0, 2.0, 4.05)),  # ceiling
            Box(4.2, 0.1, 4.2, color=WHITE, **wall).move_to((0.0, 4.05, 2.0)),  # back
            Box(0.1, 4.2, 4.2, color=RED, **wall).move_to((-2.05, 2.0, 2.0)),
            Box(0.1, 4.2, 4.2, color=GREEN, **wall).move_to((2.05, 2.0, 2.0)),
        ]
        tall = Box(1.2, 1.2, 2.4, color=WHITE, **wall).move_to((-0.7, 2.6, 1.2)).rotate(0.3, axis=(0.0, 0.0, 1.0))
        short = Box(1.2, 1.2, 1.2, color=WHITE, **wall).move_to((0.75, 1.3, 0.6)).rotate(-0.3, axis=(0.0, 0.0, 1.0))
        panel = Box(1.0, 1.0, 0.02, color="#000000", emissive="#fff4e0", emissive_strength=6.0, cast_shadows=False).move_to((0.0, 2.0, 3.99))
        lamp = PointLight(position=(0.0, 2.0, 3.75), color="#fff4e0", intensity=0.0, size=0.5)
        # Light bounced off the walls, which a renderer without global
        # illumination does not compute: a dim light without shadows near
        # the open side.
        bounce = PointLight(position=(0.0, -0.5, 2.0), color="#fff4e0", intensity=0.0, cast_shadows=False)

        self.add(*room, panel, lamp, bounce)
        self.play(FadeIn(tall), FadeIn(short), lamp.animate.set_intensity(28.0), bounce.animate.set_intensity(4.0), run_time=2.0)
        self.play(Rotate(tall, -0.6, axis=(0.0, 0.0, 1.0)), Rotate(short, 0.6, axis=(0.0, 0.0, 1.0)), run_time=3.0)
        self.play(self.camera.animate.move_to((0.0, -5.0, 2.0)), run_time=2.0)
        self.wait(1.0)



