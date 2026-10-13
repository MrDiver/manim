"""Focus: depth of field and a focus pull along a row of spheres.

Shows the camera's lens: ``set_aperture`` opens it (how blurred far
things get), ``focus_on`` pulls the focus to a point, animated like any
camera move.
"""

from __future__ import annotations

from manimx.animation import Scene
from manimx.userland import FadeIn, Sphere, Typst

COLORS = ("#e07a5f", "#f2cc8f", "#87c2a5", "#5ea3d0", "#9b8ec4")


class Focus(Scene):
    def construct(self) -> None:
        self.camera.move_to((6.0, -9.0, 2.5)).look_at((0.0, 4.0, 0.0))
        self.camera.set_field_of_view(0.55)
        self.camera.set_aperture(0.025)
        spheres = [Sphere(0.7, color=color, roughness=0.35).move_to((-1.0 + 1.2 * i, -2.0 + 3.5 * i, 0.0)) for i, color in enumerate(COLORS)]
        title = Typst("focus", '#set text(size: 40pt, fill: rgb("#ece6e2"))').move_to((5.0, -2.0, 2.0))
        title.rotate(1.2, axis=(1.0, 0.0, 0.0))

        self.camera.focus_on(spheres[0].position.value)
        self.play(*(FadeIn(sphere) for sphere in spheres), FadeIn(title), run_time=1.0)
        for sphere in spheres[1:]:
            self.play(self.camera.animate.focus_on(sphere.position.value), run_time=1.2)
        self.play(self.camera.animate.focus_on(spheres[0].position.value), run_time=2.0)
