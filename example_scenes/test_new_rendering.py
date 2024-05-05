from manim import *


class Test(Scene):
    def construct(self) -> None:
        c = Circle()
        self.play(Create(c))

class Test2(Scene):
    def construct(self) -> None:
        c = VGroup(*[Circle().set_fill(opacity=1) for _ in range(99)]).arrange_in_grid().scale(0.1)
        c2 = VGroup(*[Circle().set_fill(color=GREEN, opacity=1) for _ in range(99)]).arrange_in_grid().scale(0.1).shift(RIGHT/2)
        self.play(Create(c), Create(c2), run_time=10)

with tempconfig({"renderer": "opengl", "preview": True, "parallel": False}):
    Test2().render()
