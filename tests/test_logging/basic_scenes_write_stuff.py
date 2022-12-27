from __future__ import annotations

from manim._config.init_config import config
from manim.animation.creation import Write
from manim.constants import DOWN, LARGE_BUFF
from manim.mobject.text.tex_mobject import MathTex, Tex
from manim.mobject.types.vectorized_mobject import VGroup
from manim.scene.scene import Scene
from manim.utils.color import YELLOW

# This module is used in the CLI tests in tests_CLi.py.


class WriteStuff(Scene):
    def construct(self):
        example_text = Tex("This is a some text", tex_to_color_map={"text": YELLOW})
        example_tex = MathTex(
            "\\sum_{k=1}^\\infty {1 \\over k^2} = {\\pi^2 \\over 6}",
        )
        group = VGroup(example_text, example_tex)
        group.arrange(DOWN)
        group.width = config["frame_width"] - 2 * LARGE_BUFF

        self.play(Write(example_text))
