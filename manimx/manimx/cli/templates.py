"""File templates for ``manimx new`` and ``manimx generate``.

Templates are plain ``str.format`` strings; literal braces are doubled.
"""

from __future__ import annotations

SCENES_INIT = '''"""The project's scenes, one module each.

Scenes are registered in manimx.json (``manimx generate scene`` does it),
not imported here: each scene is imported on its own, so a broken scene
never breaks the others.
"""
'''

ANIMATION_SCENE = '''"""{cls}: an animation scene.

Render it alone with ``manimx render {cls}``, or the whole timeline with
``manimx render``.
"""

from __future__ import annotations

from manimx.animation import DOWN, PI, UP, Circle, FadeIn, FadeOut, Scene, Square


class {cls}(Scene):
    def construct(self) -> None:
        square = Square(2.0, color="#5ea3d0", fill_opacity=0.8)
        circle = Circle(1.0, color="#e07a5f", fill_opacity=0.8).shift(DOWN * 2)

        self.play(FadeIn(square, shift=UP * 0.5))
        self.play(square.animate.rotate(PI / 4).shift(UP), FadeIn(circle), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(square), FadeOut(circle))
'''

SIMULATION_SCENE = '''"""{cls}: a simulation scene.

The step function is the whole simulation: it moves nodes by ``dt``
however you like. It runs once when the scene is prepared and is stored
as keyframes, so rendering never simulates again. Sounds played inside
the step start at the simulated time.
"""

from __future__ import annotations

import numpy as np

from manimx.animation import DOWN, LEFT, UP, Dot, FadeIn, FadeOut, Rectangle, Scene, Simulation

GRAVITY = 9.81
FLOOR = -2.5


def click(sample_rate: int = 48_000) -> np.ndarray:
    """A short percussive click: a decaying sine (a mono wave array)."""
    t = np.arange(int(0.08 * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.02)).astype(np.float32)


class {cls}(Scene):
    def construct(self) -> None:
        floor = Rectangle(10, 0.08, color="#3d4a6b", fill_opacity=1).move_to(DOWN * 2.79)
        ball = Dot(0.25, color="#f2cc8f").move_to(LEFT * 3 + UP * 2)
        velocity = np.array([1.2, 0.0, 0.0])
        sound = click()

        def step(dt: float) -> None:
            velocity[1] -= GRAVITY * dt
            ball.shift(velocity * dt)
            if ball.position.y.value < FLOOR and velocity[1] < 0:
                ball.shift(UP * (FLOOR - ball.position.y.value))
                impact = -velocity[1]
                velocity[1] = impact * 0.75
                simulation.play_wave(sound, volume=min(1.0, impact / 8.0))

        simulation = Simulation(step, dt=1 / 480, duration=4.0)

        self.play(FadeIn(floor), FadeIn(ball), run_time=0.5)
        self.play(simulation)
        self.play(FadeOut(floor), FadeOut(ball), run_time=0.5)
'''

KINDS = {"animation": ANIMATION_SCENE, "simulation": SIMULATION_SCENE}

PYPROJECT = """[project]
name = "{package}"
version = "0.1.0"
description = "A manimx project."
requires-python = ">=3.11"
dependencies = ["manimx"]

[tool.uv]
package = false
{sources}"""

UV_SOURCES = """
[tool.uv.sources]
manimx = {{ path = "{path}", editable = true }}
"""

GITIGNORE = """renders/
.venv/
__pycache__/
"""

README = """# {name}

A [manimx](https://github.com/ManimCommunity/manim) project.

```sh
uv sync                              # install manimx into .venv
uv run manimx generate scene Intro   # add a scene (file, registry, timeline)
uv run manimx render --preview       # quick low-resolution video of the timeline
uv run manimx render                 # full quality
```

Scenes live in `scenes/`, the timeline and render settings in `manimx.json`.
"""
