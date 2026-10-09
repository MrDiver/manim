from __future__ import annotations

from PIL import Image

from .examples.manim_logo import config, manim_logo
from .renderer import RenderService


def main() -> None:
    """Render Manim's logo scene with the WebGPU renderer and show it."""
    image = RenderService().render(manim_logo().snapshot(), config())
    Image.fromarray(image).show()


if __name__ == "__main__":
    main()
