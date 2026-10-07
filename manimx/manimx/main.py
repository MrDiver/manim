from __future__ import annotations

from PIL import Image

from .examples.manim_logo import config, manim_logo
from .renderer import WGPURenderer


def main() -> None:
    """Render Manim's logo scene with the WebGPU renderer and show it."""
    image = WGPURenderer().render(manim_logo(), config())
    Image.fromarray(image).show()


if __name__ == "__main__":
    main()
