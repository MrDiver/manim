"""``python -m manimx.main``: render Manim's logo and show it, to check an install."""

from __future__ import annotations

from PIL import Image

from .drivers import render_image
from .examples.manim_logo import config, manim_logo


def main() -> None:
    """Render Manim's logo and show it."""
    Image.fromarray(render_image(manim_logo(), config=config())).show()


if __name__ == "__main__":
    main()
