"""Colors and the caption every showcase scene starts with.

Colors are chosen here, by the project: manimx's classes take colors but
never pick them.
"""

from __future__ import annotations

from manimx.animation import Group
from manimx.userland import DOWN, UP, Typst

INK = "#ece6e2"
MUTED = "#8a93a8"
RED = "#e07a5f"
SAND = "#f2cc8f"
GREEN = "#87c2a5"
BLUE = "#5ea3d0"
VIOLET = "#9b8ec4"
NAVY = "#3d4a6b"
PALETTE = (RED, SAND, GREEN, BLUE, VIOLET)


def caption(title: str, subtitle: str) -> Group:
    """A title and a line about what the scene shows, at the top."""
    heading = Typst(title, f'#set text(size: 30pt, weight: "bold", fill: rgb("{INK}"))')
    line = Typst(subtitle, f'#set text(size: 16pt, fill: rgb("{MUTED}"))').shift(DOWN * 0.45)
    return Group(heading, line).shift(UP * 3.3)
