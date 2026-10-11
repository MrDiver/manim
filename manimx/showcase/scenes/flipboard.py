"""Flipboard: a departures board whose cells count through the alphabet.

CycleCharacters turns one Code into another cell by cell: each cell
counts up or down to its new character, all at the same speed, or
column after column with a lag. Flipboard does the same with every step
folded like the flap of a split-flap display. Plain text is Code
without a language. TypeCode types text in behind a cursor, Untype takes
it back, and Unwrite undoes Write.
"""

from __future__ import annotations

from manimx.animation import Group, Scene
from manimx.userland import (
    DOWN,
    LEFT,
    UP,
    Blink,
    Code,
    CycleCharacters,
    FadeIn,
    FadeOut,
    Flipboard,
    Rectangle,
    TypeCode,
    Typst,
    Untype,
    Unwrite,
    Write,
)

from .style import INK, MUTED, NAVY, SAND, caption

BOARD = f'#set text(fill: rgb("{SAND}"))\n#show raw: set text(size: 30pt, weight: "bold")'
TYPED = f'#set text(fill: rgb("{MUTED}"))\n#show raw: set text(size: 20pt)'
TOP_LEFT = LEFT * 3.6 + UP * 1.9
FLAPS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:."
"""What the board's flaps carry: no lowercase, as on a real board."""

TIMES = [
    """\
10:42  BERLIN      ON TIME
10:55  HAMBURG     DELAYED
11:05  MUNICH      GATE 4
11:20  COLOGNE     ON TIME""",
    """\
11:35  LEIPZIG     ON TIME
11:48  DRESDEN     GATE 7
12:02  FRANKFURT   DELAYED
12:15  STUTTGART   ON TIME""",
    """\
12:30  VIENNA      BOARDING
12:44  PRAGUE      ON TIME
13:01  ZURICH      GATE 2
13:17  AMSTERDAM   CANCELLED""",
]


def board(text: str) -> Code:
    return Code(text, None, BOARD).move_to(TOP_LEFT)


def flaps(code: Code, columns: int) -> Group:
    """A dark flap behind every cell, with its hinge across the middle."""
    grid = code.grid
    cells = Group(name="flaps")
    for row in range(len(code.rows)):
        for column in range(columns):
            left, bottom, right, top = grid.cell(row, column)
            middle = code.in_parent(((left + right) / 2, (bottom + top) / 2))
            cells.add(Rectangle(grid.advance * 0.9, grid.pitch * 0.88, fill_color=NAVY, fill_opacity=0.55, stroke_width=0.0).move_to(middle))
            cells.add(Rectangle(grid.advance * 0.9, 0.012, fill_color="#0b1020", fill_opacity=1.0, stroke_width=0.0).move_to(middle))
    return cells


class Departures(Scene):
    def construct(self) -> None:
        title = caption("Flipboard", "CycleCharacters and Flipboard count cells through the alphabet; TypeCode, Untype, Unwrite")
        self.play(FadeIn(title), run_time=0.6)

        heading = Typst("Departures", f'#set text(size: 28pt, weight: "bold", fill: rgb("{INK}"))').move_to(UP * 2.45 + LEFT * 2.55)
        self.play(Write(heading), run_time=1.0)

        current = board(TIMES[0])
        background = flaps(current, max(len(row) for text in TIMES for row in text.split("\n")))
        self.play(FadeIn(background), FadeIn(current), run_time=0.8)
        self.wait(0.6)

        for text, change in zip(TIMES[1:], (CycleCharacters, Flipboard), strict=True):
            following = board(text)
            self.play(change(current, following, alphabet=FLAPS, lag_ratio=0.08 if change is Flipboard else 0.0), run_time=2.4)
            current = following
            self.wait(0.8)

        notice = Code("Next update in 5 minutes. Mind the gap.", None, TYPED).move_to(LEFT * 3.6 + DOWN * 1.6)
        cursor = notice.cursor(fill_color=INK, fill_opacity=1.0, stroke_width=0.0)
        self.play(TypeCode(notice, cursor, jitter=0.6, seed=4))
        self.play(Blink(cursor, times=3), run_time=1.5)
        self.play(Untype(notice, cursor, speed=40.0))
        self.play(FadeOut(cursor), run_time=0.2)

        self.play(Unwrite(heading), run_time=1.0)
        self.play(*(FadeOut(node) for node in (current, background, title)), run_time=0.6)
