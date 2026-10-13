"""Events: a scene that waits, like a presenter waiting for a click.

``wait_for(event)`` ends a segment: the clip holds its last frame until
the event fires, then plays on. The project sets each event's time (under
"events" in manimx.json); in the preview, the clip's track has an events
row with a pill per event, dragged to time it. A presenter's click can
fire the same events live later.

``emit(event)`` fires an event from inside a scene, here "answered" when
the answer is drawn, which other clips can wait for.
"""

from __future__ import annotations

from manimx.animation import Scene
from manimx.userland import (
    DOWN,
    UP,
    Create,
    FadeIn,
    FadeOut,
    Integer,
    RegularPolygon,
    Typst,
    Write,
)

from .style import BLUE, INK, MUTED, SAND, caption

TEXT = f'#set text(size: 34pt, fill: rgb("{INK}"))'


class Events(Scene):
    def construct(self) -> None:
        title = caption("Events", "wait_for() holds the clip until its event fires; drag the pills to time them")
        self.play(FadeIn(title), run_time=0.6)

        question = Typst("How many sides has a hexagon?", TEXT).shift(UP * 1.4)
        self.play(Write(question), run_time=1.2)
        self.wait_for("answer")  # the presenter asks the room first

        hexagon = RegularPolygon(6, 1.4, color=SAND, stroke_width=6.0).shift(DOWN * 0.6)
        count = Integer(0, f'#set text(size: 40pt, fill: rgb("{BLUE}"))').shift(DOWN * 0.6)
        self.add(count)
        self.play(Create(hexagon), count.animate.set_value(6), run_time=1.6)
        self.emit("answered")
        note = Typst("six", f'#set text(size: 24pt, fill: rgb("{MUTED}"))').shift(DOWN * 2.5)
        self.play(FadeIn(note, shift=UP * 0.2), run_time=0.5)
        self.wait_for("next")

        self.play(*(FadeOut(node) for node in (title, question, hexagon, count, note)), run_time=0.6)
