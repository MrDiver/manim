"""Voice: an animation timed by a recorded voice.

A remake of reanimate's voice example
(https://reanimate.readthedocs.io/en/latest/voice/, "Setting up
triggers"), with its recording: assets/audio/voice.m4a, on the
timeline's audio track. ``manimx transcribe assets/audio/voice.m4a
--text script.txt`` (or Transcribe in ``manimx ui``) timed its words into
voice.transcript.json, and every word is an event: ``voice.circle``,
``voice.circle.2`` the second time it is said.

The scene is written along the transcript as it is now: it waits until
each word's time and puts a ``cue`` there. The cues keep it in sync when
the recording changes: record again, slower or faster, transcribe again,
and the clip's time bends so every cue lands on its word, without
touching this file. ``start`` and ``stop`` fly the circle and the square
from one word to another while the words go on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manimx.animation import Animation, Group, Scene, linear
from manimx.userland import (
    PI,
    Circle,
    CubicBezier,
    FadeOut,
    Line,
    MoveAlongPath,
    Rectangle,
    Rotate,
    Square,
    Typst,
)
from manimx.userland.markers import bounds
from manimx.voice import Transcript

PAPER = "#fafafa"
INK = "#000000"
WORDS = '#set text(size: 66pt, font: "New Computer Modern", fill: black)'
TRANSCRIPT = Transcript.load(Path(__file__).parents[1] / "assets" / "audio" / "voice.transcript.json")
UNIT = 8 / 9
"""Scene units per unit of reanimate's 16 by 9 screen (ours is 8 high)."""
EDGE = 8.0
"""Beyond the frame's sides: where the line ends."""
TICK = 0.39
"""Half the height of the ticks around the word."""
GAP = 0.22
"""Space between the word and its ticks."""
DOT = 0.27
SIDE = 0.9


def arc(start: tuple[float, float], top: tuple[float, float], end: tuple[float, float]) -> CubicBezier:
    """reanimate's quadratic Bezier flight path through ``top``'s pull, as a cubic one, in our units."""
    p0, p1, p2 = (np.array(point) * UNIT for point in (start, top, end))
    return CubicBezier(*(point.tolist() for point in (p0, p0 + 2 / 3 * (p1 - p0), p2 + 2 / 3 * (p1 - p2), p2)))


def word_on_line(text: str) -> Group:
    """``|text|`` in a gap of the line across the frame (an empty text: ``||``)."""
    parts = []
    half = 0.0
    if text:
        word = Typst(text, WORDS)
        low, high = np.asarray(bounds(word).value)
        word.shift(-(low + high) / 2 * np.array([1.0, 1.0, 0.0]))
        half = (high[0] - low[0]) / 2
        parts.append(word)
    side = half + GAP
    style = {"color": INK, "stroke_width": 6}
    parts += [
        Line((-EDGE, 0, 0), (-side, 0, 0), **style),
        Line((side, 0, 0), (EDGE, 0, 0), **style),
        Line((-side, -TICK, 0), (-side, TICK, 0), **style),
        Line((side, -TICK, 0), (side, TICK, 0), **style),
    ]
    return Group(*parts, name=f"word {text}")


class Voice(Scene):
    def construct(self) -> None:
        names = TRANSCRIPT.event_names("voice")
        said = dict(zip(names, (word.start for word in TRANSCRIPT.words), strict=True))
        self.add(Rectangle(15, 8.5, color=PAPER, fill_opacity=1, stroke_width=0))
        shown = word_on_line("")
        self.add(shown)

        flights = {}

        def fly(node, path: CubicBezier, spin: float = 0.0) -> None:
            """Fly ``node`` along ``path`` from now until its ``land``."""
            node.move_to(path.controls.value[0])
            self.add(node)
            animations: list[Animation] = [MoveAlongPath(node, path, by_length=False, rate=linear)]
            if spin:
                animations.append(Rotate(node, spin, rate=linear))
            flights[node.name] = self.start(*animations)

        def land(node) -> None:
            self.stop(flights.pop(node.name))
            self.remove(node)

        def flash() -> None:
            shade = Rectangle(15, 8.5, color=INK, fill_opacity=0.5, stroke_width=0)
            self.start(FadeOut(shade), run_time=0.9)

        circle = Circle(DOT, color=INK, fill_opacity=1, stroke_width=0, name="circle")
        square = Square(SIDE, color=INK, stroke_width=5, name="square")
        three = [Circle(DOT, color=INK, fill_opacity=1, stroke_width=0, name=f"circle {index}") for index in range(3)]
        up_left = arc((6, -DOT / UNIT), (0, 6), (-6, -DOT / UNIT))
        down_right = arc((-6, SIDE / 2 / UNIT), (0, -6), (6, SIDE / 2 / UNIT))
        up_right = arc((-6, -DOT / UNIT), (0, 6), (6, -DOT / UNIT))
        tumbles = (said["voice.square.2"] - said["voice.square"]) * 0.75 * PI  # about 135 degrees a second

        def destroy(dot) -> None:
            land(dot)
            flash()

        effects = {
            "voice.flash": flash,
            "voice.circle": lambda: fly(circle, up_left),
            "voice.square": lambda: fly(square, down_right, -tumbles),
            "voice.square.2": lambda: land(square),
            "voice.circle.2": lambda: land(circle),
            "voice.circles": lambda: [fly(dot, up_right) for dot in three],
            "voice.flash.2": lambda: destroy(three[0]),
            "voice.flash.3": lambda: destroy(three[1]),
            "voice.flash.4": lambda: destroy(three[2]),
        }

        for name, word in zip(names, TRANSCRIPT.words, strict=True):
            self.wait_until(word.start)
            self.cue(name)
            self.remove(shown)
            shown = word_on_line(word.text.strip(".,!?"))
            self.add(shown)
            if name in effects:
                effects[name]()
        self.wait_until(TRANSCRIPT.words[-1].end + 1.5)
