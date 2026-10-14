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
touching this file. ``start`` and ``stop`` fly the circles and the square
from one word to another while the words go on.

As in reanimate, every flying shape is clipped to one half of the frame
(the circles to the upper, the square to the lower) and drawn behind the
line, so it rises out of the line and sinks back into it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from manimx.animation import Animation, ClippingGroup, Group, Node, Scene, linear
from manimx.userland import (
    TAU,
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
from manimx.userland.rate_functions import ease_out_quad
from manimx.voice import Transcript

PAPER = "#fafafa"
INK = "#000000"
WORDS = '#set text(size: 66pt, font: "New Computer Modern", fill: black)'
TRANSCRIPT = Transcript.load(Path(__file__).parents[1] / "assets" / "audio" / "voice.transcript.json")
UNIT = 8 / 9
"""Scene units per unit of reanimate's 16 by 9 screen (ours is 8 high)."""
WIDTH, HEIGHT = 16 * UNIT, 9 * UNIT
RADIUS = 0.3 * UNIT
SIDE = 1.0 * UNIT
STROKE = {"color": INK, "stroke_width": 6}


def arc(start: tuple[float, float], top: tuple[float, float], end: tuple[float, float]) -> CubicBezier:
    """reanimate's quadratic Bezier flight path pulled towards ``top``, as a cubic one, in our units."""
    p0, p1, p2 = (np.array(point) * UNIT for point in (start, top, end))
    return CubicBezier(*(point.tolist() for point in (p0, p0 + 2 / 3 * (p1 - p0), p2 + 2 / 3 * (p1 - p2), p2)))


def half(sign: float) -> Rectangle:
    """The upper (+1) or lower (-1) half of the frame, as a clipping mask."""
    return Rectangle(WIDTH, HEIGHT / 2, fill_opacity=1, stroke_width=0).move_to((0.0, sign * HEIGHT / 4, 0.0))


def word_box(text: str) -> Group:
    """reanimate's textHandler: the word on a paper box over the line,
    between two ticks (an empty word: just the ticks)."""
    parts: list[Node] = []
    width = 0.0
    if text:
        word = Typst(text, WORDS)
        low, high = np.asarray(bounds(word).value)
        word.shift(-(low + high) / 2 * np.array([1.0, 1.0, 0.0]))
        width = high[0] - low[0]
        parts.append(word)
    side = (width + 0.5 * UNIT) / 2
    paper = Rectangle(2 * side, UNIT, color=PAPER, fill_opacity=1, stroke_width=0)
    ticks = [Line((x, -0.5 * UNIT, 0), (x, 0.5 * UNIT, 0), **STROKE) for x in (-side, side)]
    return Group(paper, *parts, *ticks, name=f"word {text}")


class Voice(Scene):
    def construct(self) -> None:
        names = TRANSCRIPT.event_names("voice")
        words = dict(zip(names, TRANSCRIPT.words, strict=True))
        self.add(Rectangle(15, 8.5, color=PAPER, fill_opacity=1, stroke_width=0))

        circle = Circle(RADIUS, color=INK, fill_opacity=1, stroke_width=0, name="circle")
        square = Square(SIDE, **STROKE, name="square")
        three = [Circle(RADIUS, color=INK, fill_opacity=1, stroke_width=0, name=f"circle {index}") for index in range(3)]
        clipped = {node.name: ClippingGroup(half(-1 if node is square else 1), node, name=f"clipped {node.name}") for node in [circle, square, *three]}
        # Their place in the drawing order: behind the line and the words
        # (nodes are drawn in the order they were first added).
        for group in clipped.values():
            self.add(group)
            self.remove(group)
        self.add(Line((-WIDTH, 0, 0), (WIDTH, 0, 0), **STROKE))

        flights = {}

        def fly(node: Node, path: CubicBezier, turn: float = 0.0) -> None:
            """Fly ``node`` along ``path`` from now until it lands."""
            node.move_to(path.controls.value[0])
            self.add(clipped[node.name])
            animations: list[Animation] = [MoveAlongPath(node, path, by_length=False, rate=linear)]
            if turn:
                animations.append(Rotate(node, turn, rate=linear))
            flights[node.name] = self.start(*animations)

        def land(node: Node) -> None:
            self.stop(flights.pop(node.name))
            self.remove(clipped[node.name])

        def flash(word: str) -> None:
            shade = Rectangle(15, 8.5, color=INK, fill_opacity=0.7, stroke_width=0)
            spoken = words[word]
            self.start(FadeOut(shade), run_time=spoken.end - spoken.start, rate=ease_out_quad)  # 0.7 (1 - t)^2

        def middle(word: str) -> float:
            return (words[word].start + words[word].end) / 2

        up_left = arc((6, -0.3), (0, 6), (-6, -0.3))
        up_right = arc((-6, -0.3), (0, 6), (6, -0.3))
        down_right = arc((-6, 0.5), (0, -6), (6, 0.5))  # reanimate's square path turned by 180 degrees

        # What happens when: (time, action); the words put their cues.
        moments: list[tuple[float, Callable[[], None]]] = []
        for name in ("voice.flash", "voice.flash.2", "voice.flash.3", "voice.flash.4"):
            moments.append((words[name].start, lambda name=name: flash(name)))
        moments += [
            (words["voice.circle"].start, lambda: fly(circle, up_left)),
            (middle("voice.circle.2"), lambda: land(circle)),
            (words["voice.square"].start, lambda: fly(square, down_right, -TAU)),
            (middle("voice.square.2"), lambda: land(square)),
        ]
        for index, dot in enumerate(three):
            moments.append((words["voice.circles"].start + 0.2 * index, lambda dot=dot: fly(dot, up_right)))
            moments.append((words[f"voice.flash.{index + 2}"].start, lambda dot=dot: land(dot)))

        shown = word_box("")
        self.add(shown)

        def say(name: str) -> None:
            nonlocal shown
            self.cue(name)
            self.remove(shown)
            shown = word_box(words[name].text.strip(".,!?"))
            self.add(shown)

        moments += [(word.start, lambda name=name: say(name)) for name, word in words.items()]
        for time, action in sorted(moments, key=lambda moment: moment[0]):  # stable: as listed at equal times
            self.wait_until(time)
            action()
        self.wait_until(TRANSCRIPT.words[-1].start + 2.0)
