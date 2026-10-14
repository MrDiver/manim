"""Dice: six dice rolled into their bowls, then 100 000 of them.

A die here has three faces (one, two or three pips) and three bowls under
it, one per number. Each die is rolled at the top: it flips over and over
(it squashes to its edge and opens again, showing another face each time)
and lands on its number, then drops into that bowl with a bounce, on top
of the dice already there.

First one panel: six dice, one after another. Then the camera pulls back
and shows that this panel is one of ten thousand, all rolling at the same
time: about 100 000 dice, half a million shapes, on screen at the end. It
is a stress test as much as an animation.

``Die`` is the object (a group: its body and its pips), ``Roll`` the
animation made for it: it changes the die's height (the flip) and which
pips show, as keyframes the core evaluates without calling Python, so
100 000 rolls cost no more Python per frame than one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from manimx.animation import Animation, Change, Group, Scene, Target
from manimx.animation.signals.contributions import Keyframes
from manimx.userland import Dot, Polyline, RoundedRectangle
from manimx.userland.rate_functions import ease_out_bounce, smooth

BACKGROUND = "#000000"
COLORS = ("#e5b51e", "#24a865", "#4a4999")  # yellow, green, blue, as in the reference
SIDE = 0.5
"""A die's side."""
GAP = 0.04
"""Between stacked dice."""
BOWL_WIDTH = 1.0
BOWL_DEPTH = 0.32
FACES = 3
"""Numbers on a die, and bowls per panel."""

PANEL_WIDTH = 3.5
PANEL_HEIGHT = 6.0
COLUMNS, ROWS = 168, 60
"""Panels around the first one: 10 080 of them."""
DICE_PER_PANEL = 10

FIRST_DICE = 6
"""Dice rolled in the first panel before the camera pulls back."""


class Die(Group):
    """A die with ``FACES`` faces: a rounded body and the pips of its
    number. Its three pips sit on the diagonal; the number decides which
    show (one: the middle; two: the ends; three: all)."""

    PIPS = {1: (False, True, False), 2: (True, False, True), 3: (True, True, True)}

    def __init__(self, color: str, value: int = 1, name: str | None = None) -> None:
        self.body = RoundedRectangle(SIDE, SIDE, corner_radius=0.09, color=color, fill_opacity=1, stroke_width=0)
        offset = SIDE * 0.27
        self.pips = [
            Dot(radius=SIDE * 0.075, color="#ffffff").move_to((x, -x, 0.0)) for x in (-offset, 0.0, offset)
        ]
        super().__init__(self.body, *self.pips, name=name)
        self.show(value)

    def show(self, value: int) -> Die:
        """Show ``value`` at once (while composing the scene)."""
        for pip, shown in zip(self.pips, self.PIPS[value], strict=True):
            pip.set_alpha(1.0 if shown else 0.0)
        return self


@dataclass(eq=False)
class Roll(Animation):
    """Roll ``die`` onto ``value``: it flips ``flips`` times (about its
    horizontal axis: it squashes to its edge and opens again) and shows
    the faces ``faces`` in between, one per flip, then ``value``.

    The flip is the die's height, ``|cos|`` of the turned angle; a face
    changes while the die is on its edge, where the change cannot be
    seen. Both are keyframes (``planned_contributions``): the core plays
    them without Python, whatever the number of dice."""

    die: Die
    value: int
    faces: Sequence[int] = ()
    flips: int = 3
    run_time: float = 1.0
    rate: Callable[[float], float] = staticmethod(smooth)  # type: ignore[assignment]
    samples: int = 12
    """Keyframes per flip for the height."""
    _times: list[float] = field(init=False, repr=False)
    _height: list[float] = field(init=False, repr=False)
    _pips: list[list[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shown = [*self.faces[: self.flips - 1]]
        while len(shown) < self.flips - 1:
            shown.append(1 + len(shown) % FACES)
        shown.append(self.value)  # the face of each flip: the last is the number rolled
        times: list[float] = []
        for index in range(self.flips * self.samples + 1):
            times.append(index / (self.flips * self.samples))
        for flip in range(self.flips):  # just before and after each edge: the face changes there
            edge = (flip + 0.5) / self.flips
            times += [edge - 1e-4, edge + 1e-4]
        times.sort()
        self._times = [t * self.run_time for t in times]
        self._height = [max(abs(math.cos(math.pi * self.flips * t)), 0.04) for t in times]
        face = [shown[min(int(t * self.flips + 0.5), self.flips - 1)] for t in times]  # a face from edge to edge
        self._pips = [[1.0 if Die.PIPS[number][pip] else 0.0 for number in face] for pip in range(3)]

    def targets(self) -> Sequence[Target]:
        return [Target(self.die.scale.axes[1], Change.ABSOLUTE), *(Target(pip.alpha, Change.ABSOLUTE) for pip in self.die.pips)]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        t = alpha * self.run_time
        return [float(np.interp(t, self._times, values)) for values in (self._height, *self._pips)]

    def planned_contributions(self) -> Sequence[Keyframes]:
        return [Keyframes(self.run_time, self._times, values) for values in (self._height, *self._pips)]

    def nodes(self) -> Sequence[Die]:
        return (self.die,)


def bowls(center: tuple[float, float]) -> list[Polyline]:
    """The three bowls of a panel, side by side, their floor at ``center``'s height."""
    x0, y0 = center
    lip = 0.12
    shapes = []
    for index in range(FACES):
        x = x0 + (index - 1) * BOWL_WIDTH
        half = BOWL_WIDTH / 2 - 0.04
        floor = half - lip
        points = [(x - half, y0 + BOWL_DEPTH, 0.0), (x - floor, y0, 0.0), (x + floor, y0, 0.0), (x + half, y0 + BOWL_DEPTH, 0.0)]
        shapes.append(Polyline(points, color="#ffffff", stroke_width=3))
    return shapes


@dataclass
class Panel:
    """One set of three bowls and the dice rolled into them."""

    center: tuple[float, float]
    stacks: list[int] = field(default_factory=lambda: [0] * FACES)

    def top(self) -> tuple[float, float, float]:
        """Where a die is rolled: above the middle bowl."""
        return (self.center[0], self.center[1] + PANEL_HEIGHT / 2 - SIDE, 0.0)

    def landing(self, value: int) -> tuple[float, float, float]:
        """Where the next die of ``value`` lands: on the stack in its bowl."""
        floor = self.center[1] - PANEL_HEIGHT / 2 + 0.5
        height = self.stacks[value - 1]
        self.stacks[value - 1] += 1
        return (self.center[0] + (value - 2) * BOWL_WIDTH, floor + GAP + SIDE / 2 + height * (SIDE + GAP), 0.0)

    def shapes(self) -> list[Polyline]:
        return bowls((self.center[0], self.center[1] - PANEL_HEIGHT / 2 + 0.5))


def zoom_rate(start: float, end: float) -> Callable[[float], float]:
    """A rate that makes a frame height go from ``start`` to ``end`` at a
    steady zoom (the same factor each moment), eased in and out: the
    height grows exponentially, not linearly."""
    ratio = end / start

    def rate(t: float) -> float:
        return (ratio ** smooth(t) - 1.0) / (ratio - 1.0)

    return rate


class Dice(Scene):
    def construct(self) -> None:
        rng = np.random.default_rng(6)
        height = PANEL_HEIGHT + 0.6
        self.camera.set_frame_height(height)
        moments: list[tuple[float, Callable[[], None]]] = []  # (scene time, action)

        def roll(panel: Panel, at: float, run_time: float, drop: float) -> float:
            """Roll a die in ``panel`` from scene time ``at`` and drop it
            into its bowl; returns when it has landed."""
            value = int(rng.integers(1, FACES + 1))
            die = Die(COLORS[int(rng.integers(len(COLORS)))]).move_to(panel.top())
            flips = int(rng.integers(3, 6))
            faces = [int(face) for face in rng.integers(1, FACES + 1, size=flips)]
            landing = panel.landing(value)
            moments.append((at, lambda: self.start(Roll(die, value, faces, flips=flips), run_time=run_time)))
            moments.append((at + run_time + 0.1, lambda: self.start(die.animate.move_to(landing), run_time=drop, rate=ease_out_bounce)))
            return at + run_time + 0.1 + drop

        # One panel: six dice, one after another.
        first = Panel((0.0, 0.0))
        self.add(*first.shapes())
        at = 0.5
        for _ in range(FIRST_DICE):
            at = roll(first, at, run_time=1.2, drop=0.8) + 0.3

        # Every other panel, all rolling at once while the camera pulls back.
        pull_back = 24.0
        begin = at + 0.5
        panels = [first]
        for row in range(ROWS):
            for column in range(COLUMNS):
                center = ((column - COLUMNS // 2) * PANEL_WIDTH, (row - ROWS // 2) * PANEL_HEIGHT)
                if center != (0.0, 0.0):
                    panels.append(Panel(center))
        others = [shape for panel in panels[1:] for shape in panel.shapes()]
        moments.append((begin, lambda: self.add(*others)))  # out of view until the camera pulls back
        for panel in panels:
            count = DICE_PER_PANEL - (FIRST_DICE if panel is first else 0)
            start = begin + float(rng.uniform(0.0, 1.5))
            step = (pull_back - 2.5 - (start - begin)) / max(count, 1)
            for index in range(count):
                roll(panel, start + index * step, run_time=float(rng.uniform(0.7, 1.1)), drop=0.6)
        wide = ROWS * PANEL_HEIGHT * 1.02
        moments.append((begin, lambda: self.start(self.camera.animate.set_frame_height(wide), run_time=pull_back, rate=zoom_rate(height, wide))))

        for time, action in sorted(moments, key=lambda moment: moment[0]):  # stable: as listed at equal times
            self.wait_until(time)
            action()
        self.wait_until(begin + pull_back + 1.5)
