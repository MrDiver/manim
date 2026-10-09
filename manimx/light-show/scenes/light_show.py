"""LightShow: a camera flight through a club, with moving heads on the beat.

32 seconds, 16 bars at 120 BPM, in five sections. Every section starts
with ``wait_until`` at its bar, so it lands on the beat however long the
one before took, and plays animations of ``lights.py``: functions of
their progress, which fit the beat because their counts fit the
section (32 strobe flashes in four bars are eighth notes).

====  ========  ==========================================================
bars  section   the lights
====  ========  ==========================================================
0-3   intro     the heads open one per beat, deep blue, then circle,
                pulsing with the kick
4-7   build     cyan and magenta circles around the sculpture, pulsing;
                a strobe speeds up into the drop
8-11  drop      sweeps across the floor, strobing on eighths, a new color
                every bar
12-14 ballyhoo  crossing figure-eights, a rainbow across the rig, a chase
15    hit       everything on the sculpture in white, then blackout
====  ========  ==========================================================

Meanwhile the camera flies in through the arches, circles the
sculpture, climbs to a view from above and ends facing the stage, where
``Finale`` continues. The mirror ball, the sculpture, the arches, the
floor grid and the speaker cones move on their own (derived signals of
``scene_time``, see ``rig.build_stage``).
"""

from __future__ import annotations

import math

from manimx.animation import Scene
from manimx.userland import IN, OUT

from .lights import Aim, Chase, Fade, Flash, Pulse, Rainbow, Recolor, Strobe, circle, still
from .music import BAR, BEAT, DROP, SHOW_LENGTH, show_track
from .rig import AMBER, CYAN, MAGENTA, RED, SCULPTURE, VIOLET, WHITE, MovingHead, Stage, build_stage, fly, on_floor

FULL = 700.0
"""A moving head's intensity when fully open."""

CAMERA_PATH = [
    (0.0, (0.0, -36.0, 1.3), (0.0, 0.0, 2.6)),
    (6.0, (0.0, -24.0, 1.6), (0.0, 0.0, 2.6)),
    (11.0, (0.4, -12.0, 2.0), (0.0, 0.0, 2.4)),
    (15.0, (4.8, -6.0, 2.3), (0.0, 0.0, 2.2)),
    (19.0, (8.0, 1.5, 3.0), (0.0, 0.0, 2.2)),
    (23.0, (-2.0, 7.0, 4.2), (0.0, -2.0, 1.2)),
    (27.5, (-11.0, -9.0, 11.0), (0.0, 1.0, 1.5)),
    (SHOW_LENGTH, (0.0, -13.0, 3.4), (0.0, 8.6, 3.7)),
]
FLOOR = on_floor(*SCULPTURE[:2])
DROP_COLORS = (RED, AMBER, VIOLET, CYAN)


class LightShow(Scene):
    def construct(self) -> None:
        stage = build_stage(self.time)
        heads = stage.heads
        for i, head in enumerate(heads):
            head.point_at(circle(FLOOR, ring_radius(i), 1.0, phase(i, heads))(0.0))
        self.add(stage.root)
        self.camera.set_field_of_view(math.radians(48))
        self.camera.near, self.camera.far = 0.1, 160.0
        fly(self.camera, self.time, CAMERA_PATH)
        self.play_wave(show_track())

        # Intro: one head opens per beat, then they circle with the kick.
        self.play(*(Flash(head.light, peak=FULL, rest=0.35 * FULL, decay=0.5) for head in heads), lag_ratio=1.0, run_time=BEAT)
        self.wait_until(len(heads) * BEAT)
        self.play(*intro(heads), run_time=2 * BAR - len(heads) * BEAT)

        self.wait_until(4 * BAR)
        self.play(*build(heads), run_time=3 * BAR)
        self.play(*into_the_drop(heads), run_time=BAR)

        self.wait_until(DROP)
        self.play(*drop(stage.front, stage.back), run_time=4 * BAR)

        self.wait_until(12 * BAR)
        self.play(*ballyhoo(heads), run_time=3 * BAR)

        self.wait_until(15 * BAR)
        self.play(*hit(heads), *clear_the_floor(stage), run_time=BAR)


def ring_radius(i: int) -> float:
    return 3.5 + 1.5 * (i % 2)


def phase(i: int, heads: list[MovingHead]) -> float:
    return i * math.tau / len(heads)


def intro(heads: list[MovingHead]):
    """Bars 2-3 (after the heads opened): slow circles around the
    sculpture, pulsing with the kick, half cyan, half magenta."""
    for i, head in enumerate(heads):
        yield Aim(head, circle(FLOOR, ring_radius(i), 0.2, phase(i, heads)), blend=0.0)
        yield Pulse(head.light, beats=7, peak=FULL, rest=0.35 * FULL, decay=0.4)
        yield Recolor(head.light, [CYAN if i % 2 else MAGENTA])


def build(heads: list[MovingHead]):
    """Bars 4-6: circles around the sculpture, alternating directions,
    pulsing on every beat."""
    for i, head in enumerate(heads):
        yield Aim(head, circle(FLOOR, 2.5 + 2.0 * (i % 3), 0.85 if i % 2 else -0.85, i * 1.3), blend=0.1)
        yield Pulse(head.light, beats=12, peak=FULL, rest=0.25 * FULL, decay=0.4)
        yield Recolor(head.light, [CYAN if i % 2 else MAGENTA])


def into_the_drop(heads: list[MovingHead]):
    """Bar 7: all white on the sculpture, a strobe speeding up from 4 to
    16 flashes a second."""
    for head in heads:
        yield Aim(head, still(SCULPTURE), blend=0.5)
        yield Strobe(head.light, flashes=20, level=0.8 * FULL, duty=0.35, speedup=4.0)
        yield Recolor(head.light, [WHITE])


def drop(front: list[MovingHead], back: list[MovingHead]):
    """Bars 8-11: the front sweeps fans across the floor (one swing a
    bar), strobing on eighths; the back circles fast and hits on the
    kicks; a new color every bar."""
    for i, head in enumerate(front):
        spread = (i - 2) / 2  # -1 .. 1 across the truss

        def sweep(u: float, spread: float = spread):
            swing = math.sin(math.tau * 4 * u)
            return on_floor(9.0 * spread * swing + 2.0 * spread, -1.5 + 4.0 * math.cos(math.tau * 2 * u))

        yield Aim(head, sweep, blend=0.05)
        yield Strobe(head.light, flashes=32, level=FULL, rest=0.08 * FULL, duty=0.45)
        yield Recolor(head.light, DROP_COLORS)
    for i, head in enumerate(back):
        yield Aim(head, circle(FLOOR, 2.0, 2.8 if i % 2 else -2.8, i * math.pi / 2), blend=0.05)
        yield Pulse(head.light, beats=16, peak=FULL, rest=0.1 * FULL, decay=0.25)
        yield Recolor(head.light, DROP_COLORS[2:] + DROP_COLORS[:2])


def ballyhoo(heads: list[MovingHead]):
    """Bars 12-14: crossing figure-eights, a rainbow running across the
    rig, and a chase lighting one head after another on sixteenths."""
    count = len(heads)
    for i, head in enumerate(heads):

        def figure(u: float, i: int = i):
            return on_floor(6.0 * math.sin(9.6 * u + i), 0.5 + 4.0 * math.sin(19.2 * u + 2 * i))

        yield Aim(head, figure, blend=0.1)
        yield Rainbow(head.light, turns=1.5, offset=i / count)
    yield Chase([head.light for head in heads], steps=48, level=FULL, rest=0.3 * FULL)


def hit(heads: list[MovingHead]):
    """Bar 15: everything on the sculpture, full white, fading out into
    the blackout at the end of the bar."""
    for head in heads:
        yield Aim(head, still(SCULPTURE), blend=0.15)
        yield Fade(head.light.intensity, 0.0, start=1.3 * FULL)
        yield Recolor(head.light, [WHITE])


def clear_the_floor(stage: Stage):
    """Bar 15: the sculpture sinks into the floor and the mirror ball
    rises into the rig, so ``Finale`` finds the floor empty after the
    blackout."""
    yield stage.sculpture.animate.shift(IN * 4.0)
    yield stage.ball.animate.shift(OUT * 6.0)
