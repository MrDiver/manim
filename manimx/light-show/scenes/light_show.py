"""LightShow: a camera flight through a club, with moving heads on the beat.

32 seconds, 16 bars at 120 BPM, in five sections, each one ``play`` call
holding an ``Aim`` and a few ``Drive`` animations per moving head:

====  ========  ==========================================================
bars  section   the lights
====  ========  ==========================================================
0-3   intro     the heads open one per beat, deep blue, slow circles
4-7   build     cyan and magenta circles around the sculpture, pulsing
                with the kick; a strobe speeds up into the drop
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

import numpy as np
from manimx.animation import Scene, smooth

from .music import BAR, BEAT, DROP, SHOW_LENGTH, kick_envelope, show_track
from .rig import (
    AMBER,
    CYAN,
    DEEP_BLUE,
    MAGENTA,
    RED,
    SCULPTURE,
    VIOLET,
    WHITE,
    Aim,
    Drive,
    MovingHead,
    Stage,
    build_stage,
    fly,
    hue,
    on_floor,
    rgb,
)

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


class LightShow(Scene):
    def construct(self) -> None:
        stage = build_stage(self.time)
        self.add(stage.root)
        self.camera.set_field_of_view(math.radians(48))
        self.camera.near, self.camera.far = 0.1, 160.0
        fly(self.camera, self.time, CAMERA_PATH)
        self.play_wave(show_track())

        heads = stage.heads
        front, back = stage.front, stage.back
        self.play(*intro(heads), run_time=4 * BAR)
        self.play(*build(heads), run_time=4 * BAR)
        self.play(*drop(front, back), run_time=4 * BAR)
        self.play(*ballyhoo(heads), run_time=3 * BAR)
        self.play(*hit(heads), *clear_the_floor(stage), run_time=BAR)


def color_of(head: MovingHead, value) -> Drive:
    return Drive(head.light.color, value)


def intensity_of(head: MovingHead, value) -> Drive:
    return Drive(head.light.intensity, value)


def circle_around(center, radius: float, speed: float, phase: float):
    return lambda t: center + radius * np.array([math.cos(speed * t + phase), math.sin(speed * t + phase), 0.0])


def intro(heads: list[MovingHead]):
    """Bars 0-3: one head opens per beat with a flash, then breathes."""
    for i, head in enumerate(heads):
        opens = (i + 1) * BEAT

        def level(t: float, opens: float = opens) -> float:
            if t < opens:
                return 0.0
            flash = math.exp(-(t - opens) / 0.25)
            return FULL * (0.35 + 0.65 * flash + 0.3 * kick_envelope(t))

        ring = circle_around(
            on_floor(*SCULPTURE[:2]),
            3.5 + 1.5 * (i % 2),
            0.35,
            i * math.tau / len(heads),
        )
        yield Aim(head, ring, blend=1.0)
        yield intensity_of(head, level)
        yield color_of(
            head,
            lambda t, i=i: rgb(DEEP_BLUE) if t < 2 * BAR else rgb(CYAN if i % 2 else MAGENTA),
        )


def build(heads: list[MovingHead]):
    """Bars 4-7: circles around the sculpture pulsing with the kick; the
    last bar strobes faster and faster, all white, aimed at the middle."""
    start = 4 * BAR
    for i, head in enumerate(heads):
        ring = circle_around(
            on_floor(*SCULPTURE[:2]),
            2.5 + 2.0 * (i % 3),
            0.9 * (1 if i % 2 else -1),
            i * 1.3,
        )

        def target(t: float, ring=ring):
            if t < 3 * BAR:
                return ring(t)
            mix = min(1.0, (t - 3 * BAR) / BAR)  # pull in to the sculpture
            return (1 - mix) * ring(t) + mix * SCULPTURE

        def level(t: float) -> float:
            if t < 3 * BAR:
                return FULL * (0.25 + 0.75 * kick_envelope(start + t, 0.2))
            # A strobe speeding up from 4 to 16 flashes per second: the
            # phase is the integral of that frequency.
            since = t - 3 * BAR
            phase = (4.0 * since + 6.0 * since**2 / BAR) % 1.0
            return FULL * (1.0 if phase < 0.35 else 0.0) * (0.6 + 0.4 * since / BAR)

        yield Aim(head, target, blend=0.6)
        yield intensity_of(head, level)
        yield color_of(
            head,
            lambda t, i=i: rgb(WHITE) if t >= 3 * BAR else rgb(CYAN if i % 2 else MAGENTA),
        )


DROP_COLORS = (RED, AMBER, VIOLET, CYAN)


def drop(front: list[MovingHead], back: list[MovingHead]):
    """Bars 8-11: the front sweeps fans across the floor, strobing on
    eighths; the back hits on the kicks; a new color every bar."""
    for i, head in enumerate(front):
        spread = (i - 2) / 2  # -1 .. 1 across the truss

        def sweep(t: float, spread: float = spread):
            swing = math.sin(math.pi * t / BAR * 2)
            return on_floor(
                9.0 * spread * swing + 2.0 * spread,
                -1.5 + 4.0 * math.cos(math.pi * t / BAR),
            )

        def level(t: float) -> float:
            eighth = (t % (BEAT / 2)) / (BEAT / 2)
            return FULL * (1.0 if eighth < 0.45 else 0.08)

        yield Aim(head, sweep, blend=0.25)
        yield intensity_of(head, level)
        yield color_of(head, lambda t: rgb(DROP_COLORS[int(t // BAR) % 4]))
    for i, head in enumerate(back):
        ring = circle_around(on_floor(*SCULPTURE[:2]), 2.0, 2.2 * (1 if i % 2 else -1), i * math.pi / 2)
        yield Aim(head, ring, blend=0.25)
        yield intensity_of(head, lambda t: FULL * (0.1 + 0.9 * kick_envelope(DROP + t, 0.12)))
        yield color_of(head, lambda t: rgb(DROP_COLORS[(int(t // BAR) + 2) % 4]))


def ballyhoo(heads: list[MovingHead]):
    """Bars 12-14: crossing figure-eights, a rainbow running across the
    rig, and a chase that lights one head after another on sixteenths."""
    count = len(heads)
    for i, head in enumerate(heads):

        def figure(t: float, i: int = i):
            return on_floor(6.0 * math.sin(1.6 * t + i), 0.5 + 4.0 * math.sin(3.2 * t + 2 * i))

        def level(t: float, i: int = i) -> float:
            lead = int(t / (BEAT / 4)) % count
            return FULL * (1.0 if lead == i else 0.3)

        yield Aim(head, figure, blend=0.5)
        yield intensity_of(head, level)
        yield color_of(head, lambda t, i=i: hue(i / count + 0.25 * t))


def clear_the_floor(stage: Stage):
    """Bar 15, from its second beat: the sculpture sinks into the floor and
    the mirror ball rises into the rig, so ``Finale`` finds the floor
    empty after the blackout."""

    def away(t: float) -> float:
        return smooth(min(1.0, max(0.0, (t - BEAT) / (2.5 * BEAT))))

    bottom = stage.sculpture.position.z.value
    yield Drive(stage.sculpture.position.z, lambda t: bottom - 4.0 * away(t))
    top = stage.ball.position.z.value
    yield Drive(stage.ball.position.z, lambda t: top + 6.0 * away(t))


def hit(heads: list[MovingHead]):
    """Bar 15: everything on the sculpture, full white, fading into a
    blackout on the last beat."""
    for head in heads:
        yield Aim(head, lambda t: SCULPTURE, blend=0.3)
        yield intensity_of(head, lambda t: FULL * 1.3 * max(0.0, 1.0 - t / (3.5 * BEAT)) ** 1.5)
        yield color_of(head, lambda t: rgb(WHITE))
