"""The soundtrack, synthesized: a 120 BPM track for the show and the finale.

Nothing is loaded from disk. Every instrument is a few lines of numpy (a
sine sweep for the kick, filtered noise for hats and claps, detuned saws
for the pad), and the arrangement is a list of bars. The same beat grid
drives the lights: :func:`kick_envelope` is what the strobes, the floor
grid and the speaker cones follow, so picture and sound cannot drift
apart.
"""

from __future__ import annotations

import math

import numpy as np
from manimx.animation.audio import DEFAULT_SAMPLE_RATE

RATE = DEFAULT_SAMPLE_RATE
BPM = 120.0
BEAT = 60.0 / BPM
"""Seconds per beat."""
BAR = 4 * BEAT

SHOW_BARS = 16
SHOW_LENGTH = SHOW_BARS * BAR
KICK_BARS = range(2, 15)
"""Bars of the show with a kick on every beat (the intro and the last bar have none)."""
DROP = 8 * BAR
"""Where the drop lands."""

# A minor: Am, F, C, G, one chord per bar (MIDI note numbers).
CHORDS = ((57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62))
BASS = (33, 29, 36, 31)


def kick_times() -> list[float]:
    return [bar * BAR + beat * BEAT for bar in KICK_BARS for beat in range(4)]


def kick_envelope(t: float, decay: float = 0.16) -> float:
    """1 on every kick of the show, decaying exponentially until the next."""
    bar = int(t // BAR)
    if bar not in KICK_BARS:
        return 0.0
    return math.exp(-(t % BEAT) / decay)


# -- instruments -----------------------------------------------------------------


def _time(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * RATE)) / RATE


def _hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12)


def kick() -> np.ndarray:
    t = _time(0.45)
    frequency = 45 + 110 * np.exp(-t / 0.04)
    body = np.sin(2 * np.pi * np.cumsum(frequency) / RATE) * np.exp(-t / 0.22)
    click = np.random.default_rng(1).standard_normal(len(t)) * np.exp(-t / 0.003) * 0.3
    return np.tanh(1.6 * (body + click)) * 0.9


def hat(length: float = 0.05, seed: int = 2) -> np.ndarray:
    t = _time(length * 4)
    noise = np.diff(np.random.default_rng(seed).standard_normal(len(t) + 1))  # a first-order high-pass
    return noise * np.exp(-t / length) * 0.12


def clap() -> np.ndarray:
    t = _time(0.3)
    noise = np.random.default_rng(3).standard_normal(len(t))
    band = noise - np.convolve(noise, np.ones(12) / 12, mode="same")  # take off the lows
    bursts = sum(np.exp(-np.maximum(t - d, 0) / 0.01) * (t >= d) for d in (0.0, 0.012, 0.024))
    return band * (0.6 * bursts + np.exp(-t / 0.12)) * 0.25


def saw(frequency: float, t: np.ndarray, harmonics: int = 10) -> np.ndarray:
    """A band-limited saw (the first ``harmonics`` partials): fewer partials sound darker."""
    return sum(np.sin(2 * np.pi * k * frequency * t) / k for k in range(1, harmonics + 1)) * (2 / np.pi)


def pad(chord: tuple[int, ...], seconds: float, brightness: int = 6) -> np.ndarray:
    """A stereo chord of detuned saws with a slow swell."""
    t = _time(seconds)
    sides = []
    for detune in (-0.08, 0.08):
        voices = sum(saw(_hz(note + detune), t, brightness) + saw(_hz(note - 12 + detune), t, 3) for note in chord)
        sides.append(voices)
    envelope = np.minimum(t / 0.6, 1.0) * np.minimum((seconds - t) / 0.4, 1.0)
    return np.stack(sides, axis=1) * envelope[:, None] * 0.05


def bass(note: int, seconds: float) -> np.ndarray:
    t = _time(seconds)
    tone = np.tanh(2.5 * np.sin(2 * np.pi * _hz(note) * t)) + 0.4 * np.sin(np.pi * _hz(note) * t)
    return tone * np.exp(-t / 0.18) * np.minimum(t / 0.004, 1.0) * 0.22


def stab(chord: tuple[int, ...]) -> np.ndarray:
    t = _time(0.35)
    return sum(saw(_hz(note + 12), t, 14) for note in chord) * np.exp(-t / 0.09) * 0.05


def riser(seconds: float) -> np.ndarray:
    """Noise and a sine sweeping up, growing louder."""
    t = _time(seconds)
    noise = np.diff(np.random.default_rng(4).standard_normal(len(t) + 1))
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 1800 * (t / seconds) ** 2) / RATE)
    return (0.08 * noise + 0.1 * sweep) * (t / seconds) ** 2


def boom(seed: int = 0) -> np.ndarray:
    """A firework burst: a low thump with a crackling tail, stereo."""
    rng = np.random.default_rng(seed)
    t = _time(2.5)
    thump = np.sin(2 * np.pi * np.cumsum(30 + 80 * np.exp(-t / 0.05)) / RATE) * np.exp(-t / 0.4)
    sides = []
    for _ in range(2):
        crackle = (rng.random(len(t)) < 0.0012 * np.exp(-t / 0.9)) * rng.standard_normal(len(t))
        crackle = np.convolve(crackle, np.exp(-np.arange(80) / 12), mode="same")
        sides.append(0.8 * thump + 0.35 * crackle)
    return np.stack(sides, axis=1) * 0.6


def whoosh(seed: int = 0) -> np.ndarray:
    """A rocket going up."""
    t = _time(1.1)
    noise = np.diff(np.random.default_rng(10 + seed).standard_normal(len(t) + 1))
    whistle = np.sin(2 * np.pi * np.cumsum(900 + 1400 * t) / RATE) * 0.15
    return (0.06 * noise + whistle * 0.3) * np.sin(np.pi * t / t[-1]) * 0.6


# -- mixing ----------------------------------------------------------------------


class Mix:
    """A stereo buffer to place sounds into."""

    def __init__(self, seconds: float) -> None:
        self.samples = np.zeros((int(seconds * RATE), 2))

    def add(self, sound: np.ndarray, at: float, pan: float = 0.0, volume: float = 1.0) -> None:
        """Mono sounds are panned (-1 left .. 1 right) with constant power."""
        if sound.ndim == 1:
            angle = (pan + 1) * np.pi / 4
            sound = np.sqrt(2) * np.outer(sound, [np.cos(angle), np.sin(angle)])
        start = int(at * RATE)
        end = min(start + len(sound), len(self.samples))
        if end > start:
            self.samples[start:end] += volume * sound[: end - start]

    def master(self) -> np.ndarray:
        peak = np.max(np.abs(self.samples)) or 1.0
        return np.tanh(self.samples / peak * 1.2).astype(np.float32) * 0.85


def show_track() -> np.ndarray:
    """The 16 bars of the show: an intro with the pad, the beat from bar 2, a
    riser into the drop at bar 8, and a last bar that rings out into the
    blackout."""
    mix = Mix(SHOW_LENGTH + 1.0)
    the_kick, the_clap, the_hats = kick(), clap(), (hat(0.04, 2), hat(0.12, 5))
    for bar in range(SHOW_BARS):
        start = bar * BAR
        chord = CHORDS[bar % 4]
        if bar < 15:
            mix.add(
                pad(chord, BAR + 0.3, 4 if bar < 8 else 8),
                start,
                volume=1.0 if bar >= 2 else 3.0 + 1.0 * bar,
            )
        for beat in range(4):
            at = start + beat * BEAT
            if bar in KICK_BARS:
                mix.add(the_kick, at)
            if 4 <= bar < 15 and bar != 7:
                mix.add(the_hats[1], at + BEAT / 2, pan=0.3)
            if 8 <= bar < 15:
                if beat in (1, 3):
                    mix.add(the_clap, at, pan=-0.1)
                for sixteenth in (1, 3):
                    mix.add(the_hats[0], at + sixteenth * BEAT / 4, pan=-0.35)
            if 4 <= bar < 15:  # offbeat bass, eighths with an octave jump after the drop
                for offset in (0.5,) if bar < 8 else (0.0, 0.5):
                    mix.add(
                        bass(BASS[bar % 4] + (12 if offset and bar >= 8 else 0), BEAT / 2),
                        at + offset * BEAT,
                    )
        if 8 <= bar < 15:
            for at in (0.0, 1.5, 3.0):
                mix.add(stab(chord), start + at * BEAT, pan=0.4 if at else -0.4)
    mix.add(riser(2 * BAR), 6 * BAR, volume=1.4)
    for i in range(16):  # a snare roll into the drop, getting denser
        mix.add(the_clap, 7 * BAR + BAR * (1 - 0.5 ** (i / 4)), volume=0.3 + 0.04 * i)
    mix.add(pad(CHORDS[0], BAR + 1.0, 10), 15 * BAR, volume=1.3)
    mix.add(the_kick, 15 * BAR, volume=1.2)
    return mix.master()


FINALE_LENGTH = 13.0
FINALE_BEATS = (0.5, 1.0, 1.5, 2.0)
"""When the parts of the logo arrive (circle, square, triangle, M)."""
FINALE_X = 3.0
"""When the X slams in."""


def finale_track() -> np.ndarray:
    """The finale's music: four hits as the logo arrives, the X on a big
    kick, then a long Am pad under the fireworks (whose bangs the
    simulation plays itself, at the simulated times)."""
    mix = Mix(FINALE_LENGTH + 1.0)
    the_kick = kick()
    for at in FINALE_BEATS:
        mix.add(the_kick, at, volume=0.8)
        mix.add(stab(CHORDS[0]), at, volume=0.7)
    mix.add(riser(1.0), FINALE_X - 1.0, volume=0.8)
    mix.add(the_kick, FINALE_X, volume=1.2)
    mix.add(pad(CHORDS[0], FINALE_LENGTH - FINALE_X, 8), FINALE_X, volume=1.4)
    return mix.master()
