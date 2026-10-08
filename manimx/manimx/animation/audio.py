"""Audio: sounds placed in scenes, mixed per timeline.

A :class:`Sound` is plain sample data (float32, ``(samples, channels)``,
values in [-1, 1]) with its sample rate, loaded from a file or made from
an array. Scene code places sounds like events, at the cursor and
without advancing it (``scene.play_sound``, ``scene.play_wave``, or a
simulation's ``play_wave`` at the simulated time).

Sounds are never retimed: a sound starts at the global time its scene
point is shown, through the clip's time map, and then plays at its own
speed. Slowing a scene down does not lower its pitch; it moves the
starts of its sounds apart.
"""

from __future__ import annotations

import subprocess
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 48_000


@dataclass(frozen=True, eq=False)
class Sound:
    samples: np.ndarray
    """float32 ``(count, channels)``, values in [-1, 1]."""
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or samples.shape[1] not in (1, 2):
            raise ValueError(f"sound samples must have shape (count,) or (count, 1 or 2), got {samples.shape}")
        if self.sample_rate <= 0:
            raise ValueError("a sample rate must be positive")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @classmethod
    def load(cls, path: str | Path) -> Sound:
        """Read a sound file: WAV directly, anything else through ffmpeg."""
        path = Path(path)
        if path.suffix.lower() == ".wav":
            try:
                return _read_wav(path)
            except wave.Error:
                pass  # e.g. float WAV: let ffmpeg decode it
        return _decode_with_ffmpeg(path)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def as_stereo(self, sample_rate: int) -> np.ndarray:
        """Samples resampled (linearly) to ``sample_rate``, as two channels."""
        samples = self.samples
        if sample_rate != self.sample_rate and len(samples):
            count = max(1, round(len(samples) * sample_rate / self.sample_rate))
            source = np.arange(len(samples)) / self.sample_rate
            target = np.arange(count) / sample_rate
            samples = np.stack([np.interp(target, source, channel) for channel in samples.T], axis=1)
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        return samples.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class SoundEvent:
    """A sound placed in a segment."""

    sound: Sound
    time: float
    """Time within the segment."""
    volume: float = 1.0


def mix(
    placed: Iterable[tuple[float, float, Sound, float]], start: float, end: float, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Mix ``(time, skip, sound, volume)`` entries into stereo float32 samples
    for times ``[start, end)``; ``skip`` seconds of a sound are left out
    (it was cut). The result is not clipped."""
    count = max(0, round((end - start) * sample_rate))
    output = np.zeros((count, 2), dtype=np.float32)
    converted: dict[int, np.ndarray] = {}  # one sound often plays many times
    for time, skip, sound, volume in placed:
        samples = converted.get(id(sound))
        if samples is None:
            samples = converted[id(sound)] = sound.as_stereo(sample_rate)
        first = round((time - start) * sample_rate) - round(skip * sample_rate)
        lo, hi = max(first, 0), min(first + len(samples), count)
        if lo < hi:
            output[lo:hi] += volume * samples[lo - first : hi - first]
    return output


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Write samples as 16-bit PCM WAV (clipped to [-1, 1])."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(samples.shape[1])
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> Sound:
    with wave.open(str(path), "rb") as file:
        width, channels, rate = file.getsampwidth(), file.getnchannels(), file.getframerate()
        data = file.readframes(file.getnframes())
    if width == 1:
        samples = (np.frombuffer(data, np.uint8).astype(np.float32) - 128) / 128
    elif width == 3:
        raw = np.frombuffer(data, np.uint8).reshape(-1, 3)
        values = (raw[:, 0].astype(np.int32) | raw[:, 1].astype(np.int32) << 8 | raw[:, 2].astype(np.int32) << 16) << 8 >> 8
        samples = values.astype(np.float32) / 2**23
    else:
        dtype = {2: "<i2", 4: "<i4"}[width]
        samples = np.frombuffer(data, dtype).astype(np.float32) / 2 ** (8 * width - 1)
    return Sound(samples.reshape(-1, channels), rate)


def _decode_with_ffmpeg(path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Sound:
    if not path.exists():
        raise FileNotFoundError(path)
    command = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sample_rate), "-"]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {result.stderr.decode(errors='replace').strip()}")
    return Sound(np.frombuffer(result.stdout, "<f4").reshape(-1, 2), sample_rate)
