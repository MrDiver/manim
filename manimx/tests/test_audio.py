"""Sounds in scenes, simulations and timelines, and audio in rendered output."""

import shutil
import subprocess
import wave

import numpy as np
import pytest

from manimx.animation import Scene, Simulation, Sound, Timeline, mix, write_wav
from manimx.userland import Dot
from manimx.userland.rate_functions import linear

RATE = 1000  # a low sample rate keeps sample positions easy to check
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def prepared(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def click(length: int = 10) -> np.ndarray:
    return np.ones(length, dtype=np.float32) * 0.5


def onsets(samples: np.ndarray) -> list[float]:
    """Times (s) where sound starts after silence."""
    loud = np.abs(samples[:, 0]) > 1e-6
    starts = np.nonzero(loud & ~np.concatenate([[False], loud[:-1]]))[0]
    return [index / RATE for index in starts]


# --- Sound ------------------------------------------------------------------


def test_sounds_hold_mono_or_stereo_samples_and_resample() -> None:
    mono = Sound(np.linspace(-1, 1, 100), 100)
    assert mono.samples.shape == (100, 1) and mono.duration == 1.0 and not mono.samples.flags.writeable

    stereo = mono.as_stereo(200)
    assert stereo.shape == (200, 2) and stereo.dtype == np.float32
    np.testing.assert_allclose(stereo[:, 0], stereo[:, 1])
    assert stereo[100, 0] == pytest.approx(mono.samples[50, 0], abs=0.02)
    with pytest.raises(ValueError, match="shape"):
        Sound(np.zeros((10, 3)), 100)
    with pytest.raises(ValueError, match="positive"):
        Sound(np.zeros(10), 0)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_wav_files_load_at_every_sample_width(tmp_path, width) -> None:
    path = tmp_path / "tone.wav"
    values = np.array([0.0, 0.5, -0.5, 0.25])
    scale = 2 ** (8 * width - 1)
    if width == 1:
        data = (values * 128 + 128).astype(np.uint8).tobytes()
    elif width == 3:
        ints = (values * scale).astype("<i4")
        data = b"".join(int(v).to_bytes(3, "little", signed=True) for v in ints)
    else:
        data = (values * scale).astype({2: "<i2", 4: "<i4"}[width]).tobytes()
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(width)
        file.setframerate(8000)
        file.writeframes(data)

    sound = Sound.load(path)
    assert sound.sample_rate == 8000
    np.testing.assert_allclose(sound.samples[:, 0], values, atol=1 / 120)


def test_write_wav_round_trips_and_clips(tmp_path) -> None:
    path = tmp_path / "out.wav"
    write_wav(path, np.array([0.0, 0.5, 2.0, -3.0]), 4000)

    sound = Sound.load(path)
    np.testing.assert_allclose(sound.samples[:, 0], [0.0, 0.5, 1.0, -1.0], atol=1e-4)


@needs_ffmpeg
def test_other_formats_decode_through_ffmpeg(tmp_path) -> None:
    source = tmp_path / "tone.wav"
    write_wav(source, np.sin(np.linspace(0, 200, 8000)) * 0.5, 8000)
    flac = tmp_path / "tone.flac"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(source), str(flac)], check=True)

    sound = Sound.load(flac)
    assert sound.samples.shape[1] == 2 and sound.duration == pytest.approx(1.0, abs=0.01)
    with pytest.raises(FileNotFoundError):
        Sound.load(tmp_path / "missing.mp3")
    (tmp_path / "broken.mp3").write_bytes(b"not audio")
    with pytest.raises(RuntimeError, match="could not decode"):
        Sound.load(tmp_path / "broken.mp3")


def test_mix_places_skips_and_scales() -> None:
    sound = Sound(np.arange(1, 11, dtype=np.float32) / 10, RATE)
    out = mix([(0.005, 0.0, sound, 1.0), (0.0, 0.004, sound, 2.0), (-1.0, 0.0, sound, 1.0)], 0.0, 0.02, RATE)

    assert out.shape == (20, 2)
    assert out[0, 0] == pytest.approx(1.0)  # the second entry starts 4 samples in: 0.5 * 2
    assert out[5, 0] == pytest.approx(0.1 + 2 * 1.0)
    assert out[15, 0] == 0.0


# --- scenes and simulations ------------------------------------------------------


def test_scenes_play_sounds_without_moving_the_cursor(tmp_path) -> None:
    path = tmp_path / "click.wav"
    write_wav(path, click(), RATE)

    def construct(scene):
        scene.add(Dot())
        scene.wait(0.1)
        sound = scene.play_sound(path, volume=0.5)
        assert sound.duration == pytest.approx(0.01)
        scene.play_wave(click(), RATE)
        scene.play_wave(path, volume=0.0)  # files work too; silent here
        scene.play_wave(np.stack([np.zeros(10), click()], axis=1), RATE, volume=2.0)  # stereo: right only
        scene.wait(0.2)

    scene = prepared(construct)
    assert scene.duration == pytest.approx(0.3)
    stereo = scene.audio(sample_rate=RATE)
    assert stereo[100, 1] - stereo[100, 0] == pytest.approx(1.0, abs=1e-3)
    audio = scene.audio(sample_rate=RATE)
    assert audio.shape == (300, 2) and onsets(audio) == [0.1]
    assert audio[100, 0] == pytest.approx(0.25 + 0.5, abs=1e-3)


def test_simulations_play_sounds_at_simulated_times() -> None:
    ball = Dot(name="ball")

    def step(dt):
        ball.shift((dt, 0, 0))
        if abs(simulation.time - 0.25) < 1e-9:
            simulation.play_wave(click(), RATE)

    simulation = Simulation(step, dt=0.05, duration=0.5)

    def construct(scene):
        scene.add(ball)
        scene.wait(0.5)
        scene.play(simulation)

    assert onsets(prepared(construct).audio(sample_rate=RATE)) == [0.75]


# --- timelines ---------------------------------------------------------------------


def two_clicks(scene):
    """A click at 0.1 s, a hold for "next", a click 0.2 s into the second part."""
    scene.add(Dot())
    scene.wait(0.1)
    scene.play_wave(click(), RATE)
    scene.wait(0.4)
    scene.wait_for("next")
    scene.wait(0.2)
    scene.play_wave(click(), RATE)
    scene.wait(0.3)


def test_timeline_sounds_follow_clip_timing_without_changing_pitch() -> None:
    timeline = Timeline(events={"next": 2.0})
    clip = timeline.add(prepared(two_clicks), start=1.0)
    assert onsets(timeline.audio(sample_rate=RATE)) == [1.1, 2.2]

    clip.slow_motion(0.0, 0.5, factor=2.0)  # the first part plays twice as slow
    audio = timeline.audio(sample_rate=RATE)
    assert onsets(audio) == pytest.approx([1.2, 2.2])
    assert np.count_nonzero(audio[1200:1300, 0]) == 10  # same length: no pitch change


def test_cuts_skip_sounds_or_enter_them_midway() -> None:
    timeline = Timeline(events={"next": 0.0})
    timeline.add(prepared(two_clicks)).cut(0.105, 0.65)  # 5 ms into the first click, before the second

    audio = timeline.audio(sample_rate=RATE)
    assert np.count_nonzero(audio[:, 0]) == 5  # the rest of the first click only

    late = Timeline()
    late.add(prepared(two_clicks)).cut(start=0.6)  # the first segment is skipped entirely
    assert onsets(late.audio(sample_rate=RATE)) == [0.1]
