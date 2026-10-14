"""Sound for editing elsewhere: stems and an OpenTimelineIO timeline of them."""

import json

import numpy as np
import pytest
from manimx.animation import AudioClip, Sound, Timeline, write_wav
from manimx.animation.timing import Track
from manimx.drivers.session import export_session
from manimx.voice import Script

RATE = 1000


def test_stems_have_the_track_volume_and_every_track(tmp_path) -> None:
    tone = Sound(np.full(2 * RATE, 0.5, dtype=np.float32), RATE)
    timeline = Timeline(
        audio_clips=[AudioClip(tone, start=0.0, track=0), AudioClip(tone, start=1.0, track=1, volume=0.5), AudioClip(tone, script=Script.new("x", "hello"), track=0)],
        audio_tracks={1: Track(name="Music/FX", volume=0.25, muted=True)},
        script_tracks={0: Track(name="Audio 1")},
    )
    assert timeline.track_name("audio", 0) == "Audio 1" and timeline.track_name("audio", 1) == "Music/FX"
    stems = timeline.stems(sample_rate=RATE)
    assert list(stems) == [("scripts", 0), ("audio", 0), ("audio", 1)]  # as the preview shows them
    assert stems[("audio", 1)][1500, 0] == pytest.approx(0.0625)  # the clip's and the track's volume, though muted
    assert timeline.audio(sample_rate=RATE)[1500, 0] == pytest.approx(1.0)  # two tones; the muted track is not heard

    video = tmp_path / "renders" / "demo.mp4"
    video.parent.mkdir()
    (video.parent / "demo-stems").mkdir()
    (video.parent / "demo-stems" / "Gone.wav").write_bytes(b"")  # a track deleted since the last export
    session = export_session(timeline, video, sample_rate=RATE, fps=10)
    assert [stem.file.name for stem in session.stems] == ["Audio 1.wav", "Audio 1 2.wav", "Music_FX (muted).wav"]
    assert sorted(path.name for path in (video.parent / "demo-stems").iterdir()) == ["Audio 1 2.wav", "Audio 1.wav", "Music_FX (muted).wav"]
    assert [stem.muted for stem in session.stems] == [False, False, True]
    loaded = Sound.load(session.stems[2].file)
    assert loaded.sample_rate == RATE and len(loaded.samples) == 3 * RATE
    assert loaded.samples[1500, 0] == pytest.approx(0.0625, abs=1e-6)  # 24 bits

    otio = json.loads(session.otio.read_text())
    tracks = otio["tracks"]["children"]
    assert [(track["name"], track["kind"]) for track in tracks] == [("Video", "Video"), ("Audio 1", "Audio"), ("Audio 1", "Audio"), ("Music/FX (muted)", "Audio")]
    clip = tracks[3]["children"][0]
    assert clip["media_reference"]["target_url"] == str((video.parent / "demo-stems" / "Music_FX (muted).wav").resolve())
    assert clip["source_range"]["duration"] == {"OTIO_SCHEMA": "RationalTime.1", "rate": 10, "value": 30.0}


def test_wav_files_have_16_or_24_bits(tmp_path) -> None:
    samples = np.linspace(-1, 1, 101, dtype=np.float32)
    write_wav(tmp_path / "deep.wav", samples, RATE, bits=24)
    assert np.abs(Sound.load(tmp_path / "deep.wav").samples[:, 0] - samples).max() < 1e-6
    with pytest.raises(ValueError, match="16 or 24 bits"):
        write_wav(tmp_path / "odd.wav", samples, RATE, bits=8)
