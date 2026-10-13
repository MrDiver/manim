"""Voice: transcripts, their words as events, alignment, subtitles, and
scenes synced to words (manimx/voice/, timing.py, project.py).
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from manimx.animation import Scene, Sound, Timeline
from manimx.animation.timing import AudioClip
from manimx.drivers import FrameSource, ImageSequence, RenderSettings, VideoFile, render
from manimx.examples.navigation import config
from manimx.project import ClipEntry, Project
from manimx.voice import (
    Subtitle,
    Transcript,
    Word,
    align,
    srt,
    transcript_path,
    trim_silence,
)
from manimx.voice.transcript import tokens

TEXT = "Every circle, every square.\n[middle]\nTime has run out for the square\nand now the circle."


def timed(text: str, start: float = 1.0, step: float = 0.5) -> Transcript:
    """``text`` with a word every ``step`` seconds from ``start``, each 0.4 s long."""
    return Transcript(text, tuple(Word(token.text, start + index * step, start + index * step + 0.4) for index, token in enumerate(tokens(text))))


def test_words_have_keys_sections_and_lines() -> None:
    words = tokens("Let's go!\n[part-2]\nVoice. — 3")
    assert [(word.text, word.key, word.section, word.line) for word in words] == [
        ("Let's", "lets", None, 0),
        ("go!", "go", None, 0),
        ("Voice.", "voice", "part-2", 2),
        ("3", "3", "part-2", 2),
    ]  # a dash is no word, a marker is no word


def test_every_word_is_an_event() -> None:
    transcript = timed(TEXT)
    events = transcript.events("voice")
    assert events["voice.every"] == 1.0 and events["voice.every.2"] == 2.0
    assert events["voice.square"] == 2.5 and events["voice.square.2"] == 6.0  # said again
    assert events["voice.middle.square"] == 6.0  # the first after [middle]
    assert events["voice.[middle]"] == 3.0 and events["voice.middle.time"] == 3.0
    assert events["voice.circle.2"] == events["voice.middle.circle"] == 8.0
    assert transcript.event_names("voice")[:4] == ["voice.every", "voice.circle", "voice.every.2", "voice.square"]


def test_subtitles_are_lines_split_at_pauses() -> None:
    transcript = timed(TEXT)
    assert transcript.subtitles() == [
        Subtitle(1.0, 2.9, "Every circle, every square."),
        Subtitle(3.0, 6.4, "Time has run out for the square"),
        Subtitle(6.5, 8.4, "and now the circle."),
    ]
    long = Transcript("one two three four", (Word("one", 0, 0.4), Word("two", 0.5, 0.9), Word("three", 2.0, 2.4), Word("four", 2.5, 2.9)))
    assert [item.text for item in long.subtitles(characters=10)] == ["one two", "three four"]  # at the long pause
    assert srt(long.subtitles(characters=10)[:1]) == "1\n00:00:00,000 --> 00:00:00,900\none two\n\n"


def test_the_text_takes_the_times_of_the_words_it_matches() -> None:
    heard = [Word("If", 1.0, 1.2), Word("you", 1.2, 1.4), Word("flash", 1.5, 2.0), Word("3", 3.0, 3.5), Word("dots", 4.0, 4.5), Word("um", 5.0, 5.2)]
    transcript = align("Every flash, three new dots", heard)
    starts = [(word.text, round(word.start, 3), round(word.end, 3)) for word in transcript.words]
    assert starts == [
        ("Every", 1.0, 1.4),  # replaces "If you": their span
        ("flash,", 1.5, 2.0),  # matches
        ("three", 3.0, 3.312),  # "three new" replaces "3": its span, by letters
        ("new", 3.312, 3.5),
        ("dots", 4.0, 4.5),
    ]  # "um" is not in the text: left out


def test_a_transcript_file_edited_by_hand_keeps_its_times(tmp_path) -> None:
    path = transcript_path(tmp_path / "voice.m4a")
    assert path.name == "voice.transcript.json"
    timed(TEXT).save(path)
    assert Transcript.load(path).words == timed(TEXT).words
    data = json.loads(path.read_text())
    data["text"] = data["text"].replace("Time has run out", "Time ran out")
    path.write_text(json.dumps(data))
    edited = Transcript.load(path)
    assert [word.text for word in edited.words][4:7] == ["Time", "ran", "out"]
    assert edited.events("voice")["voice.square.2"] == timed(TEXT).events("voice")["voice.square.2"]
    path.write_text("[]")
    with pytest.raises(ValueError, match="is not a transcript"):
        Transcript.load(path)
    with pytest.raises(ValueError, match="2 words but 1 have times"):
        Transcript("two words", (Word("two", 0, 1),))


def test_word_spans_lose_the_silence_around_them() -> None:
    rate = 16_000
    samples = np.zeros(3 * rate, dtype=np.float32)
    samples[rate : 2 * rate] = 0.5 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)  # a word from 1 s to 2 s
    samples += np.random.default_rng(1).normal(0, 1e-3, samples.shape).astype(np.float32)  # room noise
    [word] = trim_silence([Word("word", 0.5, 2.8)], Sound(samples, rate))
    assert word.start == pytest.approx(1.0, abs=0.04) and word.end == pytest.approx(2.0, abs=0.04)


class Synced(Scene):
    """Moves on 1 s of scene time; its cue at 0.5 s is synced to a word."""

    def construct(self) -> None:
        self.wait(0.5)
        self.cue("voice.square")
        self.wait(0.5)


def test_scenes_follow_the_words_of_audio_clips() -> None:
    sound = Sound(np.zeros(10 * 8000, dtype=np.float32), 8000)
    voice = AudioClip(sound, start=2.0, name="voice", transcript=timed(TEXT))
    timeline = Timeline(audio_clips=[voice])
    clip = timeline.add(Synced().prepare())
    assert timeline.event_times()["voice.square"] == 4.5  # 2 s + 2.5 s into the sound
    assert clip.compiled_time(4.5) == pytest.approx(0.5)  # the cue lands on the word
    voice.start = 3.0
    timeline.audio_changed()
    assert clip.compiled_time(5.5) == pytest.approx(0.5)  # moved with the audio clip
    voice.cut_in = 3.0  # "square" is cut away: the cue is free again
    timeline.audio_changed()
    assert "voice.square" not in timeline.event_times() and "voice.square.2" in timeline.event_times()
    clip.bind_cues({"voice.square": "voice.middle.square"})
    assert clip.compiled_time(3.0 + 6.0 - 3.0) == pytest.approx(0.5)
    assert timeline.subtitles()[0] == Subtitle(3.0, 6.4, "Time has run out for the square")
    timeline.fire("voice.middle.square", 4.0)  # a time set by hand wins
    assert clip.compiled_time(4.0) == pytest.approx(0.5)


def test_projects_read_transcripts_and_cue_bindings(tmp_path) -> None:
    root = tmp_path / "project"
    (root / "scenes").mkdir(parents=True)
    (root / "scenes" / "__init__.py").write_text("")  # a package, like every project's (not a namespace one)
    (root / "scenes" / "synced.py").write_text(
        "from manimx import Scene\n\n\nclass Synced(Scene):\n    def construct(self):\n        self.wait(0.5)\n        self.cue('square')\n        self.wait(0.5)\n"
    )
    (root / "assets").mkdir()
    from manimx.animation import write_wav

    write_wav(root / "assets" / "voice.wav", np.zeros(8 * 8000, dtype=np.float32), 8000)
    timed(TEXT).save(root / "assets" / "voice.transcript.json")
    data = {
        "format": 3,
        "name": "voiced",
        "scenes": {"Synced": {"module": "scenes.synced"}},
        "timeline": {"clips": [{"scene": "Synced", "start": 0.0, "cues": {"square": "voice.middle.square"}}], "audio": [{"file": "assets/voice.wav", "start": 1.0}]},
    }
    (root / "manimx.json").write_text(json.dumps(data))
    project = Project.load(root)
    assert project.timeline[0].cues == {"square": "voice.middle.square"}
    timeline = project.build_timeline()
    assert timeline.clips[0].compiled_time(7.0) == pytest.approx(0.5)  # 1 s + the second "square" at 6 s
    assert project.to_json()["timeline"]["clips"][0]["cues"] == {"square": "voice.middle.square"}
    project.save_transcript("assets/voice.wav", None)
    assert project.transcript("assets/voice.wav") is None and not (root / "assets" / "voice.transcript.json").exists()
    data["timeline"]["clips"][0]["cues"] = {"square": 3}
    (root / "manimx.json").write_text(json.dumps(data))
    with pytest.raises(Exception, match="cues map cue names to event names"):
        Project.load(root)
    assert ClipEntry("Synced").cues == {}


def test_videos_carry_the_subtitles_and_an_srt_file(tmp_path) -> None:
    lines = [Subtitle(0.2, 0.6, "first"), Subtitle(0.9, 2.0, "second")]
    source = FrameSource(Synced().prepare().frame_at, 1.0, subtitles=lambda: lines)
    settings = RenderSettings(config(64, 36), fps=10, start=0.5, end=1.0)
    render(source, VideoFile(tmp_path / "video.mp4"), settings)
    render(source, ImageSequence(tmp_path / "frames"), settings)
    shifted = "1\n00:00:00,000 --> 00:00:00,100\nfirst\n\n2\n00:00:00,400 --> 00:00:00,600\nsecond\n\n"
    assert (tmp_path / "video.srt").read_text() == shifted  # only the rendered range, from its start
    assert (tmp_path / "frames" / "subtitles.srt").read_text() == shifted
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(tmp_path / "video.mp4")],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert result.stdout.split() == ["video", "subtitle"]


def test_the_example_voice_scene_follows_its_recording() -> None:
    root = Path(__file__).parent / "projects" / "example-project"
    spec = importlib.util.spec_from_file_location("example_voice", root / "scenes" / "voice.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Voice = module.Voice

    transcript = Transcript.load(root / "assets" / "audio" / "voice.transcript.json")
    assert transcript.text.startswith("Everything in this animation is timed by my voice.")
    prepared = Voice().prepare()
    assert len(prepared.segments[0].cues) == len(transcript.words) == 46
    # Spoken 10 % slower and a second later: every word still lands on its cue.
    slower = Transcript(transcript.text, tuple(Word(word.text, 1.1 * word.start, 1.1 * word.end) for word in transcript.words))
    sound = Sound.load(root / "assets" / "audio" / "voice.m4a")
    timeline = Timeline(audio_clips=[AudioClip(sound, start=1.0, name="voice", transcript=slower)])
    clip = timeline.add(prepared)
    for word in (transcript.words[10], transcript.words[30]):
        assert clip.compiled_time(1.0 + 1.1 * word.start) == pytest.approx(word.start, abs=1e-6)


def test_transcribe_writes_down_what_whisper_hears(monkeypatch) -> None:
    from manimx.voice import alignment

    rate = 8000
    samples = np.zeros(3 * rate, dtype=np.float32)
    samples[rate // 2 : rate] = 0.5  # "Hello." from 0.5 s, then "Bye" from 1.5 s
    samples[3 * rate // 2 : 2 * rate] = 0.5
    samples[rate // 4 :] += np.random.default_rng(2).normal(0, 1e-3, 11 * rate // 4).astype(np.float32)  # room noise after digital silence
    heard = [Word("Hello.", 0.3, 1.4), Word("Bye", 1.5, 2.5)]
    monkeypatch.setattr(alignment, "recognize", lambda sound, model, language: heard)
    transcript = alignment.transcribe(Sound(samples, rate))
    assert transcript.text == "Hello.\nBye"  # a line per sentence
    spans = [(word.start, word.end) for word in transcript.words]
    assert np.allclose(spans, [(0.5, 1.0), (1.5, 2.0)], atol=0.03)  # the loud parts (50 ms smoothing)
    assert alignment.transcribe(Sound(samples, rate), "Hello. Goodbye").words[1].start == pytest.approx(1.5)
    assert alignment._mono(Sound(np.ones((48_000, 2), dtype=np.float32), 48_000), 16_000).shape == (16_000,)


def test_transcribing_needs_faster_whisper(monkeypatch) -> None:
    import builtins

    from manimx.voice import recognize

    real = builtins.__import__

    def without_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_whisper)
    with pytest.raises(RuntimeError, match="pip install 'manimx\\[voice\\]'"):
        recognize(Sound(np.zeros(800, dtype=np.float32), 8000))
