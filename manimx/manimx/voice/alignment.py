"""Alignment: the times of a transcript's words, from the recording.

    words = recognize(Sound.load("voice.m4a"))        # Whisper's words, timed
    transcript = align(text, words)                   # the author's text, timed
    transcript = transcribe(sound)                    # both; Whisper's text as the text

Whisper (faster-whisper, https://github.com/SYSTRAN/faster-whisper, an
optional dependency: ``uv add 'manimx[voice]'``) recognizes the
recording's words with their times (``word_timestamps``, from the
attention of its decoder on the audio, see
https://github.com/openai/whisper/blob/main/whisper/timing.py). Whisper
does not always hear what the author wrote ("3" for "three", a word
missed, one heard twice), so :func:`align` matches the author's text to
the recognized words and takes the times from there:

* words that match (by :func:`~manimx.voice.transcript.word_key`) take
  the recognized word's times;
* a run of words that replaces a run of recognized ones shares their
  time span, by letters;
* words with no recognized counterpart share the gap they are in, by
  letters (or sit at its start if there is none);
* recognized words the text does not have are left out.

The matching is a longest common subsequence of the two word lists
(difflib's ``SequenceMatcher``, Ratcliff and Obershelp,
https://docs.python.org/3/library/difflib.html). Since it only needs
timed words, the same :func:`align` re-times an edited text from the
words of the transcript before the edit: words the edit kept keep their
times (also those moved by hand), new words fit in between.

Whisper lets a word run on into the pauses around it ("and" for a second
of silence, or starting in the silence before it). :func:`trim_silence`
moves every start to the first and every end to the last moment of the
word that is louder than the recording's quiet parts.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence

import numpy as np

from ..animation.audio import Sound
from .transcript import Transcript, Word, tokens, word_key

INSTALL = "uv add 'manimx[voice]' in the project (or pip install 'manimx[voice]')"
"""How to install the optional dependency transcribing needs."""

WHISPER_RATE = 16_000
"""The sample rate Whisper listens at."""

DEFAULT_MODEL = "small"
"""Whisper's model when none is given (a project's is its ``voice.model``
setting): downloaded once, ~470 MB; multilingual, Whisper finds the
language itself."""

UNSURE = 0.1
"""Recognized words Whisper is less sure of than this are left out (at
the end of a recording it sometimes hears a last word again)."""


def recognize(sound: Sound, model: str = DEFAULT_MODEL, language: str | None = None) -> list[Word]:
    """The words Whisper hears in ``sound``, with their times."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(f"transcribing needs faster-whisper: {INSTALL}") from None
    samples = _mono(sound, WHISPER_RATE)
    whisper = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(samples, word_timestamps=True, language=language, vad_filter=False)
    return [
        Word(word.word.strip(), float(word.start), float(word.end))
        for segment in segments
        for word in segment.words or ()
        if word.probability >= UNSURE and word_key(word.word)
    ]


def transcribe(sound: Sound, text: str | None = None, model: str = DEFAULT_MODEL, language: str | None = None) -> Transcript:
    """``sound``'s transcript: ``text`` (default: what Whisper hears) with
    the times Whisper hears its words at, ends trimmed."""
    heard = trim_silence(recognize(sound, model, language), sound)
    if text is None:
        text = _lines(heard)
    return align(text, heard)


def time_take(sound: Sound, text: str, offset: float = 0.0, model: str = DEFAULT_MODEL, language: str | None = None) -> tuple[Word, ...]:
    """The words of ``text`` timed from a take of it: the sound recorded
    for a script part, which starts ``offset`` seconds in (after the
    countdown). Times count from that start; what is said before it
    (counting along) is left out."""
    heard = [word for word in trim_silence(recognize(sound, model, language), sound) if word.end > offset]
    if not heard:
        raise ValueError("Whisper heard no words in the take")
    shifted = [Word(word.text, max(0.0, word.start - offset), word.end - offset) for word in heard]
    return align(text, shifted).words


def align(text: str, timed: Sequence[Word]) -> Transcript:
    """``text`` with the times of the words of ``timed`` it matches (see the module docs)."""
    wanted = tokens(text)
    keys = [token.key for token in wanted]
    times: list[tuple[float, float] | None] = [None] * len(wanted)
    matcher = difflib.SequenceMatcher(None, keys, [word_key(word.text) for word in timed], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                times[i1 + offset] = (timed[j1 + offset].start, timed[j1 + offset].end)
        elif tag == "replace":
            _share(times, keys, i1, i2, timed[j1].start, timed[j2 - 1].end)
    # Words with no counterpart share the gap between their neighbours.
    index = 0
    while index < len(times):
        if times[index] is not None:
            index += 1
            continue
        end = index
        while end < len(times) and times[end] is None:
            end += 1
        before = times[index - 1][1] if index > 0 else 0.0  # type: ignore[index]
        after = times[end][0] if end < len(times) else before  # type: ignore[index]
        _share(times, keys, index, end, before, max(before, after))
        index = end
    return Transcript(text, tuple(Word(token.text, *span) for token, span in zip(wanted, times, strict=True)))  # type: ignore[misc]


def trim_silence(words: Sequence[Word], sound: Sound, hop: float = 0.01) -> list[Word]:
    """``words`` without the silence at their start and end: each spans
    from its first to its last loud moment (louder than the recording's
    quiet parts, its 10th percentile, by 12 dB, over 50 ms)."""
    samples = _mono(sound, sound.sample_rate)
    step = max(1, round(hop * sound.sample_rate))
    count = len(samples) // step
    if count == 0:
        return list(words)
    frames = samples[: count * step].reshape(count, step)
    # Power over 50 ms, so a click in a pause is not a word.
    power = np.convolve(np.mean(frames**2, axis=1), np.ones(5) / 5, mode="same")
    loudness = 10 * np.log10(power + 1e-12)
    heard = loudness[loudness > -90.0]  # without digital silence (before a recording starts)
    loud = loudness > np.percentile(heard if heard.size else loudness, 10) + 12.0
    trimmed = []
    for word in words:
        first, last = int(word.start / hop), min(count, int(np.ceil(word.end / hop)))
        voiced = np.flatnonzero(loud[first:last])
        if voiced.size == 0:
            trimmed.append(word)
            continue
        start = max(word.start, float(first + voiced[0]) * hop)
        end = min(word.end, float(first + voiced[-1] + 1) * hop)
        trimmed.append(Word(word.text, start, max(end, start + hop)))
    return trimmed


def _share(times: list[tuple[float, float] | None], keys: Sequence[str], first: int, end: int, start: float, stop: float) -> None:
    """Give words ``first`` to ``end`` the span ``start`` to ``stop``, each by its letters."""
    lengths = np.array([len(keys[index]) for index in range(first, end)], dtype=float)
    edges = start + (stop - start) * np.concatenate(([0.0], np.cumsum(lengths))) / lengths.sum()
    for offset, index in enumerate(range(first, end)):
        times[index] = (float(edges[offset]), float(edges[offset + 1]))


def _lines(words: Sequence[Word]) -> str:
    """Recognized words as text: a new line after every sentence."""
    lines, line = [], []
    for word in words:
        line.append(word.text)
        if word.text.endswith((".", "!", "?")):
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def _mono(sound: Sound, sample_rate: int) -> np.ndarray:
    """``sound`` as one float32 channel at ``sample_rate``."""
    samples = sound.samples.mean(axis=1)
    if sound.sample_rate != sample_rate:
        times = np.arange(round(len(samples) * sample_rate / sound.sample_rate)) / sample_rate
        samples = np.interp(times, np.arange(len(samples)) / sound.sample_rate, samples)
    return samples.astype(np.float32)
