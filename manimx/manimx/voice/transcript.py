"""Transcripts: what is said, word by word, with times.

A :class:`Transcript` is a text as its author writes it (punctuation,
line breaks, ``[section]`` markers) and one :class:`Word` per word of
that text, with when it is said. A script's parts are transcripts
(script.py); alignment.py times their words (Whisper, matched to the
text), and :meth:`Transcript.fake` times them as if read out evenly.

    transcript = Transcript.fake("Every circle, every square.")
    transcript.subtitles()   # [Subtitle(0.0, 1.9, "Every circle, every square.")]

A word's name in events is its letters and digits in lower case
("Let's" is ``lets``, "voice." is ``voice``, see :func:`word_key`);
markers are only names, not spoken, and show in no subtitle.

Design decisions:

* **The text is the author's.** Whisper's own text is only a first draft:
  the words of the author's text keep the times of the recognized words
  they match (alignment.py), so event names and subtitles come from what
  the author wrote.
* **Events are word starts.** A cue lands where the word begins; the end
  of a word is less precise (it is only used for subtitles and drawing).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

SUBTITLE_CHARACTERS = 42
"""The longest subtitle line, in characters (a common broadcast limit)."""

PAUSE_CHARACTERS = 20.0
"""How much a pause draws a long line's split to it: characters of
imbalance per second of pause."""

_MARKER =re.compile(r"^\[([\w-]+)\]$")


@dataclass(frozen=True, slots=True)
class Word:
    """A spoken word: as written in the text, and when (seconds of the sound file)."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Token:
    """A word of the text, before it has a time."""

    text: str
    key: str
    """Its letters and digits in lower case: the name in its events."""
    section: str | None
    """The last ``[marker]`` before it."""
    line: int
    """The text line it is on (a subtitle never spans two lines)."""


@dataclass(frozen=True, slots=True)
class Subtitle:
    start: float
    end: float
    text: str


def tokens(text: str) -> list[Token]:
    """The words of ``text``, without its ``[section]`` markers."""
    result: list[Token] = []
    section = None
    for line, content in enumerate(text.splitlines()):
        for part in content.split():
            marker = _MARKER.match(part)
            if marker:
                section = marker.group(1)
                continue
            key = word_key(part)
            if key:
                result.append(Token(part, key, section, line))
    return result


def word_key(text: str) -> str:
    """``text``'s letters and digits in lower case ("Let's": ``lets``)."""
    return "".join(character for character in text.lower() if character.isalnum())


@dataclass(frozen=True, eq=False)
class Transcript:
    """The text of a recording and the time of each of its words."""

    text: str
    words: tuple[Word, ...]
    """One per word of ``text`` (see :func:`tokens`), in order."""

    def __post_init__(self) -> None:
        count = len(tokens(self.text))
        if count != len(self.words):
            raise ValueError(f"the transcript's text has {count} words but {len(self.words)} have times")

    @classmethod
    def fake(
        cls, text: str, start: float = 0.0, letters_per_second: float = 14.0, sentence_pause: float = 0.3, line_pause: float = 0.5
    ) -> Transcript:
        """``text`` as if read out evenly, before anything is recorded
        (reanimate's fakeTranscript): each word takes time by its letters,
        one more for the space after it (14 a second: about 120 words a
        minute), with a pause after each sentence and each line.

        A scene written along it has cues whose words have no events yet:
        they bend nothing, and the clip plays at this pace until the
        recording is transcribed, then its cues land on the spoken words."""
        words = []
        time = start
        tokens_ = tokens(text)
        for index, token in enumerate(tokens_):
            end = time + len(token.key) / letters_per_second
            words.append(Word(token.text, time, end))
            time = end + 1 / letters_per_second
            following = tokens_[index + 1] if index + 1 < len(tokens_) else None
            if following is not None and following.line != token.line:
                time += line_pause
            elif token.text.endswith((".", "!", "?")):
                time += sentence_pause
        return cls(text, tuple(words))

    @property
    def tokens(self) -> list[Token]:
        return tokens(self.text)

    def subtitles(self, characters: int = SUBTITLE_CHARACTERS) -> list[Subtitle]:
        """The words as subtitles: one per text line, lines longer than
        ``characters`` split at their longest pauses."""
        lines: dict[int, list[Word]] = {}
        for token, word in zip(self.tokens, self.words, strict=True):
            lines.setdefault(token.line, []).append(word)
        subtitles = [subtitle for words in lines.values() for subtitle in _split(words, characters)]
        # A subtitle ends where the next one starts, at the latest.
        return [
            Subtitle(subtitle.start, min(subtitle.end, following.start) if following else subtitle.end, subtitle.text)
            for subtitle, following in zip(subtitles, [*subtitles[1:], None], strict=True)
        ]


def srt(subtitles: Iterable[Subtitle]) -> str:
    """``subtitles`` as a SubRip (.srt) file."""

    def stamp(seconds: float) -> str:
        milliseconds = max(0, round(seconds * 1000))
        hours, rest = divmod(milliseconds, 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        return f"{hours:02}:{minutes:02}:{rest // 1000:02},{rest % 1000:03}"

    return "".join(f"{number}\n{stamp(item.start)} --> {stamp(item.end)}\n{item.text}\n\n" for number, item in enumerate(subtitles, 1))


def _split(words: Sequence[Word], characters: int) -> list[Subtitle]:
    """``words`` (one text line) as subtitles of at most ``characters``,
    split at the longest pause between two words."""
    text = " ".join(word.text for word in words)
    if len(text) <= characters or len(words) == 1:
        return [Subtitle(words[0].start, words[-1].end, text)]
    # Split into halves of about the same length, preferring a pause
    # (a second of pause is worth PAUSE_CHARACTERS of balance).
    best, best_score = 1, float("inf")
    for index in range(1, len(words)):
        left = len(" ".join(word.text for word in words[:index]))
        right = len(" ".join(word.text for word in words[index:]))
        pause = words[index].start - words[index - 1].end
        score = max(left, right) - PAUSE_CHARACTERS * pause
        if score < best_score:
            best, best_score = index, score
    return _split(words[:best], characters) + _split(words[best:], characters)
