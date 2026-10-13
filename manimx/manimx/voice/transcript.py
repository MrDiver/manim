"""Transcripts: what a recording says, word by word, with times.

A :class:`Transcript` is the text of a recording (as its author writes
it, with punctuation, line breaks and ``[section]`` markers) and one
:class:`Word` per word of that text, with the seconds of the sound file
in which it is spoken. alignment.py makes the times (Whisper, matched to
the text); here they are only data.

    transcript = Transcript.load("assets/audio/voice.transcript.json")
    transcript.events("voice")    # {"voice.everything": 0.88, ...}
    transcript.subtitles()        # [Subtitle(0.88, 4.76, "Everything in ..."), ...]

Words become events, so scenes sync to them with ``self.cue(...)`` or
``self.wait_for(...)``. The timeline adds the times of every transcript
on its audio tracks (timing.py); the names, for the audio clip ``voice``:

* ``voice.square``: the first "square" (``voice.square.2`` the second);
* ``voice.middle.square``: the first "square" after the ``[middle]``
  marker (``voice.middle.square.2`` the second after it);
* ``voice.[middle]``: the first word after the ``[middle]`` marker.

A word's name is its letters and digits in lower case ("Let's" is
``lets``, "voice." is ``voice``); markers are only names, not spoken,
and show in no subtitle.

Design decisions:

* **The text is the author's.** Whisper's own text is only a first draft:
  the author corrects it (or brings a script), and the words keep the
  times of the recognized words they match (alignment.py). So event names
  and subtitles come from the text the author wrote.
* **One file per recording**, next to it: ``voice.m4a`` has its
  transcript in ``voice.transcript.json``. Every audio clip of that file
  uses it, and editing it in the preview changes all of them.
* **Events are word starts.** A cue lands where the word begins; the end
  of a word is less precise (it is only used for subtitles and drawing).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSCRIPT_SUFFIX = ".transcript.json"
"""``voice.m4a``'s transcript is ``voice.transcript.json``."""

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

    @property
    def tokens(self) -> list[Token]:
        return tokens(self.text)

    def event_names(self, name: str) -> list[str]:
        """Each word's own event, in order: ``voice.square``, then
        ``voice.square.2`` for the second "square" (what a scene cues)."""
        seen: dict[str, int] = {}
        names = []
        for token in self.tokens:
            count = seen[token.key] = seen.get(token.key, 0) + 1
            names.append(f"{name}.{token.key}" if count == 1 else f"{name}.{token.key}.{count}")
        return names

    def events(self, name: str) -> dict[str, float]:
        """Every word's events (see the module docs), prefixed by ``name``,
        at the seconds of the sound file the word starts."""
        events: dict[str, float] = {}
        seen: dict[tuple[str | None, str], int] = {}
        for token, word in zip(self.tokens, self.words, strict=True):
            scopes = (None,) if token.section is None else (None, token.section)
            for scope in scopes:
                count = seen[scope, token.key] = seen.get((scope, token.key), 0) + 1
                prefix = name if scope is None else f"{name}.{scope}"
                events[f"{prefix}.{token.key}" if count == 1 else f"{prefix}.{token.key}.{count}"] = word.start
            if token.section is not None:
                events.setdefault(f"{name}.[{token.section}]", word.start)
        return events

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

    # -- files ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "words": [{"text": word.text, "start": round(word.start, 3), "end": round(word.end, 3)} for word in self.words],
        }

    @classmethod
    def from_json(cls, data: Any) -> Transcript:
        if not isinstance(data, dict) or not isinstance(data.get("text"), str) or not isinstance(data.get("words"), list):
            raise ValueError('a transcript is {"text": ..., "words": [...]}')
        words = tuple(Word(str(word["text"]), float(word["start"]), float(word["end"])) for word in data["words"])
        text = data["text"]
        if len(tokens(text)) != len(words):
            # The text was edited by hand: its words take the times of the words they match.
            from .alignment import align

            return align(text, words)
        return cls(text, words)

    @classmethod
    def load(cls, path: str | Path) -> Transcript:
        try:
            return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path} is not a transcript: {error}") from None

    def save(self, path: str | Path) -> None:
        """Write it as JSON, a word per line (easy to read and to diff)."""
        data = self.to_json()
        words = ",\n".join(f"    {json.dumps(word, ensure_ascii=False)}" for word in data["words"])
        text = json.dumps(data["text"], ensure_ascii=False)
        Path(path).write_text(f'{{\n  "text": {text},\n  "words": [\n{words}\n  ]\n}}\n', encoding="utf-8")


def transcript_path(sound: str | Path) -> Path:
    """Where the transcript of the sound file ``sound`` is kept."""
    sound = Path(sound)
    return sound.with_name(sound.stem + TRANSCRIPT_SUFFIX)


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
