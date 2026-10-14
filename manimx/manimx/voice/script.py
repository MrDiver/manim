"""Scripts: what is said in a project, in parts recorded one at a time.

A script is a text file of the project, ``scripts/narration.txt``,
written before anything is recorded:

    Everything in this animation is timed by my voice.
    Every flash, every circle, every square.
    ---
    [middle]
    I decide when everything begins and ends.

A line ``---`` ends a *part*: what is recorded in one go (a take). A line
``[middle]`` names a section, for event names (see transcript.py). Each
part goes onto the timeline as an audio clip of its own
(``{"script": "narration", "part": 1}`` in manimx.json). Its words are
events at once: before the part is recorded, at fake times (read out
evenly, :meth:`Transcript.fake`), which can be moved by hand; once a take
is recorded, at the times Whisper hears them. So scenes synced to the
script with cues follow the plan first, then the recording.

Next to the text, ``scripts/narration.json`` keeps what the text cannot:
each part's word times and its take:

    {"parts": [{"words": [{"text": "Everything", "start": 0.0, "end": 0.7}, ...],
                "take": {"file": "assets/audio/narration-1.webm", "offset": 3.0}},
               {"words": [...], "take": null}]}

A word's times count from the start of its part; a take's ``offset`` is
where in the take's file the part starts (after the countdown).

Design decisions:

* **The text is its own file**, plain text, so it can be written in any
  editor, diffed and reviewed; only the times live in JSON.
* **Event names count over the whole script**, not per part: the second
  "square" is ``narration.square.2`` in whichever part it is, and moving
  a ``---`` line renames nothing.
* **Editing the text keeps times:** each part's new words take the times
  of the old words of that part they match (alignment.align); a new part
  starts with fake times.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .transcript import Subtitle, Transcript, Word, tokens

SCRIPTS_FOLDER = "scripts"
"""Where a project's scripts are."""

_PART_BREAK = re.compile(r"^\s*-{3,}\s*$")


@dataclass(frozen=True)
class Take:
    """A part's recording: the sound file and where in it the part starts."""

    file: str
    offset: float = 0.0


@dataclass(frozen=True)
class Part:
    """What is recorded in one go: its text, its words' times (from the
    part's start) and its take, if recorded."""

    text: str
    words: tuple[Word, ...]
    take: Take | None = None

    @property
    def transcript(self) -> Transcript:
        return Transcript(self.text, self.words)

    @property
    def planned_length(self) -> float:
        """How long the part lasts before it is recorded: until its last word ends, and a breath."""
        return (self.words[-1].end if self.words else 0.0) + 0.5

    @property
    def title(self) -> str:
        """Its first words, to show it by."""
        words = [token.text for token in tokens(self.text)]
        return " ".join(words[:6]) + (" …" if len(words) > 6 else "")


def split_parts(text: str) -> list[str]:
    """``text``'s parts: the stretches between ``---`` lines."""
    parts: list[list[str]] = [[]]
    for line in text.splitlines():
        if _PART_BREAK.match(line):
            parts.append([])
        else:
            parts[-1].append(line)
    return ["\n".join(lines).strip("\n") for lines in parts]


@dataclass(frozen=True, eq=False)
class Script:
    """A script: its name (the prefix of its events), its text and parts."""

    name: str
    text: str
    parts: tuple[Part, ...] = field(default=())

    @classmethod
    def new(cls, name: str, text: str, previous: Script | None = None) -> Script:
        """The script ``text``; its parts keep the times (and takes) of
        ``previous``'s parts in the same place where their words match,
        others are timed as if read out evenly."""
        from .alignment import align

        parts = []
        for index, part_text in enumerate(split_parts(text)):
            old = previous.parts[index] if previous is not None and index < len(previous.parts) else None
            if old is not None and old.words:
                parts.append(Part(part_text, align(part_text, old.words).words, old.take))
            else:
                fake = Transcript.fake(part_text).words
                parts.append(Part(part_text, tuple(Word(word.text, round(word.start, 3), round(word.end, 3)) for word in fake)))
        return cls(name, text, tuple(parts))

    def with_part(self, index: int, part: Part) -> Script:
        parts = list(self.parts)
        parts[index] = part
        return replace(self, parts=tuple(parts))

    def event_names(self) -> list[list[list[str]]]:
        """Per part, per word, its event names (see the module docs):
        counted over the whole script, sections carried across parts."""
        seen: dict[tuple[str | None, str], int] = {}
        marked: set[str] = set()
        section: str | None = None
        names: list[list[list[str]]] = []
        for part in self.parts:
            part_names = []
            for token in tokens(part.text):
                section = token.section if token.section is not None else section
                word_names = []
                for scope in (None,) if section is None else (None, section):
                    count = seen[scope, token.key] = seen.get((scope, token.key), 0) + 1
                    prefix = self.name if scope is None else f"{self.name}.{scope}"
                    word_names.append(f"{prefix}.{token.key}" if count == 1 else f"{prefix}.{token.key}.{count}")
                if section is not None and section not in marked:
                    marked.add(section)
                    word_names.append(f"{self.name}.[{section}]")
                part_names.append(word_names)
            names.append(part_names)
        return names

    def subtitles(self, index: int) -> list[Subtitle]:
        """Part ``index``'s subtitles, in seconds from the part's start."""
        return self.parts[index].transcript.subtitles()

    # -- files ------------------------------------------------------------------

    def save(self, folder: Path) -> None:
        """Write ``<name>.txt`` and ``<name>.json`` into ``folder``."""
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.name}.txt").write_text(self.text if self.text.endswith("\n") else self.text + "\n", encoding="utf-8")
        parts = []
        for part in self.parts:
            words = ",\n".join(f"        {json.dumps({'text': w.text, 'start': round(w.start, 3), 'end': round(w.end, 3)}, ensure_ascii=False)}" for w in part.words)
            take = "null" if part.take is None else json.dumps({"file": part.take.file, "offset": round(part.take.offset, 3)})
            parts.append(f'    {{\n      "take": {take},\n      "words": [\n{words}\n      ]\n    }}')
        (folder / f"{self.name}.json").write_text('{\n  "parts": [\n' + ",\n".join(parts) + "\n  ]\n}\n", encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, name: str) -> Script:
        """Read ``<name>.txt`` (and its times from ``<name>.json``, if any)."""
        text = (folder / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")
        timing = folder / f"{name}.json"
        previous = None
        if timing.exists():
            try:
                previous = cls(name, "", tuple(_part(data) for data in json.loads(timing.read_text(encoding="utf-8"))["parts"]))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{timing} is not a script's timing: {error}") from None
        return cls.new(name, text, previous)  # unchanged words match exactly: their times stay


def script_names(folder: Path) -> list[str]:
    """The scripts in ``folder``, by name."""
    return sorted(path.stem for path in folder.glob("*.txt")) if folder.is_dir() else []


def valid_name(name: str) -> bool:
    """A script name: letters, digits, - and _, starting with a letter (it starts event names)."""
    return bool(re.fullmatch(r"[A-Za-z][\w-]*", name))


def _part(data: Any) -> Part:
    words = tuple(Word(str(word["text"]), float(word["start"]), float(word["end"])) for word in data["words"])
    take = data.get("take")
    return Part("", words, None if take is None else Take(str(take["file"]), float(take.get("offset", 0.0))))
