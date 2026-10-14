"""The edits the preview page makes, as commands that can be undone.

Every edit is a :class:`UICommand`: plain data saying what to change.
:meth:`ProjectEditor.run <manimx.web.editing.ProjectEditor.run>` applies
it and gets back the command that undoes it, which it keeps on the undo
stack; undoing runs that command and keeps *its* inverse on the redo
stack. So every command only has to know how to apply itself and what
its inverse is, and undo and redo are the same operation.

    command = parse("/api/clip", {"index": 0, "start": 2.0})  # EditClips
    editor.run(command)                                        # moved
    editor.undo()                                              # back again
    editor.redo()                                              # moved again

Design decisions:

* **Commands are data, the editor applies them.** A command does not
  touch the project itself; the editor dispatches on its type with
  ``match`` (see ``ProjectEditor._apply``). The rules of the project (a
  following clip keeps its place when the one before is removed, ...) stay
  in one place, and a command is easy to test and to print.
* **Inverses are computed when a command runs**, from the state it
  changes, not stored with it: a redo after an undo is exact even when
  other edits changed things around it.
* **A drag is one step.** The page sends a live edit at every step of a
  drag, all with the same ``gesture`` id. Only the first step's inverse is
  kept, so one undo puts the dragged clips back where the drag started
  (see ``ProjectEditor.run``).
* **Removed clips are kept, not prepared again.** Undoing a removal puts
  the very same clip (with its prepared scene, cut and speed) back; only a
  scene whose file changed in the meantime is prepared again.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..project import ClipEntry

if TYPE_CHECKING:
    from ..animation.timing import Clip
    from ..voice.script import Script

CLIP_FIELDS = ("start", "layer", "cut", "speed", "cues")


class EditError(ValueError):
    """An edit that does not fit the project (the page shows the message)."""


class UICommand(ABC):
    """One edit of the project from the preview page (see the module docs)."""

    @property
    @abstractmethod
    def label(self) -> str:
        """What the edit does, for the page's undo and redo buttons."""


@dataclass(frozen=True)
class ClipEdit:
    """New values for some of a clip entry's fields (:data:`CLIP_FIELDS`)."""

    index: int
    updates: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class EditClips(UICommand):
    """Change clips' start, layer, cut or speed (a drag moves several at once)."""

    edits: tuple[ClipEdit, ...]

    @property
    def label(self) -> str:
        fields = {key for edit in self.edits for key, _ in edit.updates}
        verb = "cut" if "cut" in fields else "move" if fields & {"start", "layer"} else "change the speed of"
        return f"{verb} {_clips(len(self.edits))}"


@dataclass(frozen=True)
class AddClip(UICommand):
    """Put a new clip of ``scene`` on the timeline (it is prepared then),
    at position ``index`` of the clip list (None: last)."""

    scene: str
    start: float
    layer: int = 0
    cut: tuple[float | None, float | None] | None = None
    speed: float = 1.0
    index: int | None = None

    @property
    def label(self) -> str:
        return f"add {self.scene}"


@dataclass(frozen=True)
class RemoveClips(UICommand):
    """Take clips off the timeline. A clip that followed a removed one keeps
    its place, unless ``keep_followers`` is off (undoing an added clip: the
    next clip follows the one before again)."""

    indices: tuple[int, ...]
    keep_followers: bool = True

    @property
    def label(self) -> str:
        return f"remove {_clips(len(self.indices))}"


@dataclass(frozen=True)
class SplitClips(UICommand):
    """Split each clip at global ``time`` in two: the clip is cut to end
    there, and a new clip of its scene starts there, cut to go on from the
    same point, with its layer and speed."""

    indices: tuple[int, ...]
    time: float

    @property
    def label(self) -> str:
        return f"split {_clips(len(self.indices))}"


@dataclass(frozen=True)
class DuplicateClips(UICommand):
    """A copy of each clip (its cut, layer and speed), right after it."""

    indices: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"duplicate {_clips(len(self.indices))}"


@dataclass(frozen=True)
class AddAudio(UICommand):
    """Put a sound file of the project, or a part of a script, on an
    audio track, at position ``index`` of the audio clip list (None: last)."""

    file: str | None
    start: float
    track: int = 0
    cut: tuple[float | None, float | None] | None = None
    volume: float = 1.0
    index: int | None = None
    script: str | None = None
    part: int = 0

    @property
    def label(self) -> str:
        if self.script is not None:
            return f"place part {self.part + 1} of {self.script}"
        return f"add {(self.file or '').rsplit('/', 1)[-1]}"


@dataclass(frozen=True)
class RemoveAudio(UICommand):
    """Take audio clips off their tracks."""

    indices: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"remove {_count(len(self.indices), 'an audio clip', 'audio clips')}"


@dataclass(frozen=True)
class EditAudio(UICommand):
    """Change audio clips' start, track, cut or volume."""

    edits: tuple[ClipEdit, ...]

    @property
    def label(self) -> str:
        fields = {key for edit in self.edits for key, _ in edit.updates}
        verb = "cut" if "cut" in fields else "move" if fields & {"start", "track"} else "change the volume of"
        return f"{verb} {_count(len(self.edits), 'an audio clip', 'audio clips')}"


@dataclass(frozen=True)
class Batch(UICommand):
    """Several commands as one step: run in order, undone in reverse."""

    commands: tuple[UICommand, ...]
    name: str

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class RemovedClip:
    """A clip taken off the timeline, as it was, to put back."""

    index: int
    entry: ClipEntry
    clip: Clip
    follower: ClipEntry | None
    """The next clip's entry before the removal, when it followed this clip
    (it got this clip's start then, to keep its place)."""
    scene_version: int
    """The editor's version of the scene when it was removed: a newer one
    means the scene's file changed, and the clip is prepared again."""


@dataclass(frozen=True)
class InsertClips(UICommand):
    """Put removed clips back (the undo of :class:`RemoveClips`), in index order."""

    clips: tuple[RemovedClip, ...]

    @property
    def label(self) -> str:
        return f"restore {_clips(len(self.clips))}"


@dataclass(frozen=True)
class SetEvent(UICommand):
    """Set an event's time, or with None remove it."""

    name: str
    time: float | None

    @property
    def label(self) -> str:
        return f"time {self.name}" if self.time is not None else f"remove {self.name}"


@dataclass(frozen=True)
class SetScript(UICommand):
    """Make the script ``name`` this one (None: delete it)."""

    name: str
    script: Script | None

    @property
    def label(self) -> str:
        return f"{'change' if self.script is not None else 'delete'} the script {self.name}"


@dataclass(frozen=True)
class SetScriptText(UICommand):
    """Write the script ``name``'s text (a new script if there is none):
    its words keep the times of the words they match, new words are timed
    as if read out (see manimx/voice/script.py)."""

    name: str
    text: str

    @property
    def label(self) -> str:
        return f"edit the script {self.name}"


@dataclass(frozen=True)
class MoveWords(UICommand):
    """Set the times of words of a script part: ``(index, start, end)``
    each, in seconds from the part's start."""

    script: str
    part: int
    words: tuple[tuple[int, float, float], ...]

    @property
    def label(self) -> str:
        return f"time {_count(len(self.words), 'a word', 'words')}"


@dataclass(frozen=True)
class EditTrack(UICommand):
    """Change a track: ``kind`` "scenes" or "audio", its ``index``, and
    any of name, hidden, locked, muted, solo and volume."""

    kind: str
    index: int
    updates: tuple[tuple[str, Any], ...]

    @property
    def label(self) -> str:
        keys = {key for key, _ in self.updates}
        what = "rename" if keys == {"name"} else "change the volume of" if keys == {"volume"} else "change"
        return f"{what} a track"


@dataclass(frozen=True)
class SetSetting(UICommand):
    """Set the project setting at ``path``."""

    path: tuple[str, ...]
    value: Any

    @property
    def label(self) -> str:
        return f"set {'.'.join(self.path)}"


def parse(route: str, body: dict[str, Any]) -> UICommand:
    """The command a POST to ``route`` with JSON ``body`` asks for, checked."""
    match route, body:
        case "/api/clip", {"clips": list(edits)}:
            return EditClips(tuple(_clip_edit(edit) for edit in edits))
        case "/api/clip", {"index": _}:
            return EditClips((_clip_edit(body),))
        case "/api/clip/add", {"clips": list(clips)}:
            return Batch(tuple(_add_clip(clip) for clip in clips), f"paste {_clips(len(clips))}")
        case "/api/clip/add", {"scene": str()}:
            return _add_clip(body)
        case "/api/clip/remove", {"indices": list(indices)}:
            return RemoveClips(tuple(_index(index) for index in indices))
        case "/api/clip/split", {"indices": list(indices), "time": time}:
            return SplitClips(tuple(_index(index) for index in indices), _number(time, "time", minimum=0.0))
        case "/api/clip/duplicate", {"indices": list(indices)}:
            return DuplicateClips(tuple(_index(index) for index in indices))
        case "/api/event", {"name": str(name)} if name:
            time = body.get("time")
            return SetEvent(name, None if time is None else _number(time, "time", minimum=0.0))
        case "/api/setting", {"path": list(path)} if path and all(isinstance(part, str) for part in path):
            return SetSetting(tuple(path), body.get("value"))
        case "/api/audio/add", {"clips": list(clips)}:
            return Batch(tuple(_add_audio(clip) for clip in clips), f"paste {_count(len(clips), 'an audio clip', 'audio clips')}")
        case "/api/audio/add", {"file": str()} | {"script": str()}:
            return _add_audio(body)
        case "/api/audio/remove", {"indices": list(indices)}:
            return RemoveAudio(tuple(_index(index) for index in indices))
        case "/api/audio/edit", {"clips": list(edits)}:
            return EditAudio(tuple(_audio_edit(edit) for edit in edits))
        case "/api/audio/edit", {"index": _}:
            return EditAudio((_audio_edit(body),))
        case "/api/script", {"name": str(name), "text": str(text)}:
            return SetScriptText(_script_name(name), text)
        case "/api/script", {"name": str(name), "delete": True}:
            return SetScript(name, None)
        case "/api/script/words", {"script": str(script), "part": part, "words": list(words)}:
            return MoveWords(script, _index(part), tuple(_word_time(word) for word in words))
        case "/api/track", {"kind": "scenes" | "audio" as kind, "index": index}:
            return EditTrack(kind, _layer(index), tuple(track_updates({key: value for key, value in body.items() if key not in ("kind", "index")}).items()))
        case "/api/track", _:
            raise EditError(f"{route} does not take {body}")
        case "/api/script" | "/api/script/words", _:
            raise EditError(f"{route} does not take {body}")
        case "/api/audio/add" | "/api/audio/remove" | "/api/audio/edit", _:
            raise EditError(f"{route} does not take {body}")
        case "/api/clip" | "/api/clip/add" | "/api/clip/remove" | "/api/clip/split" | "/api/clip/duplicate" | "/api/event" | "/api/setting", _:
            raise EditError(f"{route} does not take {body}")
    raise EditError(f"no edit {route}")


def clip_updates(changes: dict[str, Any]) -> dict[str, Any]:
    """``changes`` to a clip entry, checked."""
    unknown = set(changes) - set(CLIP_FIELDS)
    if unknown:
        raise EditError(f"clips have no {', '.join(sorted(unknown))}")
    updates: dict[str, Any] = {}
    for key, value in changes.items():
        match key, value:
            case "start", None:
                updates[key] = None
            case "start", _:
                updates[key] = _number(value, "start", minimum=0.0)
            case "layer", _:
                updates[key] = _layer(value)
            case "cut", _:
                updates[key] = _cut(value)
            case "speed", _:
                updates[key] = _number(value, "speed")
                if updates[key] <= 0.0:
                    raise EditError("a speed must be positive")
            case "cues", dict() if all(isinstance(cue, str) and isinstance(event, str) and event for cue, event in value.items()):
                updates[key] = dict(value)
            case "cues", _:
                raise EditError("cues map cue names to event names")
    return updates


AUDIO_FIELDS = ("start", "track", "cut", "volume")


def audio_updates(changes: dict[str, Any]) -> dict[str, Any]:
    """``changes`` to an audio entry, checked."""
    unknown = set(changes) - set(AUDIO_FIELDS)
    if unknown:
        raise EditError(f"audio clips have no {', '.join(sorted(unknown))}")
    updates: dict[str, Any] = {}
    for key, value in changes.items():
        match key:
            case "start":
                updates[key] = _number(value, "start", minimum=0.0)
            case "track":
                updates[key] = _layer(value)
                if updates[key] < 0:
                    raise EditError("audio tracks start at 0")
            case "cut":
                updates[key] = _cut(value)
            case "volume":
                updates[key] = _number(value, "volume", minimum=0.0)
    return updates


def _add_audio(body: Any) -> AddAudio:
    """``{"file" or "script" and "part", "start", "track", "cut", "volume"}``."""
    if not isinstance(body, dict) or not (isinstance(body.get("file"), str) or isinstance(body.get("script"), str)):
        raise EditError("an audio clip needs a file or a script part")
    updates = audio_updates({key: body[key] for key in AUDIO_FIELDS if key in body})
    script = body.get("script")
    return AddAudio(
        None if script is not None else body["file"], updates.get("start", 0.0), updates.get("track", 0), updates.get("cut"),
        updates.get("volume", 1.0), script=script, part=_index(body.get("part", 0)) if script is not None else 0,
    )  # fmt: skip


def _audio_edit(edit: Any) -> ClipEdit:
    if not isinstance(edit, dict):
        raise EditError("an audio clip edit is an object")
    changes = {key: value for key, value in edit.items() if key != "index"}
    return ClipEdit(_index(edit.get("index")), tuple(audio_updates(changes).items()))


def track_updates(changes: dict[str, Any]) -> dict[str, Any]:
    """``changes`` to a track, checked."""
    updates: dict[str, Any] = {}
    for key, value in changes.items():
        match key, value:
            case "name", str():
                updates[key] = value.strip()[:60]
            case "hidden" | "locked" | "muted" | "solo", bool():
                updates[key] = value
            case "volume", _:
                updates[key] = min(_number(value, "volume", minimum=0.0), 4.0)
            case _:
                raise EditError(f"tracks have no {key} like {value!r}")
    return updates


def _script_name(name: str) -> str:
    from ..voice.script import valid_name

    if not valid_name(name):
        raise EditError("a script's name starts with a letter and has only letters, digits, - and _ (it starts its events' names)")
    return name


def _word_time(word: Any) -> tuple[int, float, float]:
    """``{"index", "start", "end"}`` of a word, checked."""
    if not isinstance(word, dict):
        raise EditError("a word's time is an object")
    start = _number(word.get("start"), "start", minimum=0.0)
    end = _number(word.get("end"), "end", minimum=start)
    return (_index(word.get("index")), start, end)


def _add_clip(body: Any) -> AddClip:
    """``{"scene", "start", "layer", "cut", "speed"}`` (all but the scene optional)."""
    if not isinstance(body, dict) or not isinstance(body.get("scene"), str):
        raise EditError("a new clip needs a scene")
    updates = clip_updates({key: body[key] for key in ("layer", "cut", "speed") if key in body})
    start = _number(body.get("start", 0.0), "start", minimum=0.0)
    return AddClip(body["scene"], start, updates.get("layer", 0), updates.get("cut"), updates.get("speed", 1.0))


def _clip_edit(edit: Any) -> ClipEdit:
    if not isinstance(edit, dict):
        raise EditError("a clip edit is an object")
    changes = {key: value for key, value in edit.items() if key != "index"}
    return ClipEdit(_index(edit.get("index")), tuple(clip_updates(changes).items()))


def _index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EditError(f"there is no clip {value!r}")
    return value


def _layer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EditError("a layer is a whole number")
    return value


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EditError(f"{name} must be a number")
    if minimum is not None and value < minimum:
        raise EditError(f"{name} must be at least {minimum:g}")
    return float(value)


def _cut(value: Any) -> tuple[float | None, float | None] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EditError("a cut is [in, out]")
    cut_in, cut_out = (None if point is None else _number(point, "a cut point", minimum=0.0) for point in value)
    if cut_in is not None and cut_out is not None and cut_out <= cut_in:
        raise EditError("a cut's out point must come after its in point")
    return None if cut_in is None and cut_out is None else (cut_in, cut_out)


def _clips(count: int) -> str:
    return _count(count, "a clip", "clips")


def _count(count: int, one: str, many: str) -> str:
    return one if count == 1 else f"{count} {many}"


__all__ = [
    "AUDIO_FIELDS",
    "CLIP_FIELDS",
    "AddAudio",
    "AddClip",
    "Batch",
    "ClipEdit",
    "DuplicateClips",
    "EditAudio",
    "EditClips",
    "EditError",
    "InsertClips",
    "RemoveAudio",
    "RemoveClips",
    "RemovedClip",
    "SetEvent",
    "SetSetting",
    "SplitClips",
    "UICommand",
    "clip_updates",
    "parse",
]
