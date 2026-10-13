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

CLIP_FIELDS = ("start", "layer", "cut", "speed")


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
    """Put a new clip of ``scene`` on the timeline (it is prepared then)."""

    scene: str
    start: float
    layer: int = 0

    @property
    def label(self) -> str:
        return f"add {self.scene}"


@dataclass(frozen=True)
class RemoveClips(UICommand):
    """Take clips off the timeline."""

    indices: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"remove {_clips(len(self.indices))}"


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
        case "/api/clip/add", {"scene": str(scene)}:
            start = _number(body.get("start", 0.0), "start", minimum=0.0)
            return AddClip(scene, start, _layer(body.get("layer", 0)))
        case "/api/clip/remove", {"indices": list(indices)}:
            return RemoveClips(tuple(_index(index) for index in indices))
        case "/api/event", {"name": str(name)} if name:
            time = body.get("time")
            return SetEvent(name, None if time is None else _number(time, "time", minimum=0.0))
        case "/api/setting", {"path": list(path)} if path and all(isinstance(part, str) for part in path):
            return SetSetting(tuple(path), body.get("value"))
        case "/api/clip" | "/api/clip/add" | "/api/clip/remove" | "/api/event" | "/api/setting", _:
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
    return updates


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
    return "a clip" if count == 1 else f"{count} clips"


__all__ = [
    "CLIP_FIELDS",
    "AddClip",
    "ClipEdit",
    "EditClips",
    "EditError",
    "InsertClips",
    "RemoveClips",
    "RemovedClip",
    "SetEvent",
    "SetSetting",
    "UICommand",
    "clip_updates",
    "parse",
]
