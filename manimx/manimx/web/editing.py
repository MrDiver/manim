"""Editing a project from the preview page: clips, events and settings,
written to ``manimx.json``.

    editor = ProjectEditor(project.root)
    editor.edit_clip(0, {"start": 2.0, "cut": [0.5, None]})
    editor.set_event("next", 4.0)
    editor.set_setting(("output", "fps"), 30)

Every edit loads ``manimx.json``, changes it, checks it the way loading
checks it and saves it; the reloader (reload.py) sees the file change and
rebuilds the timeline, which the page then fetches. So the file stays the
one truth: the page, an editor and ``manimx settings`` all change the same
thing, and undo is the file's history.

Design decisions:

* **Saved on every edit.** There is no unsaved state in the page to lose
  or to merge; a drag is one edit when it is let go.
* **Clips by index.** The page's clips are the timeline's, which the
  project builds from its clip list in order, so clip ``i`` is entry
  ``i``. An edit names the values it changes and leaves the others.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from pathlib import Path
from typing import Any

from ..project import ClipEntry, Project, ProjectError
from ..settings import SettingsError, entries, get, set_value


class EditError(ValueError):
    """An edit that does not fit the project (the page shows the message)."""


class ProjectEditor:
    """Edits the project in ``root`` (see the module docs)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    # -- reading ------------------------------------------------------------

    def clips(self) -> list[dict[str, Any]]:
        """Each clip entry as written: start (None: after the previous
        clip), layer, cut and speed."""
        return [_entry_json(entry) for entry in self._load().timeline]

    def events(self) -> dict[str, float]:
        """The event times set in the project."""
        return dict(self._load().events)

    def settings(self) -> list[dict[str, Any]]:
        """Every settings section and value, with what a form needs to
        show and check it (see settings.py)."""
        settings = self._load().settings
        described = []
        for entry in entries():
            meta = entry.field.metadata
            item: dict[str, Any] = {"path": list(entry.path), "section": entry.is_section, "description": meta.get("description", "")}
            if not entry.is_section:
                note = meta.get("note")
                item |= {
                    "value": get(settings, entry.path),
                    "default": entry.field.default,
                    "type": type(entry.field.default).__name__,
                    "minimum": meta.get("minimum"),
                    "maximum": meta.get("maximum"),
                    "choices": list(meta["choices"]) if meta.get("choices") else None,
                    "unit": meta.get("unit"),
                    "color": bool(meta.get("color")),
                    "note": note(settings) if note is not None else "",
                }
            described.append(item)
        return described

    # -- editing ------------------------------------------------------------

    def edit_clip(self, index: int, changes: dict[str, Any]) -> None:
        """Change clip ``index``'s ``start`` (seconds, or None to follow
        the previous clip), ``layer``, ``cut`` (``[in, out]``, either None,
        or None for no cut) and ``speed``."""
        unknown = set(changes) - {"start", "layer", "cut", "speed"}
        if unknown:
            raise EditError(f"clips have no {', '.join(sorted(unknown))}")
        with self._lock:
            project = self._load()
            if not 0 <= index < len(project.timeline):
                raise EditError(f"there is no clip {index}")
            entry = project.timeline[index]
            updates: dict[str, Any] = {}
            if "start" in changes:
                updates["start"] = None if changes["start"] is None else _number(changes["start"], "start", minimum=0.0)
            if "layer" in changes:
                layer = changes["layer"]
                if isinstance(layer, bool) or not isinstance(layer, int):
                    raise EditError("a layer is a whole number")
                updates["layer"] = layer
            if "cut" in changes:
                updates["cut"] = _cut(changes["cut"])
            if "speed" in changes:
                speed = _number(changes["speed"], "speed")
                if speed <= 0.0:
                    raise EditError("a speed must be positive")
                updates["speed"] = speed
            project.timeline[index] = dataclasses.replace(entry, **updates)
            project.save()

    def set_event(self, name: str, time: float | None) -> None:
        """Set ``name``'s time (the clips waiting for it go on then), or
        with None remove it."""
        if not isinstance(name, str) or not name:
            raise EditError("an event needs a name")
        with self._lock:
            project = self._load()
            if time is None:
                project.events.pop(name, None)
            else:
                project.events[name] = _number(time, "time", minimum=0.0)
            project.save()

    def set_setting(self, path: list[str] | tuple[str, ...], value: Any) -> None:
        """Set the setting at ``path``, checked like ``manimx.json``."""
        with self._lock:
            project = self._load()
            try:
                set_value(project.settings, tuple(path), value)
            except (SettingsError, StopIteration, AttributeError) as error:
                raise EditError(str(error) or f"no setting {'.'.join(path)}") from None
            project.save()

    def _load(self) -> Project:
        try:
            return Project.load(self.root)
        except ProjectError as error:
            raise EditError(str(error)) from None


def _entry_json(entry: ClipEntry) -> dict[str, Any]:
    return {"scene": entry.scene, "start": entry.start, "layer": entry.layer, "cut": None if entry.cut is None else list(entry.cut), "speed": entry.speed}


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


__all__ = ["EditError", "ProjectEditor"]
