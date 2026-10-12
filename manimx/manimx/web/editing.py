"""Editing the previewed project live: clips, events and settings change
the server's timeline at once, and ``manimx.json`` is saved shortly after.

    editor = ProjectEditor(root, preview_settings)
    server = PreviewServer(Timeline(), settings, editor=editor)
    editor.load()                                  # prepare every clip
    editor.edit_clip(0, {"start": 2.0})           # shown in the next frame
    editor.files_changed({root / "scenes/intro.py"})  # prepare that scene again

The timeline is a function of time: a clip's start, cut, layer and speed
and the events' times only decide which scene time a global time shows,
so changing them changes no prepared scene. An edit sets them on the live
:class:`~manimx.animation.timing.Timeline` and the next frame shows it;
while a clip is dragged, the page sends the edit at every step and asks
for the frame at the playhead, so the output is the preview of the drag.

Design decisions:

* **The live timeline is the truth while the page runs.** ``manimx.json``
  is written ``save_delay`` seconds after the last edit (autosave), and
  that write does not rebuild anything: the reloader knows the file's
  modification time from the save and leaves it alone.
* **Only what changed is prepared again.** When a scene's file is saved,
  only the clips of the scenes in that file are prepared again and
  swapped into their clips (start, cut, layer and speed stay; a scene
  that became longer simply ends later). A changed Python file that is no
  scene's own module (a helper several scenes import) prepares every
  scene, since any of them may use it. A ``manimx.json`` changed by
  someone else is read again, reusing every prepared scene whose clip
  still shows the same, unchanged scene.
* **Clips by index.** The timeline is built from the project's clip list
  in order, so clip ``i`` is entry ``i``.
* **Clips that follow the previous one** (no ``start`` in the file) are
  placed again after every edit, so moving or cutting a clip carries the
  clips that follow it along.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..animation import PreparedScene, Timeline
from ..animation.timing import NEVER
from ..drivers.offline import RenderSettings
from ..project import PROJECT_FILE, ClipEntry, Project
from ..settings import SettingsError, entries, get, set_value

if TYPE_CHECKING:
    from .server import PreviewServer

CLIP_FIELDS = ("start", "layer", "cut", "speed")


class EditError(ValueError):
    """An edit that does not fit the project (the page shows the message)."""


class ProjectEditor:
    """The project in ``root``, live in a preview server (see the module docs)."""

    def __init__(self, root: str | Path, settings: Callable[[Project], RenderSettings], save_delay: float = 1.0) -> None:
        self.root = Path(root).resolve()
        self.render_settings = settings
        """The preview's render settings of a project."""
        self.save_delay = save_delay
        self.project: Project | None = None
        self.timeline = Timeline()
        self.server: PreviewServer | None = None
        self._lock = threading.RLock()
        """Held while the project or the timeline changes."""
        self._timer: threading.Timer | None = None
        self._unsaved = False
        self.saved_mtime: int | None = None
        """manimx.json's modification time after the last autosave."""

    def attach(self, server: PreviewServer) -> None:
        """Called by the server this editor edits for."""
        self.server = server

    # -- building -------------------------------------------------------------

    def load(self) -> None:
        """Read manimx.json and prepare every clip (or report why not)."""
        self._rebuild(changed=None)

    def files_changed(self, paths: Iterable[Path]) -> None:
        """Bring the preview up to date with the files in ``paths``."""
        paths = {Path(path).resolve() for path in paths}
        project_file = self.root / PROJECT_FILE
        own_save = project_file in paths and _mtime(project_file) == self.saved_mtime
        python = {path for path in paths if path.suffix == ".py"}
        if (project_file in paths and not own_save) or self.project is None:
            self._rebuild(changed=python)
        elif python:
            self._prepare_again(python)

    def _rebuild(self, changed: set[Path] | None) -> None:
        """Read manimx.json and build the timeline again; scenes are reused
        unless their module is in ``changed`` (None: prepare all)."""
        with self._lock:
            try:
                project = Project.load(self.root)
                project.unload_modules()
                old = self.project
                previous = list(zip(old.timeline, self.timeline.clips, strict=False)) if old is not None and changed is not None else []
                files = _module_files(project)
                every = changed is not None and any(path not in files.values() for path in changed)

                def prepare(index: int, entry: ClipEntry) -> PreparedScene:
                    if index < len(previous) and previous[index][0].scene == entry.scene and not every and files.get(entry.scene) not in changed:
                        return previous[index][1].scene
                    return project.prepare(entry.scene)

                timeline = project.build_timeline(prepare=prepare)
                settings = self.render_settings(project)
            except Exception as error:  # a syntax error, a failing construct(): show it, keep the last timeline
                self._report(error)
                return
            self.project, self.timeline = project, timeline
            if self.server is not None:
                self.server.replace(timeline, settings)

    def _prepare_again(self, changed: set[Path]) -> None:
        """Prepare the scenes whose module is in ``changed`` again and show
        them in their clips (every scene for any other Python file)."""
        with self._lock:
            project = self.project
            try:
                project.unload_modules()
                files = _module_files(project)
                every = any(path not in files.values() for path in changed)
                fresh = {}
                for index, entry in enumerate(project.timeline):
                    if every or files.get(entry.scene) in changed:
                        fresh[index] = project.prepare(entry.scene)
            except Exception as error:
                self._report(error)
                return

            def swap() -> None:
                for index, scene in fresh.items():
                    self.timeline.clips[index].replace_scene(scene)
                self._place_followers()

            self._change(swap)

    def _report(self, error: BaseException) -> None:
        if self.server is not None:
            self.server.report(describe(error, self.root))

    # -- reading ------------------------------------------------------------

    def clips(self) -> list[dict[str, Any]]:
        """Each clip entry: start (None: after the previous clip), layer,
        cut and speed."""
        project = self.project
        return [] if project is None else [_entry_json(entry) for entry in project.timeline]

    def settings(self) -> list[dict[str, Any]]:
        """Every settings section and value, with what a form needs to
        show and check it (see settings.py)."""
        settings = self._project().settings
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
        or None for no cut) and ``speed``, live."""
        self.edit_clips([{"index": index, **changes}])

    def edit_clips(self, edits: list[dict[str, Any]]) -> None:
        """Several clip edits as one (a group of clips dragged together):
        ``{"index": i, ...changes}`` each, checked before any applies."""
        project = self._project()
        checked = []
        for edit in edits:
            index = edit.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(project.timeline):
                raise EditError(f"there is no clip {index!r}")
            checked.append((index, _clip_updates({key: value for key, value in edit.items() if key != "index"})))

        def apply() -> None:
            for index, updates in checked:
                entry = project.timeline[index] = dataclasses.replace(project.timeline[index], **updates)
                clip = self.timeline.clips[index]
                if "layer" in updates:
                    clip.set_layer(entry.layer)
                if "cut" in updates:
                    clip.cut(*(entry.cut or (None, None)))
                if "speed" in updates:
                    clip.set_speed(entry.speed)
            self._place_followers()

        self._change(apply)

    def set_event(self, name: str, time: float | None) -> None:
        """Set ``name``'s time (the clips waiting for it go on then), or
        with None remove it, live."""
        if not isinstance(name, str) or not name:
            raise EditError("an event needs a name")
        project = self._project()
        value = None if time is None else _number(time, "time", minimum=0.0)

        def apply() -> None:
            if value is None:
                project.events.pop(name, None)
                self.timeline.unfire(name)
            else:
                project.events[name] = value
                self.timeline.fire(name, value)
            self._place_followers()

        self._change(apply)

    def set_setting(self, path: list[str] | tuple[str, ...], value: Any) -> None:
        """Set the setting at ``path``, checked like ``manimx.json``; the
        preview renders with it from the next frame."""
        project = self._project()
        with self._lock:
            try:
                set_value(project.settings, tuple(path), value)
            except (SettingsError, StopIteration, AttributeError) as error:
                raise EditError(str(error) or f"no setting {'.'.join(path)}") from None
            settings = self.render_settings(project)
            if self.server is not None:
                self.server.replace(self.timeline, settings)
        self._schedule_save()

    def _change(self, apply: Callable[[], None]) -> None:
        with self._lock:
            if self.server is not None:
                self.server.change(apply)
            else:
                apply()
        self._schedule_save()

    def _place_followers(self) -> None:
        """Start every clip without its own start where the previous one
        ends (and the others at their start), after an edit moved things."""
        previous = None
        for entry, clip in zip(self._project().timeline, self.timeline.clips, strict=True):
            start = entry.start if entry.start is not None else 0.0 if previous is None else previous.end
            if start != NEVER and clip.start != start:
                clip.move(start)
            previous = clip

    def _project(self) -> Project:
        if self.project is None:
            raise EditError("the project did not load")
        return self.project

    # -- saving ---------------------------------------------------------------

    def _schedule_save(self) -> None:
        with self._lock:
            self._unsaved = True
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.save_delay, self.save)
            self._timer.daemon = True
            self._timer.start()

    def save(self) -> None:
        """Write manimx.json now if an edit is not saved yet (an autosave
        waiting is done with it)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self.project is None or not self._unsaved:
                return
            self._unsaved = False
            self.project.save()
            self.saved_mtime = _mtime(self.root / PROJECT_FILE)


def _module_files(project: Project) -> dict[str, Path | None]:
    """Each scene's own module file."""
    return {name: project.module_file(name) for name in project.scenes}


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _entry_json(entry: ClipEntry) -> dict[str, Any]:
    return {"scene": entry.scene, "start": entry.start, "layer": entry.layer, "cut": None if entry.cut is None else list(entry.cut), "speed": entry.speed}


def _clip_updates(changes: dict[str, Any]) -> dict[str, Any]:
    """``changes`` to a clip entry, checked."""
    unknown = set(changes) - set(CLIP_FIELDS)
    if unknown:
        raise EditError(f"clips have no {', '.join(sorted(unknown))}")
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
    return updates


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


def describe(error: BaseException, root: Path) -> str:
    """The error as the page shows it (see reload.py)."""
    from .reload import _describe

    return _describe(error, root)


__all__ = ["EditError", "ProjectEditor"]
