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
* **Every edit can be undone.** Edits are commands (commands.py), run by
  :meth:`ProjectEditor.run`, which keeps each one's inverse; the history
  is cleared when someone else's ``manimx.json`` is read, since its clips
  may be others.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..animation import PreparedScene, Timeline
from ..animation.timing import NEVER
from ..drivers.offline import RenderSettings
from ..project import PROJECT_FILE, ClipEntry, Project
from ..settings import SettingsError, entries, get, set_value
from .commands import (
    AddClip,
    ClipEdit,
    EditClips,
    EditError,
    InsertClips,
    RemoveClips,
    RemovedClip,
    SetEvent,
    SetSetting,
    UICommand,
    parse,
)

if TYPE_CHECKING:
    from .server import PreviewServer

HISTORY_LENGTH = 200
"""Edits kept to undo."""


@dataclass(frozen=True)
class _Step:
    command: UICommand
    """What undoes (or, on the redo stack, redoes) the step."""
    label: str
    """What the step did (the label of the command the user gave)."""
    gesture: str | None = None


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
        self._undo: list[_Step] = []
        self._redo: list[_Step] = []
        self._scene_versions: dict[str, int] = {}
        """How often each scene was prepared again for a changed file."""

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
            self._undo.clear()  # the clips may be others now: the steps would not fit
            self._redo.clear()
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

            for scene in {project.timeline[index].scene for index in fresh} | ({*project.scenes} if every else set()):
                self._scene_versions[scene] = self._scene_versions.get(scene, 0) + 1

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

    def run(self, command: UICommand, gesture: str | None = None) -> None:
        """Apply ``command`` live and keep its undo. Commands with the same
        ``gesture`` (the steps of one drag) are undone as one."""
        with self._lock:
            inverse = self._apply(command)
            last = self._undo[-1] if self._undo else None
            if gesture is None or last is None or last.gesture != gesture:
                self._undo.append(_Step(inverse, command.label, gesture))
                del self._undo[:-HISTORY_LENGTH]
            self._redo.clear()
        self._schedule_save()

    def undo(self) -> None:
        """Undo the last edit (or nothing)."""
        self._step(self._undo, self._redo)

    def redo(self) -> None:
        """Redo the last undone edit (or nothing)."""
        self._step(self._redo, self._undo)

    def history(self) -> dict[str, str | None]:
        """What undo and redo would do now, or None."""
        with self._lock:
            return {"undo": self._undo[-1].label if self._undo else None, "redo": self._redo[-1].label if self._redo else None}

    def _step(self, source: list[_Step], target: list[_Step]) -> None:
        with self._lock:
            if not source:
                return
            step = source[-1]
            inverse = self._apply(step.command)
            source.pop()
            target.append(_Step(inverse, step.label))
        self._schedule_save()

    def _apply(self, command: UICommand) -> UICommand:
        """Apply ``command``; returns the command that undoes it."""
        project = self._project()
        match command:
            case EditClips(edits):
                for edit in edits:
                    self._check_index(edit.index)
                inverse = EditClips(tuple(ClipEdit(edit.index, tuple((key, getattr(project.timeline[edit.index], key)) for key, _ in edit.updates)) for edit in edits))
                self._change(lambda: self._edit_clips(edits))
                return inverse
            case AddClip(scene, start, layer):
                if scene not in project.scenes:
                    raise EditError(f"the project has no scene named {scene!r}")
                prepared = self._prepare(scene)
                names = {clip.name for clip in self.timeline.clips}
                name, number = scene, 2
                while name in names:
                    name, number = f"{scene}-{number}", number + 1

                def add() -> None:
                    project.timeline.append(ClipEntry(scene, start=start, layer=layer))
                    self.timeline.add(prepared, start=start, name=name, layer=layer)
                    self._place_followers()

                self._change(add)
                return RemoveClips((len(project.timeline) - 1,))
            case RemoveClips(indices):
                for index in indices:
                    self._check_index(index)
                removed: list[RemovedClip] = []
                self._change(lambda: removed.extend(self._remove_clips(sorted(set(indices), reverse=True))))
                return InsertClips(tuple(reversed(removed)))
            case InsertClips(clips):
                fresh = {item.index: self._prepare(item.entry.scene) for item in clips if item.scene_version != self._scene_versions.get(item.entry.scene, 0)}
                self._change(lambda: self._insert_clips(clips, fresh))
                return RemoveClips(tuple(item.index for item in clips))
            case SetEvent(name, time):
                inverse = SetEvent(name, project.events.get(name))
                self._change(lambda: self._set_event(name, time))
                return inverse
            case SetSetting(path, value):
                try:
                    old = get(project.settings, path)
                    set_value(project.settings, path, value)
                except (SettingsError, StopIteration, AttributeError) as error:
                    raise EditError(str(error) or f"no setting {'.'.join(path)}") from None
                settings = self.render_settings(project)
                if self.server is not None:
                    self.server.replace(self.timeline, settings)
                return SetSetting(path, old)
        raise EditError(f"cannot apply {command!r}")

    def _edit_clips(self, edits: tuple[ClipEdit, ...]) -> None:
        project = self._project()
        for edit in edits:
            updates = dict(edit.updates)
            entry = project.timeline[edit.index] = dataclasses.replace(project.timeline[edit.index], **updates)
            clip = self.timeline.clips[edit.index]
            if "layer" in updates:
                clip.set_layer(entry.layer)
            if "cut" in updates:
                clip.cut(*(entry.cut or (None, None)))
            if "speed" in updates:
                clip.set_speed(entry.speed)
        self._place_followers()

    def _remove_clips(self, indices: list[int]) -> list[RemovedClip]:
        """Remove clips, from the back so the indices before stay."""
        project = self._project()
        removed = []
        for index in indices:
            follower = None
            if index + 1 < len(project.timeline) and project.timeline[index + 1].start is None:
                # The next clip followed this one: it keeps its place.
                follower = project.timeline[index + 1]
                project.timeline[index + 1] = dataclasses.replace(follower, start=self.timeline.clips[index + 1].start)
            entry, clip = project.timeline.pop(index), self.timeline.clips[index]
            self.timeline.remove(clip)
            removed.append(RemovedClip(index, entry, clip, follower, self._scene_versions.get(entry.scene, 0)))
        self._place_followers()
        return removed

    def _insert_clips(self, clips: tuple[RemovedClip, ...], fresh: dict[int, PreparedScene]) -> None:
        project = self._project()
        for item in clips:  # in index order: each lands where it was
            if item.index in fresh:
                item.clip.replace_scene(fresh[item.index])
            project.timeline.insert(item.index, item.entry)
            self.timeline.insert(item.index, item.clip)
            if item.follower is not None:
                project.timeline[item.index + 1] = item.follower
        self._place_followers()

    def _set_event(self, name: str, time: float | None) -> None:
        project = self._project()
        if time is None:
            project.events.pop(name, None)
            self.timeline.unfire(name)
        else:
            project.events[name] = time
            self.timeline.fire(name, time)
        self._place_followers()

    def _prepare(self, scene: str) -> PreparedScene:
        try:
            return self._project().prepare(scene)
        except Exception as error:
            raise EditError(describe(error, self.root)) from None

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self._project().timeline):
            raise EditError(f"there is no clip {index!r}")

    # The calls the tests and scripts use; the page sends commands (see commands.py).

    def edit_clip(self, index: int, changes: dict[str, Any]) -> None:
        """Change clip ``index``'s ``start`` (seconds, or None to follow
        the previous clip), ``layer``, ``cut`` (``[in, out]``, either None,
        or None for no cut) and ``speed``, live."""
        self.run(parse("/api/clip", {"index": index, **changes}))

    def edit_clips(self, edits: list[dict[str, Any]]) -> None:
        """Several clip edits as one: ``{"index": i, ...changes}`` each,
        checked before any applies."""
        self.run(parse("/api/clip", {"clips": edits}))

    def add_clip(self, scene: str, start: float, layer: int = 0) -> int:
        """Put a new clip of ``scene`` on the timeline at ``start`` on
        ``layer`` (prepared now); returns its index."""
        self.run(parse("/api/clip/add", {"scene": scene, "start": start, "layer": layer}))
        return len(self._project().timeline) - 1

    def remove_clips(self, indices: list[int]) -> None:
        """Take the clips ``indices`` off the timeline (a clip that followed
        a removed one keeps its place)."""
        self.run(parse("/api/clip/remove", {"indices": indices}))

    def set_event(self, name: str, time: float | None) -> None:
        """Set ``name``'s time (the clips waiting for it go on then), or
        with None remove it, live."""
        if not isinstance(name, str) or not name:
            raise EditError("an event needs a name")
        self.run(parse("/api/event", {"name": name, "time": time}))

    def set_setting(self, path: list[str] | tuple[str, ...], value: Any) -> None:
        """Set the setting at ``path``, checked like ``manimx.json``; the
        preview renders with it from the next frame."""
        self.run(parse("/api/setting", {"path": list(path), "value": value}))

    def scenes(self) -> list[dict[str, Any]]:
        """The project's scenes: name, module and class, and how many clips show each."""
        project = self._project()
        used: dict[str, int] = {}
        for entry in project.timeline:
            used[entry.scene] = used.get(entry.scene, 0) + 1
        return [{"name": name, "module": entry.module, "class": entry.cls, "clips": used.get(name, 0)} for name, entry in project.scenes.items()]

    def _change(self, apply: Callable[[], None]) -> None:
        with self._lock:
            if self.server is not None:
                self.server.change(apply)
            else:
                apply()

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


def describe(error: BaseException, root: Path) -> str:
    """The error as the page shows it (see reload.py)."""
    from .reload import _describe

    return _describe(error, root)


__all__ = ["EditError", "ProjectEditor"]
