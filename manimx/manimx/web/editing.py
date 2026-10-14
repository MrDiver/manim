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
* **Scripts are files of their own** (``scripts/narration.txt`` and its
  times in ``narration.json``), written at once when edited: they are
  not part of ``manimx.json``. Their words are events, so an edit moves
  the clips synced to them in the next frame, like timing an event by
  hand. Timing a recorded take (Whisper) takes seconds, so it runs
  without holding the editor; its result is one edit, undone like any
  other.
* **Every edit can be undone.** Edits are commands (commands.py), run by
  :meth:`ProjectEditor.run`, which keeps each one's inverse; the history
  is cleared when someone else's ``manimx.json`` is read, since its clips
  may be others.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..animation import PreparedScene, Timeline
from ..animation.timing import NEVER
from ..drivers.offline import RenderSettings
from ..project import PROJECT_FILE, AudioEntry, ClipEntry, Project, ProjectError
from ..settings import SettingsError, entries, get, set_value
from ..voice.script import SCRIPTS_FOLDER, Part, Script, Take
from ..voice.transcript import Word
from .commands import (
    AddAudio,
    AddClip,
    Batch,
    ClipEdit,
    DuplicateClips,
    EditAudio,
    EditClips,
    EditError,
    InsertClips,
    MoveWords,
    RemoveAudio,
    RemoveClips,
    RemovedClip,
    SetEvent,
    SetScript,
    SetScriptText,
    SetSetting,
    SplitClips,
    UICommand,
    parse,
)

if TYPE_CHECKING:
    from .server import PreviewServer

AUDIO_FOLDER = "assets/audio"
"""Where sound files dropped or recorded in the page are saved."""
AUDIO_SUFFIXES = {"wav", "mp3", "ogg", "oga", "opus", "flac", "m4a", "aac", "webm"}
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
            return
        if python:
            self._prepare_again(python)
        if any(path.parent == self.root / SCRIPTS_FOLDER for path in paths):
            self._read_scripts()

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
            case AddClip(scene, start, layer, cut, speed, index):
                if scene not in project.scenes:
                    raise EditError(f"the project has no scene named {scene!r}")
                position = len(project.timeline) if index is None else index
                if not 0 <= position <= len(project.timeline):
                    raise EditError(f"no place {index!r} for a clip")
                prepared = self._prepare(scene)
                names = {clip.name for clip in self.timeline.clips}
                name, number = scene, 2
                while name in names:
                    name, number = f"{scene}-{number}", number + 1

                def add() -> None:
                    project.timeline.insert(position, ClipEntry(scene, start=start, layer=layer, cut=cut, speed=speed))
                    clip = self.timeline.add(prepared, start=start, name=name, layer=layer)
                    self.timeline.remove(clip)
                    self.timeline.insert(position, clip)
                    if cut is not None:
                        clip.cut(*cut)
                    if speed != 1.0:
                        clip.set_speed(speed)
                    self._place_followers()

                self._change(add)
                return RemoveClips((position,), keep_followers=False)
            case RemoveClips(indices, keep_followers):
                for index in indices:
                    self._check_index(index)
                removed: list[RemovedClip] = []
                self._change(lambda: removed.extend(self._remove_clips(sorted(set(indices), reverse=True), keep_followers)))
                return InsertClips(tuple(reversed(removed)))
            case SplitClips(indices, time):
                # From the last clip: a new clip goes in right after each, moving only those after it.
                return self._apply(Batch(tuple(self._split(index, time) for index in sorted(set(indices), reverse=True)), command.label))
            case DuplicateClips(indices):
                # Each copy goes after its clip, in the first gap on its track long enough for it.
                copies: list[AddClip] = []
                for index in indices:
                    self._check_index(index)
                    entry, clip = project.timeline[index], self.timeline.clips[index]
                    if clip.end == NEVER:
                        raise EditError(f"{clip.name} waits for an event that never fires: it has no end to put a copy after")
                    length = clip.end - clip.start
                    taken = [(other.start, other.end) for other in self.timeline.clips if other.layer == entry.layer]
                    taken += [(copy.start, copy.start + length) for copy in copies if copy.layer == entry.layer]
                    copies.append(AddClip(entry.scene, _free_start(taken, clip.end, length), entry.layer, entry.cut, entry.speed))
                return self._apply(Batch(tuple(copies), command.label))
            case AddAudio(file, start, track, cut, volume, index, script, part):
                position = len(project.audio) if index is None else index
                if not 0 <= position <= len(project.audio):
                    raise EditError(f"no place {index!r} for an audio clip")
                if script is not None and any(e.script == script and e.part == part for e in project.audio):
                    raise EditError(f"part {part + 1} of {script} is on the timeline already: move it there")
                entry = AudioEntry(file, start, track, cut, volume, script, part)
                try:
                    clip = project.audio_clip(entry)
                except Exception as error:  # not a sound file, not in the project, ...
                    raise EditError(describe(error, self.root)) from None

                def add_audio() -> None:
                    project.audio.insert(position, entry)
                    self.timeline.audio_clips.insert(position, clip)
                    self._audio_changed()

                self._change(add_audio)
                return RemoveAudio((position,))
            case RemoveAudio(indices):
                unique = sorted(set(indices))
                for index in unique:
                    if not 0 <= index < len(project.audio):
                        raise EditError(f"there is no audio clip {index!r}")
                restore = tuple(AddAudio(e.file, e.start, e.track, e.cut, e.volume, i, e.script, e.part) for i, e in ((i, project.audio[i]) for i in unique))

                def remove_audio() -> None:
                    for index in reversed(unique):
                        del project.audio[index]
                        del self.timeline.audio_clips[index]
                    self._audio_changed()

                self._change(remove_audio)
                return Batch(restore, command.label)
            case EditAudio(edits):
                for edit in edits:
                    if not 0 <= edit.index < len(project.audio):
                        raise EditError(f"there is no audio clip {edit.index!r}")
                inverse = EditAudio(tuple(ClipEdit(edit.index, tuple((key, getattr(project.audio[edit.index], key)) for key, _ in edit.updates)) for edit in edits))
                self._change(lambda: self._edit_audio(edits))
                return inverse
            case Batch(commands, name):
                inverses: list[UICommand] = []
                try:
                    for part in commands:
                        inverses.append(self._apply(part))  # noqa: PERF401 - kept one by one to undo a failure
                except EditError:
                    for inverse in reversed(inverses):  # all or nothing
                        self._apply(inverse)
                    raise
                return Batch(tuple(reversed(inverses)), name)
            case InsertClips(clips):
                fresh = {item.index: self._prepare(item.entry.scene) for item in clips if item.scene_version != self._scene_versions.get(item.entry.scene, 0)}
                self._change(lambda: self._insert_clips(clips, fresh))
                return RemoveClips(tuple(item.index for item in clips))
            case SetEvent(name, time):
                inverse = SetEvent(name, project.events.get(name))
                self._change(lambda: self._set_event(name, time))
                return inverse
            case SetScript(name, script):
                old = self._script(name)
                if script is None:
                    self._check_unplaced(name, 0)
                else:
                    self._check_unplaced(name, len(script.parts))
                self._change(lambda: self._set_script(name, script))
                return SetScript(name, old)
            case SetScriptText(name, text):
                old = self._script(name)
                new = Script.new(name, text, old)
                self._check_unplaced(name, len(new.parts))
                self._change(lambda: self._set_script(name, new))
                return SetScript(name, old)
            case MoveWords(name, part, words):
                old = self._script(name)
                if old is None or not 0 <= part < len(old.parts):
                    raise EditError(f"there is no part {part + 1} of a script {name}")
                moved = list(old.parts[part].words)
                for index, start, end in words:
                    if not 0 <= index < len(moved):
                        raise EditError(f"part {part + 1} of {name} has no word {index}")
                    moved[index] = Word(moved[index].text, start, end)
                new = old.with_part(part, Part(old.parts[part].text, tuple(moved), old.parts[part].take))
                self._change(lambda: self._set_script(name, new))
                return SetScript(name, old)
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
            if "cues" in updates:
                clip.bind_cues(entry.cues)
        self._place_followers()

    def _split(self, index: int, time: float) -> UICommand:
        """The commands that split clip ``index`` at global ``time``."""
        self._check_index(index)
        entry, clip = self._project().timeline[index], self.timeline.clips[index]
        compiled = clip.compiled_time(time)
        cut_in, cut_out = entry.cut or (None, None)
        playing = None if compiled is None else clip.scene.playing_time_at(compiled)
        first, last = cut_in or 0.0, clip.scene.playing_time if cut_out is None else cut_out
        if playing is None or not first + 1e-6 < playing < last - 1e-6:
            raise EditError(f"the playhead is not inside {clip.name}")
        return Batch(
            (
                EditClips((ClipEdit(index, (("cut", (cut_in, playing)),)),)),
                AddClip(entry.scene, time, entry.layer, (playing, cut_out), entry.speed, index + 1),
            ),
            "split a clip",
        )

    def _edit_audio(self, edits: tuple[ClipEdit, ...]) -> None:
        project = self._project()
        for edit in edits:
            entry = project.audio[edit.index] = dataclasses.replace(project.audio[edit.index], **dict(edit.updates))
            clip = self.timeline.audio_clips[edit.index]
            clip.start, clip.track, clip.volume = entry.start, entry.track, entry.volume
            clip.cut_in, clip.cut_out = entry.cut or (None, None)
        self._audio_changed()

    def _audio_changed(self) -> None:
        """The audio clips moved or their words changed: so did the events of their words."""
        self.timeline.audio_changed()
        self._place_followers()

    # -- scripts ----------------------------------------------------------------------

    def scripts(self) -> list[dict[str, Any]]:
        """Every script as the page shows it: text, and per part its title,
        words, take and where it is on the timeline (an audio clip index)."""
        project = self._project()
        result = []
        for name in project.script_names():
            try:
                script = project.script(name)
            except ProjectError as error:
                result.append({"name": name, "error": str(error)})
                continue
            placed = {entry.part: index for index, entry in enumerate(project.audio) if entry.script == name}
            result.append(
                {
                    "name": name,
                    "text": script.text,
                    "parts": [
                        {
                            "title": part.title,
                            "words": [{"text": w.text, "start": w.start, "end": w.end} for w in part.words],
                            "take": None if part.take is None else {"file": part.take.file, "offset": part.take.offset},
                            "length": part.planned_length,
                            "clip": placed.get(index),
                        }
                        for index, part in enumerate(script.parts)
                    ],
                }
            )
        return result

    def record_take(self, name: str, part: int, file: str, offset: float) -> str | None:
        """Make the sound file ``file`` (inside the project) part ``part``
        of script ``name``'s take, the part starting ``offset`` seconds in
        (after the countdown), and time its words with Whisper (the
        project's ``voice`` settings); one edit, undone like others. Runs
        for seconds, without holding the editor. Returns why the words
        could not be timed (they keep their planned times), or None."""
        from ..voice.alignment import time_take

        project = self._project()
        script = self._script(name)
        if script is None or not 0 <= part < len(script.parts):
            raise EditError(f"there is no part {part + 1} of a script {name}")
        try:
            sound = project.sound(file)
        except Exception as error:
            raise EditError(describe(error, self.root)) from None
        old = script.parts[part]
        problem = None
        try:
            voice = project.settings.voice
            words = time_take(sound, old.text, offset, voice.model, voice.language or None)
        except Exception as error:  # no faster-whisper, a model that does not download, ...
            words, problem = old.words, f"the words keep their planned times: {error}"
        self.run(SetScript(name, script.with_part(part, Part(old.text, words, Take(file, offset)))))
        return problem

    def _script(self, name: str) -> Script | None:
        project = self._project()
        if name not in project.script_names():
            return None
        try:
            return project.script(name)
        except Exception as error:  # a broken times file
            raise EditError(describe(error, self.root)) from None

    def _check_unplaced(self, name: str, parts: int) -> None:
        """Refuse an edit that would leave a placed part without its script."""
        for entry in self._project().audio:
            if entry.script == name and entry.part >= parts:
                raise EditError(f"part {entry.part + 1} of {name} is on the timeline: remove it there first")

    def _set_script(self, name: str, script: Script | None) -> None:
        project = self._project()
        if script is None:
            project.delete_script(name)
        else:
            project.save_script(script)
        self._update_script_clips(name)

    def _update_script_clips(self, name: str | None = None) -> None:
        """The audio clips of script parts (of ``name``, or all) made again from their scripts."""
        project = self._project()
        for index, entry in enumerate(project.audio):
            if entry.script is not None and (name is None or entry.script == name):
                self.timeline.audio_clips[index] = project.audio_clip(entry)
        self._audio_changed()

    def _read_scripts(self) -> None:
        """Script files changed on disk (not by this editor): read them again."""
        with self._lock:
            if self.project is None:
                return
            try:
                clips = [self.project.audio_clip(entry) if entry.script is not None else None for entry in self.project.audio]
            except Exception as error:
                self._report(error)
                return

            def update() -> None:
                for index, clip in enumerate(clips):
                    if clip is not None:
                        self.timeline.audio_clips[index] = clip
                self._audio_changed()

            self._change(update)

    # -- sound files ------------------------------------------------------------

    def save_asset(self, name: str, data: bytes) -> str:
        """Save an uploaded sound file (dropped or recorded in the page) to
        assets/audio/ under a free name; returns its path in the project."""
        stem, dot, suffix = Path(name).name.rpartition(".")
        if not dot or suffix.lower() not in AUDIO_SUFFIXES:
            raise EditError(f"{name} is not a sound file manimx reads ({', '.join(sorted(AUDIO_SUFFIXES))})")
        stem = re.sub(r"[^\w.-]+", "-", stem).strip("-.") or "sound"
        folder = self.root / AUDIO_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        path, number = folder / f"{stem}.{suffix.lower()}", 2
        while path.exists():
            path, number = folder / f"{stem}-{number}.{suffix.lower()}", number + 1
        path.write_bytes(data)
        return path.relative_to(self.root).as_posix()

    def audio_clips(self) -> list[dict[str, Any]]:
        """Each audio clip as the page draws it."""
        project = self.project
        if project is None:
            return []
        return [
            {
                "name": clip.name,
                "file": entry.file if entry.script is None else _take_file(clip),
                "start": clip.start,
                "end": clip.end,
                "track": entry.track,
                "cut": None if entry.cut is None else list(entry.cut),
                "volume": entry.volume,
                "length": clip.source_length,
                "script": entry.script,
                "part": entry.part,
                "offset": clip.offset,
                "recorded": clip.sound is not None,
            }
            for entry, clip in zip(project.audio, self.timeline.audio_clips, strict=True)
        ]

    def waveform(self, index: int, start: float, end: float, buckets: int) -> list[list[float]]:
        """Audio clip ``index``'s loudest and quietest sample in each of
        ``buckets`` slices of the file's seconds ``[start, end)``, for drawing."""
        clips = self.timeline.audio_clips
        if not 0 <= index < len(clips):
            raise EditError(f"there is no audio clip {index!r}")
        sound = clips[index].sound
        if sound is None:  # a script part not recorded yet
            return [[0.0, 0.0]] * max(1, min(int(buckets), 4000))
        samples = sound.samples.mean(axis=1)
        first = max(0, round(start * sound.sample_rate))
        last = min(len(samples), max(first, round(end * sound.sample_rate)))
        buckets = max(1, min(int(buckets), 4000))
        edges = np.linspace(first, last, buckets + 1).astype(int)
        return [
            [round(float(samples[a:b].min()), 3), round(float(samples[a:b].max()), 3)] if b > a else [0.0, 0.0]
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]

    def _remove_clips(self, indices: list[int], keep_followers: bool = True) -> list[RemovedClip]:
        """Remove clips, from the back so the indices before stay."""
        project = self._project()
        removed = []
        for index in indices:
            follower = None
            if keep_followers and index + 1 < len(project.timeline) and project.timeline[index + 1].start is None:
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


def _take_file(clip: Any) -> str | None:
    """The file of a script part's take (None before it is recorded)."""
    take = clip.script.parts[clip.part].take if clip.script is not None else None
    return None if take is None else take.file


def _free_start(taken: list[tuple[float, float]], after: float, length: float) -> float:
    """The first time from ``after`` on where ``length`` seconds overlap none of ``taken``."""
    start = after
    while True:
        blocking = [end for begin, end in taken if begin < start + length - 1e-9 and end > start + 1e-9]
        if not blocking:
            return start
        start = max(blocking)


def _module_files(project: Project) -> dict[str, Path | None]:
    """Each scene's own module file."""
    return {name: project.module_file(name) for name in project.scenes}


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _entry_json(entry: ClipEntry) -> dict[str, Any]:
    cut = None if entry.cut is None else list(entry.cut)
    return {"scene": entry.scene, "start": entry.start, "layer": entry.layer, "cut": cut, "speed": entry.speed, "cues": dict(entry.cues)}


def describe(error: BaseException, root: Path) -> str:
    """The error as the page shows it (see reload.py)."""
    from .reload import _describe

    return _describe(error, root)


__all__ = ["EditError", "ProjectEditor"]
