"""Projects: a folder of scenes plus ``manimx.json``, the project's state.

    my-video/
        manimx.json        render settings, the scene registry, the timeline
        pyproject.toml     depends on manimx
        scenes/            one module per scene (``manimx generate scene``)
        assets/            sounds, images
        renders/           output

``manimx.json`` is to a manimx project what ``angular.json`` is to an
Angular workspace: the CLI keeps it up to date (``generate`` registers a
scene and appends it to the timeline), and code and the coming web UI
edit the same :class:`Project` object and save it back. Section 8.3 of
the design doc: the object state is the source of truth, and the JSON
file is its snapshot. Scenes stay code; the project only references
them by module and class.

The timeline lists clips in order. A clip without a ``start`` begins
when the previous clip ends, so generated scenes simply play one after
another; an explicit ``start`` (seconds) places a clip anywhere, e.g. in
parallel with another one on a higher ``layer``. Below the clips, the
timeline's ``audio`` lists sound files on audio tracks (voice-overs,
music, recordings made in the preview), mixed into the video's sound.

What is said is written in scripts (``scripts/narration.txt``, see
manimx/voice/script.py), split into parts by ``---`` lines; a part goes
onto an audio track as ``{"script": "narration", "part": 0, "start": 3}``
and plays its recorded take, or nothing before it is recorded. Its words
are events scenes wait for or sync their cues to, and a clip's ``cues``
bind a scene's cue to another event than the one of its own name
(``{"square": "narration.middle.square"}``).
"""

from __future__ import annotations

import difflib
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .animation.audio import Sound
from .animation.scene import PreparedScene, Scene
from .animation.timing import NEVER, AudioClip, Timeline, Track
from .settings import (
    OutputSettings,
    PerformanceSettings,
    Settings,
    SettingsError,
    UnknownSetting,
)
from .settings import from_json as settings_from_json
from .settings import schema as settings_schema
from .settings import to_json as settings_to_json
from .voice.script import SCRIPTS_FOLDER, Script, script_names

PROJECT_FILE = "manimx.json"
SCHEMA_FILE = ".manimx/project.schema.json"
FORMAT_VERSION = 3
"""Version of the manimx.json format; bumped (with a migration) on breaking changes."""
SECTIONS = [item.name for item in fields(Settings)]
"""The settings sections at the top level of manimx.json."""


class ProjectError(Exception):
    """A project that cannot be found, read or built."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class SceneError(ProjectError):
    """Scene code raised while the scene was being prepared (the cause)."""

    def __init__(self, scene: str, stage: str = "being prepared") -> None:
        super().__init__(f"{scene} failed while it was {stage}")
        self.scene = scene


@dataclass
class SceneEntry:
    module: str
    """Module path inside the project, e.g. ``scenes.intro``."""
    cls: str
    """Scene class name."""


@dataclass
class ClipEntry:
    scene: str
    start: float | None = None
    """Seconds; None: when the previous clip ends."""
    name: str | None = None
    layer: int = 0
    cut: tuple[float | None, float | None] | None = None
    """In and out points in the scene's playing time."""
    speed: float = 1.0
    """How fast the clip plays: 2 is twice as fast, 0.5 slow motion."""
    cues: dict[str, str] = field(default_factory=dict)
    """The event each of the scene's cues syncs to (default: the event of its own name)."""


@dataclass
class AudioEntry:
    """A sound on the timeline's audio tracks: a sound file (music, a
    recording) or a part of a script (its take, once recorded)."""

    file: str | None
    """Path inside the project, e.g. assets/audio/music.wav (None for a script part)."""
    start: float = 0.0
    track: int = 0
    """The audio track (a script part: the script track) it is on."""
    cut: tuple[float | None, float | None] | None = None
    """In and out points in the file's seconds (a script part's: from the part's start)."""
    volume: float = 1.0
    script: str | None = None
    """The script whose part this is."""
    part: int = 0
    fade_in: float = 0.0
    """Seconds over which it rises from silence at its start."""
    fade_out: float = 0.0
    """Seconds over which it falls to silence at its end."""


@dataclass(eq=False)
class Project:
    root: Path
    name: str
    settings: Settings = field(default_factory=Settings)
    scenes: dict[str, SceneEntry] = field(default_factory=dict)
    timeline: list[ClipEntry] = field(default_factory=list)
    events: dict[str, float] = field(default_factory=dict)
    """Event times set by hand (a click; the words of scripts are events of their own)."""
    audio: list[AudioEntry] = field(default_factory=list)
    scene_tracks: dict[int, Track] = field(default_factory=dict)
    """Names, visibility, mute, solo and volume of the scene tracks (layers) that have them."""
    audio_tracks: dict[int, Track] = field(default_factory=dict)
    """The same of the audio tracks."""
    script_tracks: dict[int, Track] = field(default_factory=dict)
    """The same of the script tracks."""
    _sounds: dict[Path, tuple[int, Sound]] = field(default_factory=dict, init=False, repr=False)
    """Sound files read, with their modification time: read again when changed."""
    _scripts: dict[str, tuple[tuple[int | None, int | None], Script]] = field(default_factory=dict, init=False, repr=False)
    """Scripts read, with the modification times of their two files."""

    @property
    def output(self) -> OutputSettings:
        return self.settings.output

    @property
    def performance(self) -> PerformanceSettings:
        return self.settings.performance

    # -- files ------------------------------------------------------------------

    @classmethod
    def find(cls, start: str | Path = ".") -> Project:
        """The project containing ``start`` (searching parent folders, like
        git and the Angular CLI)."""
        directory = Path(start).resolve()
        for candidate in (directory, *directory.parents):
            if (candidate / PROJECT_FILE).is_file():
                return cls.load(candidate)
        raise ProjectError(f"no {PROJECT_FILE} in {directory} or any parent folder", hint="create a project with `manimx new`")

    @classmethod
    def load(cls, root: str | Path) -> Project:
        root = Path(root).resolve()
        try:
            data = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProjectError(f"cannot read {root / PROJECT_FILE}: {error}") from error
        while isinstance(data, dict) and data.get("format") in _MIGRATIONS:
            data = _MIGRATIONS[data["format"]](data)
        if not isinstance(data, dict) or data.get("format") != FORMAT_VERSION:
            found = data.get("format") if isinstance(data, dict) else None
            raise ProjectError(f"{PROJECT_FILE} has format {found}, this manimx reads format {FORMAT_VERSION}")
        _check_keys(data, ["$schema", "format", "name", *SECTIONS, "scenes", "timeline"], "")
        if not isinstance(data.get("name"), str):
            raise ProjectError(f'invalid {PROJECT_FILE}: "name" must be a string')
        timeline = data.get("timeline", {})
        _check_keys(timeline, ["clips", "events", "audio", "tracks"], "timeline")
        try:
            settings = settings_from_json({key: data[key] for key in SECTIONS if key in data})
        except UnknownSetting as error:
            hint = f'did you mean "{error.close}"?' if error.close else f"known here: {', '.join(error.known)}"
            raise ProjectError(f"{error} in {PROJECT_FILE}", hint=hint) from None
        except SettingsError as error:
            raise ProjectError(str(error)) from None
        try:
            scenes = {}
            for name, entry in data.get("scenes", {}).items():
                _check_keys(entry, ["module", "class"], f"scenes.{name}")
                scenes[name] = SceneEntry(entry["module"], entry.get("class", name))
            clips = []
            for index, entry in enumerate(timeline.get("clips", [])):
                _check_keys(entry, ["scene", "start", "name", "layer", "cut", "speed", "cues"], f"timeline.clips[{index}]")
                clips.append(_clip_entry(entry))
            events = {name: float(time) for name, time in timeline.get("events", {}).items()}
            audio = []
            for index, entry in enumerate(timeline.get("audio", [])):
                _check_keys(entry, ["file", "script", "part", "start", "track", "cut", "volume", "fade_in", "fade_out"], f"timeline.audio[{index}]")
                audio.append(_audio_entry(entry))
            tracks = timeline.get("tracks", {})
            _check_keys(tracks, ["scenes", "audio", "scripts"], "timeline.tracks")
            scene_tracks = {int(index): _track(data, f"timeline.tracks.scenes.{index}") for index, data in tracks.get("scenes", {}).items()}
            audio_tracks = {int(index): _track(data, f"timeline.tracks.audio.{index}") for index, data in tracks.get("audio", {}).items()}
            script_tracks = {int(index): _track(data, f"timeline.tracks.scripts.{index}") for index, data in tracks.get("scripts", {}).items()}
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ProjectError(f"invalid {PROJECT_FILE}: {error}") from error
        return cls(
            root=root, name=data["name"], settings=settings, scenes=scenes, timeline=clips, events=events, audio=audio,
            scene_tracks=scene_tracks, audio_tracks=audio_tracks, script_tracks=script_tracks,
        )  # fmt: skip

    def to_json(self) -> dict[str, Any]:
        return {
            "$schema": f"./{SCHEMA_FILE}",
            "format": FORMAT_VERSION,
            "name": self.name,
            **settings_to_json(self.settings),
            "scenes": {name: {"module": entry.module, "class": entry.cls} for name, entry in self.scenes.items()},
            "timeline": {
                "clips": [_clip_json(entry) for entry in self.timeline],
                "events": dict(self.events),
                **({"audio": [_audio_json(entry) for entry in self.audio]} if self.audio else {}),
                **({"tracks": tracks} if (tracks := _tracks_json(self.scene_tracks, self.audio_tracks, self.script_tracks)) else {}),
            },
        }

    def save(self) -> None:
        """Write manimx.json, and the schema editors use to check it."""
        text = json.dumps(self.to_json(), indent=2) + "\n"
        (self.root / PROJECT_FILE).write_text(text, encoding="utf-8")
        schema = self.root / SCHEMA_FILE
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(json.dumps(project_schema(), indent=2) + "\n", encoding="utf-8")

    # -- rendering --------------------------------------------------------------

    def render_settings(self, settings: Settings | None = None) -> Any:
        """Driver settings from the project's settings (or replacements).
        Each scene renders through its own camera (``Scene.camera``)."""
        from .drivers import RenderSettings

        settings = settings or self.settings
        output = settings.output
        return RenderSettings(
            settings.render_config(),
            fps=output.fps,
            effects=settings.effects_chain(),
            in_flight=settings.performance.frames_in_flight,
            sample_rate=output.sample_rate,
            audio=output.audio,
        )

    # -- editing ----------------------------------------------------------------

    def add_scene(self, name: str, module: str, cls: str | None = None, on_timeline: bool = True) -> None:
        if name in self.scenes:
            raise ProjectError(f"the project already has a scene named {name!r}", hint="pick another name")
        self.scenes[name] = SceneEntry(module, cls or name)
        if on_timeline:
            self.timeline.append(ClipEntry(name))

    def remove_scene(self, name: str) -> SceneEntry:
        """Unregister a scene and take its clips off the timeline."""
        self._check_scene(name)
        self.timeline = [clip for clip in self.timeline if clip.scene != name]
        return self.scenes.pop(name)

    # -- building ---------------------------------------------------------------

    def scene_class(self, name: str) -> type[Scene]:
        entry = self._check_scene(name)
        try:
            module = self._import(entry.module)
        except ProjectError:
            raise
        except Exception as error:  # a syntax error or a failing import in the scene's module
            raise SceneError(name, "imported") from error
        scene = getattr(module, entry.cls, None)
        if not (isinstance(scene, type) and issubclass(scene, Scene)):
            raise ProjectError(f"{entry.module}.{entry.cls} is not a Scene class")
        return scene

    def prepare(self, name: str) -> PreparedScene:
        scene = self.scene_class(name)
        try:
            return scene().prepare()
        except Exception as error:
            raise SceneError(name) from error

    def module_file(self, name: str) -> Path | None:
        """The file of scene ``name``'s module, if it is the project's own."""
        entry = self._check_scene(name)
        base = self.root / Path(*entry.module.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _check_scene(self, name: str) -> SceneEntry:
        entry = self.scenes.get(name)
        if entry is None:
            close = difflib.get_close_matches(name, self.scenes, n=1)
            hint = f"did you mean {close[0]!r}?" if close else f"create it with `manimx generate scene {name}`"
            raise ProjectError(f"the project has no scene named {name!r}", hint=hint)
        return entry

    def build_timeline(
        self,
        progress: Callable[[ClipEntry], None] | None = None,
        prepare: Callable[[int, ClipEntry], PreparedScene] | None = None,
    ) -> Timeline:
        """The live timeline: every clip of ``manimx.json``, prepared and
        placed. ``progress`` is called before each clip is prepared;
        ``prepare(index, entry)`` gives clip ``index``'s prepared scene
        (default: prepare it now; the preview reuses scenes whose files did
        not change)."""
        # The audio first: its words are events, which decide where clips end.
        timeline = Timeline(
            audio_clips=[self.audio_clip(entry) for entry in self.audio], events=dict(self.events),
            scene_tracks=dict(self.scene_tracks), audio_tracks=dict(self.audio_tracks), script_tracks=dict(self.script_tracks),
        )  # fmt: skip
        previous_end = 0.0
        used: set[str] = set()
        previous_name = ""
        for index, entry in enumerate(self.timeline):
            if progress is not None:
                progress(entry)
            if entry.start is None and previous_end == NEVER:
                raise ProjectError(
                    f"clip {previous_name!r} waits for an event that never fires, so the next clip has no start",
                    hint="give the next clip an explicit start, or set the event's time under \"events\" in manimx.json",
                )
            start = previous_end if entry.start is None else entry.start
            scene = self.prepare(entry.scene) if prepare is None else prepare(index, entry)
            clip = timeline.add(scene, start=start, name=_unique(entry.name or entry.scene, used), layer=entry.layer)
            if entry.cut is not None:
                clip.cut(*entry.cut)
            if entry.speed != 1.0:
                clip.set_speed(entry.speed)
            if entry.cues:
                clip.bind_cues(entry.cues)
            previous_end, previous_name = clip.end, clip.name
        return timeline

    def sound(self, file: str) -> Sound:
        """The sound file ``file`` (inside the project), read once while it is unchanged."""
        path = (self.root / file).resolve()
        if not path.is_relative_to(self.root):
            raise ProjectError(f"{file} is outside the project")
        try:
            mtime = path.stat().st_mtime_ns
        except OSError as error:
            raise ProjectError(f"cannot read the sound {file}: {error}") from None
        known = self._sounds.get(path)
        if known is None or known[0] != mtime:
            known = self._sounds[path] = (mtime, Sound.load(path))
        return known[1]

    def audio_clip(self, entry: AudioEntry) -> AudioClip:
        """The timeline's audio clip for ``entry``: its sound file, or its script part."""
        cut_in, cut_out = entry.cut or (None, None)
        if entry.script is None:
            if entry.file is None:
                raise ProjectError("an audio clip needs a file or a script part")
            return AudioClip(
                self.sound(entry.file), entry.start, entry.track, cut_in, cut_out, entry.volume, Path(entry.file).stem,
                fade_in=entry.fade_in, fade_out=entry.fade_out,
            )  # fmt: skip
        script = self.script(entry.script)
        if not 0 <= entry.part < len(script.parts):
            raise ProjectError(f"the script {entry.script!r} has no part {entry.part + 1}")
        part = script.parts[entry.part]
        sound, offset = (None, 0.0) if part.take is None else (self.sound(part.take.file), part.take.offset)
        # The entry's cut counts from the part's start, the clip's from the sound's.
        cut_in = offset + (cut_in or 0.0)
        cut_out = None if cut_out is None else offset + cut_out
        return AudioClip(
            sound, entry.start, entry.track, cut_in, cut_out, entry.volume, f"{script.name} {entry.part + 1}",
            script=script, part=entry.part, offset=offset, planned=part.planned_length, fade_in=entry.fade_in, fade_out=entry.fade_out,
        )  # fmt: skip

    @property
    def scripts_folder(self) -> Path:
        return self.root / SCRIPTS_FOLDER

    def script_names(self) -> list[str]:
        """The project's scripts, by name."""
        return script_names(self.scripts_folder)

    def script(self, name: str) -> Script:
        """The script ``name`` (scripts/<name>.txt), read once while its files are unchanged."""
        folder = self.scripts_folder
        stamps = tuple(_mtime(folder / f"{name}.{suffix}") for suffix in ("txt", "json"))
        if stamps[0] is None:
            raise ProjectError(f"the project has no script named {name!r}", hint=f"write it in {SCRIPTS_FOLDER}/{name}.txt")
        known = self._scripts.get(name)
        if known is None or known[0] != stamps:
            try:
                known = self._scripts[name] = (stamps, Script.load(folder, name))  # type: ignore[assignment]
            except (OSError, ValueError) as error:
                raise ProjectError(str(error)) from None
        return known[1]

    def save_script(self, script: Script) -> None:
        """Write ``script``'s text and times."""
        script.save(self.scripts_folder)
        folder = self.scripts_folder
        self._scripts[script.name] = (tuple(_mtime(folder / f"{script.name}.{suffix}") for suffix in ("txt", "json")), script)  # type: ignore[assignment]

    def delete_script(self, name: str) -> None:
        """Remove the script ``name``'s files (its takes stay in assets/audio/)."""
        for suffix in ("txt", "json"):
            (self.scripts_folder / f"{name}.{suffix}").unlink(missing_ok=True)
        self._scripts.pop(name, None)

    def source(self, scene: str | None = None) -> ProjectSource:
        """What the render driver renders: the timeline, or one scene."""
        if scene is not None:
            self.scene_class(scene)  # fail early, before worker processes start
        return ProjectSource(str(self.root), scene)

    def unload_modules(self) -> list[str]:
        """Forget every imported module that lives in the project's folder,
        so the next build imports the scenes as they are on disk now (after
        an edit). Returns the names of the modules forgotten.
        """
        mine = [name for name, module in list(sys.modules.items()) if self._owns(getattr(module, "__file__", None))]
        for name in mine:
            del sys.modules[name]
        importlib.invalidate_caches()
        return mine

    def _owns(self, location: str | None) -> bool:
        """Whether a module file is the project's own code: inside its
        folder, but not in a hidden folder (a .venv) or site-packages."""
        if not location:
            return False
        path = Path(location).resolve()
        if not path.is_relative_to(self.root):
            return False
        parts = path.relative_to(self.root).parts[:-1]
        return not any(part.startswith(".") or part == "site-packages" for part in parts)

    def _import(self, module: str) -> Any:
        root = str(self.root)
        top = module.partition(".")[0]
        loaded = sys.modules.get(top)
        location = getattr(loaded, "__file__", None)
        if loaded is not None and not (location and Path(location).resolve().is_relative_to(self.root)):
            # Another project's package of the same name (e.g. "scenes").
            for name in [name for name in sys.modules if name == top or name.startswith(top + ".")]:
                del sys.modules[name]
        # First on the path, also when it is there already: a project loaded
        # in between put its own folder in front.
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
        importlib.invalidate_caches()
        try:
            return importlib.import_module(module)
        except ImportError as error:
            raise ProjectError(f"cannot import {module}: {error}", hint=f"is it a module inside {root}?") from error


@dataclass(frozen=True)
class ProjectSource:
    """A picklable render source: worker processes load the project again."""

    root: str
    scene: str | None = None

    def __call__(self) -> Timeline | PreparedScene:
        project = Project.load(self.root)
        return project.build_timeline() if self.scene is None else project.prepare(self.scene)


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _unique(name: str, used: set[str]) -> str:
    candidate, number = name, 2
    while candidate in used:
        candidate, number = f"{name}-{number}", number + 1
    used.add(candidate)
    return candidate


def _check_keys(data: Any, allowed: list[str], where: str) -> None:
    """Reject unknown keys: a typo must not silently fall back to a default."""
    if not isinstance(data, dict):
        raise ProjectError(f'invalid {PROJECT_FILE}: "{where}" must be an object')
    for key in data:
        if key not in allowed:
            close = difflib.get_close_matches(key, allowed, n=1)
            place = f"{where}.{key}" if where else key
            raise ProjectError(
                f'unknown setting "{place}" in {PROJECT_FILE}',
                hint=f'did you mean "{close[0]}"?' if close else f"known here: {', '.join(allowed)}",
            )


def _from_format_1(data: dict[str, Any]) -> dict[str, Any]:
    """Format 1 had the motion blur samples in "output" and a "camera"
    section (cameras now belong to scenes)."""
    data = {key: value for key, value in data.items() if key != "camera"}
    output = data.get("output")
    if isinstance(output, dict) and isinstance(output.get("motion_blur"), int):
        output = dict(output)
        samples = output.pop("motion_blur")
        data["output"] = output
        data["effects"] = {"motion_blur": {"enabled": samples > 1, "samples": max(samples, 1)}}
    return {**data, "format": 2}


def _from_format_2(data: dict[str, Any]) -> dict[str, Any]:
    """Format 2 had the renders per frame in "effects.motion_blur.samples",
    used only with motion blur on; they are "output.samples" now."""
    effects = data.get("effects")
    blur = effects.get("motion_blur") if isinstance(effects, dict) else None
    if isinstance(blur, dict) and "samples" in blur:
        blur = dict(blur)
        samples = blur.pop("samples")
        data = {**data, "effects": {**effects, "motion_blur": blur}}
        if blur.get("enabled") and isinstance(samples, int):
            output = data.get("output")
            data["output"] = {**(output if isinstance(output, dict) else {}), "samples": samples}
    return {**data, "format": 3}


_MIGRATIONS = {1: _from_format_1, 2: _from_format_2}
"""Format version -> function turning it into the next one."""


def project_schema() -> dict[str, Any]:
    """JSON Schema of manimx.json, generated from the settings dataclasses,
    so editors complete and check the file (like angular.json)."""
    number_or_null = {"type": ["number", "null"]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "manimx project",
        "type": "object",
        "required": ["format", "name"],
        "additionalProperties": False,
        "properties": {
            "$schema": {"type": "string"},
            "format": {"const": FORMAT_VERSION, "description": "Version of this file format."},
            "name": {"type": "string", "description": "Project name; also the default video name."},
            **settings_schema()["properties"],
            "scenes": {
                "type": "object",
                "description": "Scene registry: name -> module and class (manimx generate scene adds entries).",
                "additionalProperties": {
                    "type": "object",
                    "required": ["module"],
                    "additionalProperties": False,
                    "properties": {
                        "module": {"type": "string", "description": "Module inside the project, e.g. scenes.intro."},
                        "class": {"type": "string", "description": "Scene class (default: the scene's name)."},
                    },
                },
            },
            "timeline": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "clips": {
                        "type": "array",
                        "description": "Clips in order; a clip without start begins when the previous one ends.",
                        "items": {
                            "type": "object",
                            "required": ["scene"],
                            "additionalProperties": False,
                            "properties": {
                                "scene": {"type": "string", "description": "A name from scenes."},
                                "start": {"type": "number", "description": "Start in seconds (default: after the previous clip)."},
                                "name": {"type": "string", "description": "Clip name (default: the scene's name)."},
                                "layer": {"type": "integer", "description": "Higher layers are drawn on top."},
                                "cut": {
                                    "type": "array",
                                    "items": number_or_null,
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": "In and out points in the scene's playing time.",
                                },
                                "speed": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "description": "How fast the clip plays: 2 is twice as fast, 0.5 slow motion (default 1).",
                                },
                                "cues": {
                                    "type": "object",
                                    "description": "The event each cue of the scene syncs to, e.g. a spoken word (default: the event of the cue's own name).",
                                    "additionalProperties": {"type": "string"},
                                },
                            },
                        },
                    },
                    "events": {
                        "type": "object",
                        "description": "Times (seconds) of events scenes wait for with wait_for() (the words of scripts are events too).",
                        "additionalProperties": {"type": "number"},
                    },
                    "tracks": {
                        "type": "object",
                        "description": "Names, visibility, mute, solo and volume of tracks, by number (only what differs from the default).",
                        "additionalProperties": False,
                        "properties": {
                            key: {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string"},
                                        "hidden": {"type": "boolean", "description": "Scene tracks: not drawn."},
                                        "locked": {"type": "boolean", "description": "Not edited in the preview."},
                                        "muted": {"type": "boolean"},
                                        "solo": {"type": "boolean", "description": "When tracks solo, only those are heard."},
                                        "volume": {"type": "number", "minimum": 0},
                                    },
                                },
                            }
                            for key in ("scenes", "audio", "scripts")
                        },
                    },
                    "audio": {
                        "type": "array",
                        "description": "Sounds on the audio tracks: sound files (music, recordings) and parts of scripts.",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "file": {"type": "string", "description": "Path inside the project, e.g. assets/audio/music.wav."},
                                "script": {"type": "string", "description": "A script (scripts/<name>.txt) whose part this is, instead of a file."},
                                "part": {"type": "integer", "minimum": 0, "description": "Which part of the script (0 is the first)."},
                                "start": {"type": "number", "description": "Start in seconds."},
                                "track": {"type": "integer", "description": "Audio track (0 is the first)."},
                                "cut": {
                                    "type": "array",
                                    "items": number_or_null,
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": "In and out points in the file's seconds.",
                                },
                                "volume": {"type": "number", "minimum": 0, "description": "Loudness factor (default 1)."},
                                "fade_in": {"type": "number", "minimum": 0, "description": "Seconds over which it rises from silence at its start."},
                                "fade_out": {"type": "number", "minimum": 0, "description": "Seconds over which it falls to silence at its end."},
                            },
                        },
                    },
                },
            },
        },
    }


def _clip_entry(data: dict[str, Any]) -> ClipEntry:
    cut = data.get("cut")
    return ClipEntry(
        scene=data["scene"],
        start=data.get("start"),
        name=data.get("name"),
        layer=data.get("layer", 0),
        cut=None if cut is None else (cut[0], cut[1]),
        speed=_speed(data.get("speed", 1.0)),
        cues=_cues(data.get("cues", {})),
    )


def _cues(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(event, str) and event for key, event in value.items()):
        raise ValueError(f"a clip's cues map cue names to event names, not {value!r}")
    return dict(value)


def _speed(value: Any) -> float:
    speed = float(value)
    if not speed > 0.0:
        raise ValueError(f"a clip's speed must be positive, not {value!r}")
    return speed


TRACK_FIELDS = {"name": str, "hidden": bool, "locked": bool, "muted": bool, "solo": bool, "volume": float}


def _track(data: Any, where: str) -> Track:
    _check_keys(data, list(TRACK_FIELDS), where)
    track = Track(**{key: kind(data[key]) if kind is float else data[key] for key, kind in TRACK_FIELDS.items() if key in data})
    if not isinstance(track.name, str) or not all(isinstance(getattr(track, key), bool) for key in ("hidden", "locked", "muted", "solo")):
        raise ValueError(f"{where}: a track has a name (text), hidden, locked, muted and solo (true or false) and a volume")
    if track.volume < 0:
        raise ValueError(f"{where}: a track's volume is at least 0")
    return track


def _tracks_json(scenes: dict[int, Track], audio: dict[int, Track], scripts: dict[int, Track]) -> dict[str, Any]:
    """The tracks that differ from the default, with only what differs."""
    default = Track()
    result = {}
    for key, tracks in (("scenes", scenes), ("audio", audio), ("scripts", scripts)):
        changed = {str(index): {name: getattr(track, name) for name in TRACK_FIELDS if getattr(track, name) != getattr(default, name)} for index, track in sorted(tracks.items())}
        changed = {index: data for index, data in changed.items() if data}
        if changed:
            result[key] = changed
    return result


def _audio_entry(data: dict[str, Any]) -> AudioEntry:
    cut = data.get("cut")
    file, script = data.get("file"), data.get("script")
    if (file is None) == (script is None) or not isinstance(file or script, str):
        raise ValueError("an audio clip has either a file (a path) or a script (a name)")
    return AudioEntry(
        file=file,
        start=float(data.get("start", 0.0)),
        track=int(data.get("track", 0)),
        cut=None if cut is None else (cut[0], cut[1]),
        volume=float(data.get("volume", 1.0)),
        script=script,
        part=int(data.get("part", 0)),
        fade_in=max(0.0, float(data.get("fade_in", 0.0))),
        fade_out=max(0.0, float(data.get("fade_out", 0.0))),
    )


def _audio_json(entry: AudioEntry) -> dict[str, Any]:
    data: dict[str, Any] = {"file": entry.file} if entry.script is None else {"script": entry.script, "part": entry.part}
    data["start"] = entry.start
    if entry.track:
        data["track"] = entry.track
    if entry.cut is not None:
        data["cut"] = list(entry.cut)
    if entry.volume != 1.0:
        data["volume"] = entry.volume
    if entry.fade_in:
        data["fade_in"] = entry.fade_in
    if entry.fade_out:
        data["fade_out"] = entry.fade_out
    return data


def _clip_json(entry: ClipEntry) -> dict[str, Any]:
    data: dict[str, Any] = {"scene": entry.scene}
    if entry.start is not None:
        data["start"] = entry.start
    if entry.name is not None:
        data["name"] = entry.name
    if entry.layer:
        data["layer"] = entry.layer
    if entry.cut is not None:
        data["cut"] = list(entry.cut)
    if entry.speed != 1.0:
        data["speed"] = entry.speed
    if entry.cues:
        data["cues"] = dict(entry.cues)
    return data
