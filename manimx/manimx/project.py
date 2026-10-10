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
parallel with another one on a higher ``layer``.
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

from .animation.scene import PreparedScene, Scene
from .animation.timing import NEVER, Timeline
from .settings import OutputSettings, PerformanceSettings, Settings, SettingsError, UnknownSetting
from .settings import from_json as settings_from_json
from .settings import schema as settings_schema
from .settings import to_json as settings_to_json

PROJECT_FILE = "manimx.json"
SCHEMA_FILE = ".manimx/project.schema.json"
FORMAT_VERSION = 2
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


@dataclass(eq=False)
class Project:
    root: Path
    name: str
    settings: Settings = field(default_factory=Settings)
    scenes: dict[str, SceneEntry] = field(default_factory=dict)
    timeline: list[ClipEntry] = field(default_factory=list)
    events: dict[str, float] = field(default_factory=dict)
    """Event times set by hand (clicks, transcript words come later)."""

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
        if isinstance(data, dict) and data.get("format") in _MIGRATIONS:
            data = _MIGRATIONS[data["format"]](data)
        if not isinstance(data, dict) or data.get("format") != FORMAT_VERSION:
            found = data.get("format") if isinstance(data, dict) else None
            raise ProjectError(f"{PROJECT_FILE} has format {found}, this manimx reads format {FORMAT_VERSION}")
        _check_keys(data, ["$schema", "format", "name", *SECTIONS, "scenes", "timeline"], "")
        if not isinstance(data.get("name"), str):
            raise ProjectError(f'invalid {PROJECT_FILE}: "name" must be a string')
        timeline = data.get("timeline", {})
        _check_keys(timeline, ["clips", "events"], "timeline")
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
                _check_keys(entry, ["scene", "start", "name", "layer", "cut"], f"timeline.clips[{index}]")
                clips.append(_clip_entry(entry))
            events = {name: float(time) for name, time in timeline.get("events", {}).items()}
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ProjectError(f"invalid {PROJECT_FILE}: {error}") from error
        return cls(root=root, name=data["name"], settings=settings, scenes=scenes, timeline=clips, events=events)

    def to_json(self) -> dict[str, Any]:
        return {
            "$schema": f"./{SCHEMA_FILE}",
            "format": FORMAT_VERSION,
            "name": self.name,
            **settings_to_json(self.settings),
            "scenes": {name: {"module": entry.module, "class": entry.cls} for name, entry in self.scenes.items()},
            "timeline": {"clips": [_clip_json(entry) for entry in self.timeline], "events": dict(self.events)},
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
            core=settings.performance.core,
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

    def _check_scene(self, name: str) -> SceneEntry:
        entry = self.scenes.get(name)
        if entry is None:
            close = difflib.get_close_matches(name, self.scenes, n=1)
            hint = f"did you mean {close[0]!r}?" if close else f"create it with `manimx generate scene {name}`"
            raise ProjectError(f"the project has no scene named {name!r}", hint=hint)
        return entry

    def build_timeline(self, progress: Callable[[ClipEntry], None] | None = None) -> Timeline:
        """The live timeline: every clip of ``manimx.json``, prepared and
        placed. ``progress`` is called before each clip is prepared."""
        timeline = Timeline(events=dict(self.events))
        previous_end = 0.0
        used: set[str] = set()
        for entry in self.timeline:
            if progress is not None:
                progress(entry)
            start = previous_end if entry.start is None else entry.start
            clip = timeline.add(self.prepare(entry.scene), start=start, name=_unique(entry.name or entry.scene, used), layer=entry.layer)
            if entry.cut is not None:
                clip.cut(*entry.cut)
            previous_end = clip.end
            if previous_end == NEVER:
                raise ProjectError(
                    f"clip {clip.name!r} waits for an event that never fires, so the next clip has no start",
                    hint="give the next clip an explicit start, or set the event's time under \"events\" in manimx.json",
                )
        return timeline

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


_MIGRATIONS = {1: _from_format_1}
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
                            },
                        },
                    },
                    "events": {
                        "type": "object",
                        "description": "Times (seconds) of events scenes wait for with wait_for().",
                        "additionalProperties": {"type": "number"},
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
    )


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
    return data
