"""Creating projects and scenes: the file work behind ``manimx new`` and
``manimx generate``, separate from how it is shown.

Every operation returns the list of :class:`Change` it made, so the CLI
can print ``CREATE``/``UPDATE`` lines like the Angular CLI does, and tests
can check them.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from pathlib import Path

import manimx

from ..project import PROJECT_FILE, SCHEMA_FILE, OutputSettings, Project, ProjectError
from . import templates


@dataclass(frozen=True)
class Change:
    action: str  # "create", "update" or "delete"
    path: Path
    note: str = ""


@dataclass(frozen=True)
class SceneName:
    """The names derived from what the user typed (``title-card``)."""

    cls: str  # TitleCard
    module: str  # title_card

    @classmethod
    def parse(cls, text: str) -> SceneName:
        words = [word for word in re.split(r"[^0-9A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])", text.strip()) if word]
        if not words or not words[0][0].isalpha():
            raise ProjectError(f"{text!r} is not a usable scene name", hint="use letters, digits and dashes, starting with a letter, e.g. title-card")
        name = cls("".join(word[:1].upper() + word[1:] for word in words), "_".join(word.lower() for word in words))
        if keyword.iskeyword(name.module):
            raise ProjectError(f"{text!r} is a Python keyword", hint="pick another name")
        return name


RESOLUTIONS = {
    "1080p": (1920, 1080, 60),
    "720p": (1280, 720, 30),
    "4k": (3840, 2160, 60),
}


def manimx_source() -> str | None:
    """Path of the manimx checkout this CLI runs from, if it is one (then new
    projects depend on it by path, since manimx is not published yet)."""
    root = Path(manimx.__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and 'name = "manimx"' in pyproject.read_text(encoding="utf-8"):
        return str(root)
    return None


def create_project(
    directory: Path,
    name: str,
    example: bool = True,
    output: OutputSettings | None = None,
    source: str | None = None,
) -> tuple[Project, list[Change]]:
    """A new project in ``directory`` (which must not exist or be empty)."""
    if directory.exists() and any(directory.iterdir()):
        raise ProjectError(f"{directory} already exists and is not empty", hint="pick another name or an empty folder")
    package = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not package:
        raise ProjectError(f"{name!r} is not a usable project name", hint="use letters, digits and dashes")
    changes: list[Change] = []

    def write(relative: str, text: str, note: str = "") -> None:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        changes.append(Change("create", path, note))

    directory.mkdir(parents=True, exist_ok=True)
    source = manimx_source() if source is None else source
    sources = templates.UV_SOURCES.format(path=source) if source else ""
    write("pyproject.toml", templates.PYPROJECT.format(package=package, sources=sources))
    write("README.md", templates.README.format(name=name))
    write(".gitignore", templates.GITIGNORE)
    write("scenes/__init__.py", templates.SCENES_INIT)
    write("assets/.gitkeep", "")
    project = Project(directory.resolve(), name, output=output or OutputSettings())
    project.save()
    changes[:0] = [Change("create", directory / PROJECT_FILE), Change("create", directory / SCHEMA_FILE, "lets editors check manimx.json")]
    if example:
        for scene, kind in (("intro", "animation"), ("bounce", "simulation")):
            changes += [change for change in generate_scene(project, scene, kind) if change.action == "create"]
    return project, changes


def generate_scene(project: Project, text: str, kind: str = "animation", on_timeline: bool = True) -> list[Change]:
    """Write a scene module and register it in ``manimx.json`` (appended
    to the timeline)."""
    if kind not in templates.KINDS:
        raise ProjectError(f"unknown scene kind {kind!r}", hint=f"use one of: {', '.join(templates.KINDS)}")
    name = SceneName.parse(text)
    path = project.root / "scenes" / f"{name.module}.py"
    if path.exists():
        raise ProjectError(f"{path.relative_to(project.root)} already exists", hint="pick another name")
    project.add_scene(name.cls, f"scenes.{name.module}", name.cls, on_timeline)  # raises on duplicates
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(templates.KINDS[kind].format(cls=name.cls), encoding="utf-8")
    init = path.parent / "__init__.py"
    created_init = not init.exists()
    if created_init:
        init.write_text(templates.SCENES_INIT, encoding="utf-8")
    project.save()
    note = f"registered {name.cls}" + (", appended to the timeline" if on_timeline else "")
    return [
        Change("create", path, f"{name.cls} ({kind})"),
        *([Change("create", init)] if created_init else []),
        Change("update", project.root / PROJECT_FILE, note),
    ]


def remove_scene(project: Project, name: str, delete_file: bool = False) -> list[Change]:
    entry = project.remove_scene(name)
    changes = []
    if delete_file:
        path = project.root / Path(*entry.module.split(".")).with_suffix(".py")
        if path.exists():
            path.unlink()
            changes.append(Change("delete", path))
    project.save()
    return [*changes, Change("update", project.root / PROJECT_FILE, f"removed {name}")]
