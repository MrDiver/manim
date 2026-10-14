"""Reload on save: rebuild a project's preview when its files change.

    reloader = ProjectReloader(server, project.root, preview_settings)
    reloader.start()

Every ``interval`` seconds the reloader compares the modification times
of ``manimx.json`` and the project's Python files with the last ones it
saw. After a change it loads the project again, forgets the project's own
modules (:meth:`~manimx.project.Project.unload_modules`) so the scenes are
imported as they are on disk now, builds the timeline and hands it to the
server (:meth:`~.server.PreviewServer.replace`). A build that fails keeps
the timeline shown and reports the error (:meth:`~.server.PreviewServer.report`),
which the page shows until a save fixes it.

Design decisions:

* **Polling, no dependency.** Comparing a few dozen modification times
  four times a second costs nothing, works the same on every system, and
  needs no file-watching package.
* **The whole project is rebuilt.** Preparing scenes is fast (the
  benchmark project: 0.3 s), and a change in one module can affect any
  scene that imports it, so the reloader does not try to find out which
  clips changed.
* **What is watched:** ``manimx.json``, ``*.py`` files and the scripts
  (``scripts/*.txt`` and their times, ``*.json``) in the project folder,
  except hidden folders (a ``.venv``), ``__pycache__`` and the output
  folder.
* **With a project editor** (editing.py, as in ``manimx ui``) the
  reloader only says which files changed: the editor prepares again just
  the scenes in them, and ignores its own autosaves of ``manimx.json``.
"""

from __future__ import annotations

import os
import sysconfig
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from ..drivers.offline import RenderSettings
from ..project import PROJECT_FILE, Project
from ..voice.script import SCRIPTS_FOLDER
from .editing import ProjectEditor
from .server import PreviewServer

Snapshot = dict[Path, int]


class ProjectReloader:
    """Rebuilds ``server``'s timeline when the project in ``root`` changes."""

    def __init__(
        self,
        server: PreviewServer,
        root: Path,
        settings: Callable[[Project], RenderSettings],
        interval: float = 0.25,
        editor: ProjectEditor | None = None,
    ) -> None:
        self.server = server
        self.editor = editor
        self.root = Path(root).resolve()
        self.settings = settings
        self.interval = interval
        self._output = self._output_folder()
        self._seen = self.snapshot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> Snapshot:
        """Modification times of the watched files."""
        files: Snapshot = {}
        project_file = self.root / PROJECT_FILE
        if project_file.is_file():
            files[project_file] = project_file.stat().st_mtime_ns
        skipped = {"__pycache__", self._output}
        for folder, children, names in os.walk(self.root):
            # Prune instead of filtering afterwards: a .venv holds thousands of files.
            children[:] = [child for child in children if not child.startswith(".") and child not in skipped]
            for name in names:
                script = Path(folder) == self.root / SCRIPTS_FOLDER and name.endswith((".txt", ".json"))
                if name.endswith(".py") or script:
                    path = Path(folder) / name
                    try:
                        files[path] = path.stat().st_mtime_ns
                    except OSError:  # deleted meanwhile
                        continue
        return files

    def check(self) -> bool:
        """Reload if a watched file changed since the last check; returns
        whether it did."""
        current = self.snapshot()
        if current == self._seen:
            return False
        changed = {path for path in current.keys() | self._seen.keys() if current.get(path) != self._seen.get(path)}
        self._seen = current
        if self.editor is not None:
            self.editor.files_changed(changed)
        else:
            self.reload()
        return True

    def reload(self) -> None:
        if self.editor is not None:
            self.editor.load()
            return
        try:
            project = Project.load(self.root)
            project.unload_modules()
            timeline = project.build_timeline()
            settings = self.settings(project)
        except Exception as error:  # a syntax error, a failing construct(): show it, keep the last timeline
            self.server.report(_describe(error, self.root))
        else:
            self.server.replace(timeline, settings)

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(target=self._run, name="manimx reload", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.check()

    def _output_folder(self) -> str:
        try:
            return Path(Project.load(self.root).output.directory).parts[0]
        except Exception:  # an unreadable manimx.json: watch everything
            return ""


def _describe(error: BaseException, root: Path) -> str:
    """The error and, for one raised from a scene, where in the scene's own
    code: frames of manimx and of Python's standard library (its import
    machinery) are left out, like the command line leaves them out. Paths
    inside ``root`` are shown relative to it.
    """
    lines = [f"{type(error).__name__}: {error}"]
    cause = error.__cause__
    if cause is not None:
        lines.append(f"{type(cause).__name__}: {cause}")
        if isinstance(cause, SyntaxError) and cause.filename:
            lines.append(f"  {_shown(cause.filename, root)}:{cause.lineno}")
            if cause.text:
                lines.append(f"    {cause.text.strip()}")
        own = [frame for frame in traceback.extract_tb(cause.__traceback__) if _is_users(frame.filename)]
        lines.extend(f"  {_shown(frame.filename, root)}:{frame.lineno} in {frame.name}\n    {frame.line}" for frame in own[-3:])
    return "\n".join(lines)


def _shown(filename: str, root: Path) -> str:
    path = Path(filename).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else filename


def _is_users(filename: str) -> bool:
    if filename.startswith("<"):  # <frozen importlib._bootstrap> and the like
        return False
    path = Path(filename).resolve()
    return not any(path.is_relative_to(folder) for folder in _NOT_USERS)


_NOT_USERS = (Path(__file__).resolve().parents[1], Path(sysconfig.get_paths()["stdlib"]).resolve())
"""manimx itself and Python's standard library."""


__all__ = ["ProjectReloader"]
