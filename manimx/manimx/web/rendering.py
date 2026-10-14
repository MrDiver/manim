"""Rendering the project from the preview page.

    job = RenderJob(root)
    job.start(start=0.0, end=None)   # renders/<project>.mp4, in the background
    job.status()                     # {"state": "rendering", "done": 120, "total": 600, ...}
    job.cancel()

Design decisions:

* **A separate process.** The render runs ``manimx render`` (with its
  hidden ``--progress-lines``) in the project's folder, the same as on the
  command line: it reads the saved ``manimx.json`` (the editor saves
  first), prepares the scenes itself and uses the project's full quality
  settings (samples, motion blur), its worker processes and the chosen
  H.264 encoder (the GPU's when one works). The preview's own prepared
  scenes are never touched, so editing and scrubbing go on meanwhile, and
  cancelling simply ends that process.
* **Progress as lines.** The render prints one JSON line first (the file
  and the encoder) and then one per frame written; this job keeps the
  last, and the page asks for it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .commands import EditError


class RenderJob:
    """One render at a time of the project in ``root`` (see the module docs)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._state: dict[str, Any] = {"state": "idle"}

    def start(self, start: float = 0.0, end: float | None = None, stems: bool = False) -> None:
        """Render seconds ``start`` to ``end`` (None: the end) in the background;
        with ``stems``, the sound for editing (``manimx render --stems``)."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise EditError("a render is running already")
            command = [sys.executable, "-m", "manimx", "render", "--progress-lines", "--start", str(float(start))]
            if end is not None:
                command += ["--end", str(float(end))]
            if stems:
                command.append("--stems")
            self._process = subprocess.Popen(
                command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=os.name == "posix",  # its ffmpeg and workers end with it
            )
            self._state = {"state": "preparing", "started": time.time(), "done": 0, "total": None}
            threading.Thread(target=self._follow, args=(self._process,), name="render job", daemon=True).start()

    def cancel(self) -> None:
        """Stop the render running (its unfinished file is removed)."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self._state["state"] = "cancelling"
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()

    def status(self) -> dict[str, Any]:
        """The render's state ("idle", "preparing", "rendering", "cancelling",
        "done", "cancelled", "failed"), frames done and in all, the file,
        the encoder, frames per second and seconds left."""
        with self._lock:
            state = dict(self._state)
        done, total, started = state.get("done", 0), state.get("total"), state.get("rendering since")
        if state["state"] == "rendering" and started and done:
            elapsed = time.time() - started
            state["fps"] = done / elapsed
            state["left"] = (total - done) * elapsed / done if total else None
        return state

    def video(self) -> Path | None:
        """The last finished video, if it is still there."""
        with self._lock:
            file = self._state.get("file") if self._state.get("state") == "done" else None
        return Path(file) if file and Path(file).is_file() else None

    def _follow(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None and process.stderr is not None
        for line in process.stdout:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # something else printed
            with self._lock:
                if "total" in data and self._state["state"] == "preparing":
                    self._state["state"] = "rendering"
                    self._state["rendering since"] = time.time()
                self._state.update(data)
        errors = process.stderr.read()
        code = process.wait()
        with self._lock:
            if self._state["state"] == "cancelling":
                self._state["state"] = "cancelled"
                file = self._state.get("file")
                if file:
                    Path(file).unlink(missing_ok=True)
            elif code == 0:
                self._state["state"] = "done"
            else:
                self._state["state"] = "failed"
                self._state["error"] = _last_lines(errors) or f"manimx render ended with exit code {code}"
            self._state["finished"] = time.time()


def _last_lines(text: str, count: int = 12) -> str:
    """The end of a command's error output (rich's boxes stripped)."""
    lines = [line.strip(" │╭╮╰╯─") for line in text.strip().splitlines()]
    return "\n".join(line for line in lines[-count:] if line)


__all__ = ["RenderJob"]
