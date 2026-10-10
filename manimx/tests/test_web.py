"""The browser preview: the server's page, timeline and frames."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from manimx.animation import Timeline
from manimx.cli import main
from manimx.drivers.offline import RenderSettings
from manimx.examples.navigation import Navigation, config
from manimx.project import Project
from manimx.renderer import FrameComposer, MotionBlur, RenderService
from manimx.web import PreviewServer, ProjectReloader
from PIL import Image


@pytest.fixture(scope="module")
def service() -> RenderService:
    try:
        return RenderService()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


@pytest.fixture(scope="module")
def timeline() -> Timeline:
    timeline = Timeline()
    timeline.add(Navigation().prepare(), name="navigation")
    return timeline


@pytest.fixture
def server(service, timeline):
    settings = RenderSettings(config(128, 72), fps=10, effects=(MotionBlur(samples=4),))
    server = PreviewServer(timeline, settings, title="demo", service=service)
    server.start()
    yield server
    server.close()


def get(server: PreviewServer, path: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(server.url + path.lstrip("/")) as response:
            return response.status, response.headers["Content-Type"], response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers["Content-Type"], error.read()


def test_the_page_and_its_files_are_served(server) -> None:
    status, kind, body = get(server, "/")
    assert status == 200 and kind == "text/html" and b'src="app.js"' in body
    assert get(server, "/app.js")[:2] == (200, "text/javascript")
    assert get(server, "/style.css")[:2] == (200, "text/css")
    assert get(server, "/missing.js")[0] == 404
    assert get(server, "/../server.py")[0] == 404


def test_the_timeline_describes_clips_and_segments(server, timeline) -> None:
    status, kind, body = get(server, "/api/timeline")
    data = json.loads(body)

    assert status == 200 and kind == "application/json"
    assert data["title"] == "demo" and data["duration"] == pytest.approx(timeline.duration)
    assert (data["fps"], data["width"], data["height"]) == (10, 128, 72)
    assert data["clips"] == [{"name": "navigation", "start": 0.0, "end": pytest.approx(timeline.duration), "layer": 0}]
    assert data["segments"] and all(segment["clip"] == 0 for segment in data["segments"])


def test_frames_are_rendered_at_the_asked_width_without_motion_blur(server, service, timeline) -> None:
    status, kind, body = get(server, "/api/frame?t=1.5&w=64&format=png")
    image = np.asarray(Image.open(io.BytesIO(body)))

    assert status == 200 and kind == "image/png" and image.shape == (36, 64, 4)
    expected = FrameComposer(service).compose(timeline.frame_at, dataclasses.replace(config(64, 36), time=1.5))
    np.testing.assert_array_equal(image, expected)
    status, kind, body = get(server, "/api/frame?t=999")  # clamped to the end; full width; JPEG
    assert status == 200 and kind == "image/jpeg" and Image.open(io.BytesIO(body)).size == (128, 72)


def test_bad_requests_and_failing_scenes_answer_with_errors(server, monkeypatch) -> None:
    assert get(server, "/api/frame")[0] == 400
    assert get(server, "/api/frame?t=soon")[0] == 400
    assert get(server, "/api/frame?t=1&format=gif")[0] == 400
    with pytest.raises(ValueError, match="unknown image format"):
        server.frame(1.0, image_format="gif")

    def broken(t: float):
        raise RuntimeError("the scene broke")

    monkeypatch.setattr(server, "_frame_at", broken)  # the core's frames, or the timeline's
    status, _, body = get(server, "/api/frame?t=1")
    assert status == 500 and json.loads(body) == {"error": "RuntimeError: the scene broke"}


def test_the_ui_command_serves_until_interrupted(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["new", "demo", "--example", "--resolution", "720p", "--manimx-source", "/opt/manimx"]) == 0
    monkeypatch.chdir(tmp_path / "demo")
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    def interrupted(self) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(PreviewServer, "serve_forever", interrupted)
    assert main(["ui"]) == 0
    assert opened and opened[0].startswith("http://127.0.0.1:") and opened[0] in capsys.readouterr().out

    # A scene that fails to build: the preview starts anyway and shows it.
    scene_file = next((tmp_path / "demo" / "scenes").glob("intro.py"))
    scene_file.write_text(scene_file.read_text() + "\n    oops(\n")
    assert main(["ui", "--no-open"]) == 0
    assert "shown in the page" in capsys.readouterr().out

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        assert main(["ui", "--port", str(taken.getsockname()[1]), "--no-open"]) == 1
    assert "cannot serve on port" in capsys.readouterr().err


def _write_scene(path, cls: str, body: str) -> None:
    """Write a scene module, with a modification time that always moves on."""
    path.write_text(f"from manimx import Scene, Square\n\n\nclass {cls}(Scene):\n    def construct(self):\n        {body}\n")
    stamp = path.stat().st_mtime_ns + 10**9
    os.utime(path, ns=(stamp, stamp))


def test_saving_a_scene_rebuilds_the_timeline(tmp_path, monkeypatch, service) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["new", "demo", "--example", "--resolution", "720p", "--manimx-source", "/opt/manimx"]) == 0
    root = tmp_path / "demo"
    project = Project.load(root)
    name, entry = next(iter(project.scenes.items()))
    module_file = root / Path(*entry.module.split(".")).with_suffix(".py")
    project.timeline = [clip for clip in project.timeline if clip.scene == name]
    project.save()
    _write_scene(module_file, entry.cls, "self.wait(2)")

    settings = RenderSettings(config(64, 36), fps=10)
    server = PreviewServer(Project.load(root).build_timeline(), settings, service=service)
    reloader = ProjectReloader(server, root, lambda project: settings)
    assert server.timeline.duration == pytest.approx(2.0)
    assert not reloader.check()  # nothing saved yet

    _write_scene(module_file, entry.cls, "self.add(Square()); self.wait(3)")
    assert reloader.check()
    assert (server.version, server.error) == (1, None)
    assert server.timeline.duration == pytest.approx(3.0)  # the module was imported again

    _write_scene(module_file, entry.cls, "self.wait(")
    assert reloader.check()
    assert server.version == 2 and "SyntaxError" in server.error
    where = f"{module_file.relative_to(root)}:6"
    assert where in server.error and "importlib" not in server.error  # where, without import machinery
    assert server.timeline.duration == pytest.approx(3.0)  # the last good timeline stays

    _write_scene(module_file, entry.cls, "self.wait(1)")
    assert reloader.check()
    assert (server.version, server.error) == (3, None)
    server.close()


def test_the_page_waits_for_changes(server) -> None:
    version = server.version
    assert server.wait_for_change(version, timeout=0.01) == (version, None)  # nothing happened
    status, _, body = get(server, f"/api/changes?since={version - 1}")
    assert status == 200 and json.loads(body) == {"version": version, "error": None}

    waiting = []
    thread = threading.Thread(target=lambda: waiting.append(server.wait_for_change(version)))
    thread.start()
    server.report("ValueError: broken")
    thread.join(timeout=5)
    assert waiting == [(version + 1, "ValueError: broken")]
    assert json.loads(get(server, "/api/timeline")[2])["error"] == "ValueError: broken"
    server.replace(server.timeline)
    assert server.error is None and server.version == version + 2


def test_only_the_projects_own_modules_are_unloaded(tmp_path, monkeypatch) -> None:
    import types

    own, installed = types.ModuleType("own_scene"), types.ModuleType("installed_package")
    own.__file__ = str(tmp_path / "scenes" / "own_scene.py")
    installed.__file__ = str(tmp_path / ".venv" / "lib" / "site-packages" / "installed_package.py")
    monkeypatch.setitem(sys.modules, "own_scene", own)
    monkeypatch.setitem(sys.modules, "installed_package", installed)

    loaded = Project.__new__(Project)
    loaded.root = tmp_path.resolve()
    assert "own_scene" in loaded.unload_modules()
    assert "own_scene" not in sys.modules and "installed_package" in sys.modules
