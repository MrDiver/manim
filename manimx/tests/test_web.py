"""The browser preview: the server's page, timeline and frames."""

from __future__ import annotations

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
from manimx.drivers import StillRenderer, render_image
from manimx.drivers.offline import RenderSettings
from manimx.examples.navigation import Navigation, config
from manimx.project import Project
from manimx.renderer import RenderSamples
from manimx.web import PreviewServer, ProjectReloader
from PIL import Image


@pytest.fixture(scope="module")
def renderer() -> StillRenderer:
    try:
        return StillRenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


@pytest.fixture(scope="module")
def timeline() -> Timeline:
    timeline = Timeline()
    timeline.add(Navigation().prepare(), name="navigation")
    return timeline


@pytest.fixture
def server(renderer, timeline):
    settings = RenderSettings(config(128, 72), fps=10, effects=(RenderSamples(count=4, shutter=1 / 20),))
    server = PreviewServer(timeline, settings, title="demo", renderer=renderer)
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
    (clip,) = data["clips"]
    assert {key: clip[key] for key in ("name", "start", "end", "layer")} == {"name": "navigation", "start": 0.0, "end": pytest.approx(timeline.duration), "layer": 0}
    assert clip["nodes"][0]["type"] == "Camera"
    assert data["segments"] and all(segment["clip"] == 0 for segment in data["segments"])


def test_frames_are_rendered_at_the_asked_width_with_one_render(server, timeline) -> None:
    status, kind, body = get(server, "/api/frame?t=1.5&w=64&format=png")
    image = np.asarray(Image.open(io.BytesIO(body)))

    assert status == 200 and kind == "image/png" and image.shape == (36, 64, 4)
    expected = render_image(timeline, 1.5, config(64, 36))
    np.testing.assert_array_equal(image, expected)
    status, kind, body = get(server, "/api/frame?t=999&format=png")  # past the end: the background; full width
    image = np.asarray(Image.open(io.BytesIO(body)))
    assert status == 200 and image.shape == (72, 128, 4)
    assert np.allclose(image.reshape(-1, 4), np.array(config(64, 36).background_rgba) * 255, atol=1)
    assert get(server, "/api/frame?t=1")[1] == "image/jpeg"


def test_a_frame_by_number_at_its_version_is_kept_by_the_browser(server, timeline) -> None:
    def headers(path: str) -> tuple[bytes, str | None, str | None]:
        with urllib.request.urlopen(server.url + path) as response:
            return response.read(), response.headers["Cache-Control"], response.headers["X-Version"]

    version, session = server.version, server.session
    assert headers(f"api/frame?f=15&v={version}&w=64")[1] == "no-store"  # another server's frame
    body, cache, shown = headers(f"api/frame?f=15&v={version}&s={session}&w=64&format=png")
    assert (cache, shown) == ("private, max-age=86400, immutable", str(version))
    assert body == get(server, "/api/frame?t=1.5&w=64&format=png")[2]  # frame 15 at 10 fps
    server.change(lambda: timeline.clips[0].move(0.5))
    _, cache, shown = headers(f"api/frame?f=15&v={version}&s={session}&w=64")  # an edit the page has not heard of
    assert (cache, shown) == ("no-store", str(version + 1))
    assert headers("api/frame?t=1.5")[1] == "no-store"


def test_thumbnails_show_a_clip_alone_at_a_scene_time(server, timeline) -> None:
    scene = timeline.clips[0].scene.uid
    assert server.timeline_json()["clips"][0]["scene"] == scene
    with urllib.request.urlopen(f"{server.url}api/thumbnail?clip=0&scene={scene}&at=1.5&s={server.session}&w=64") as response:
        assert response.headers["Cache-Control"] == "private, max-age=86400, immutable"
        assert Image.open(io.BytesIO(response.read())).size == (64, 36)
    server.change(lambda: timeline.clips[0].move(3.0))  # moved: the same thumbnail
    assert server.thumbnail(0, scene, 1.5, 64) is not None
    assert get(server, f"/api/thumbnail?clip=0&scene={scene + 1}&at=1&w=64")[0] == 400  # not that scene
    assert get(server, f"/api/thumbnail?clip=5&scene={scene}&at=1&w=64")[0] == 400


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


def test_saving_a_scene_rebuilds_the_timeline(tmp_path, monkeypatch, renderer) -> None:
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
    server = PreviewServer(Project.load(root).build_timeline(), settings, renderer=renderer)
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
    assert status == 200 and json.loads(body) == {"version": version, "picture": server.picture, "error": None}

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


def test_labels_give_every_paths_index_in_its_group_on_the_frame() -> None:
    from manimx.animation import Group, Scene
    from manimx.userland import LEFT, RIGHT, Square, TypstMath
    from manimx.web.labels import index_labels

    class Labeled(Scene):
        def construct(self) -> None:
            self.add(TypstMath("x + 1").shift(LEFT * 2), Group(Square(color="#ff0000"), Square(stroke_color="#00ff00", fill_color="#0000ff", fill_opacity=0.5).shift(RIGHT)).shift(RIGHT * 3))

    timeline = Timeline()
    timeline.add(Labeled().prepare(), name="labeled")
    labels = index_labels(timeline, 0.0, 16 / 9)
    assert [(label["index"], label["name"]) for label in labels[:3]] == [(0, "x"), (1, "+"), (2, "1")]
    assert labels[0]["group"] == "$ x + 1 $" and [label["index"] for label in labels[3:]] == [0, 1]
    # Fractions of the frame: the formula left of the middle, the squares right of it, all at mid height.
    assert all(label["x"] < 0.5 for label in labels[:3]) and all(label["x"] > 0.5 for label in labels[3:])
    assert all(label["y"] == pytest.approx(0.5, abs=0.05) for label in labels)
    assert [label["color"] for label in labels[3:]] == ["#ff0000", "#0000ff"]  # the stroke without fill, else the fill
    assert labels[0]["color"] == "#000000"  # Typst's default text color
    first_square = labels[3]
    assert first_square["x"] == pytest.approx(0.5 + 3 / (8 * 16 / 9))  # x = 3 of a 14.2 wide frame


def test_the_labels_route_answers_json(server) -> None:
    status, kind, body = get(server, "/api/labels?t=0.5")
    assert status == 200 and kind == "application/json" and isinstance(json.loads(body), list)


def test_scene_clips_that_play_sounds_show_them(renderer) -> None:
    from manimx.examples.bouncing import Bouncing

    timeline = Timeline()
    timeline.add(Bouncing().prepare(), name="bouncing")
    server = PreviewServer(timeline, RenderSettings(config(64, 36), fps=10), renderer=renderer)
    server.start()
    try:
        status, _, body = get(server, "/api/timeline")
        assert json.loads(body)["clips"][0]["sounds"] > 0
        status, _, body = get(server, "/api/clip/waveform?clip=0&n=16")
        peaks = json.loads(body)
        assert status == 200 and len(peaks) == 16 and max(high for _, high in peaks) > 0.01
        assert get(server, "/api/clip/waveform?clip=5&n=16")[0] == 400
    finally:
        server.close()
