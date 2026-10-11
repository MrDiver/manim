"""Editing the project from the preview page: ProjectEditor and the POST
routes of the preview server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from manimx.animation import Timeline
from manimx.cli import main
from manimx.drivers import StillRenderer
from manimx.drivers.offline import RenderSettings
from manimx.examples.navigation import config
from manimx.project import Project
from manimx.web import EditError, PreviewServer, ProjectEditor


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """A project with two clips: "Waiting" plays 1 s, waits for "go", plays 1 s; "Short" plays 2 s."""
    monkeypatch.chdir(tmp_path)
    assert main(["new", "demo", "--resolution", "720p", "--manimx-source", "/opt/manimx"]) == 0
    root = tmp_path / "demo"
    (root / "scenes" / "edited.py").write_text(
        "from manimx import Scene\n\n\n"
        "class Waiting(Scene):\n    def construct(self):\n        self.wait(1)\n        self.wait_for('go')\n        self.wait(1)\n\n\n"
        "class Short(Scene):\n    def construct(self):\n        self.wait(2)\n"
    )
    data = json.loads((root / "manimx.json").read_text())
    data["scenes"] = {"Waiting": {"module": "scenes.edited", "class": "Waiting"}, "Short": {"module": "scenes.edited", "class": "Short"}}
    data["timeline"] = {"clips": [{"scene": "Short"}, {"scene": "Waiting", "start": 0.0, "layer": 1}], "events": {}}
    (root / "manimx.json").write_text(json.dumps(data))
    return root


def test_clip_edits_are_checked_and_saved(root) -> None:
    editor = ProjectEditor(root)
    assert editor.clips()[0] == {"scene": "Short", "start": None, "layer": 0, "cut": None, "speed": 1.0}

    editor.edit_clip(0, {"start": 0.5, "layer": 2, "cut": [0.25, 1.5], "speed": 2.0})
    entry = Project.load(root).timeline[0]
    assert (entry.start, entry.layer, entry.cut, entry.speed) == (0.5, 2, (0.25, 1.5), 2.0)
    editor.edit_clip(0, {"start": None, "cut": [None, None]})  # follow the previous clip, no cut
    assert editor.clips()[0] == {"scene": "Short", "start": None, "layer": 2, "cut": None, "speed": 2.0}

    for index, changes, message in [
        (5, {"start": 1.0}, "no clip 5"),
        (0, {"length": 1.0}, "no length"),
        (0, {"start": -1}, "at least 0"),
        (0, {"start": "soon"}, "must be a number"),
        (0, {"layer": 1.5}, "whole number"),
        (0, {"cut": [2.0, 1.0]}, "after its in point"),
        (0, {"cut": [1.0]}, r"\[in, out\]"),
        (0, {"speed": 0}, "positive"),
    ]:
        with pytest.raises(EditError, match=message):
            editor.edit_clip(index, changes)


def test_speed_plays_a_clip_faster_or_slower(root) -> None:
    editor = ProjectEditor(root)
    editor.edit_clip(0, {"speed": 2.0})
    short = Project.load(root).build_timeline().clips[0]
    assert short.end - short.start == pytest.approx(1.0)
    editor.edit_clip(0, {"speed": 0.5})
    short = Project.load(root).build_timeline().clips[0]
    assert short.end - short.start == pytest.approx(4.0)
    assert json.loads((root / "manimx.json").read_text())["timeline"]["clips"][0]["speed"] == 0.5


def test_events_and_settings_are_saved(root) -> None:
    editor = ProjectEditor(root)
    editor.set_event("go", 3.0)
    assert editor.events() == {"go": 3.0}
    waiting = Project.load(root).build_timeline().clips[1]
    assert waiting.end == pytest.approx(4.0)  # goes on at 3 s, plays 1 s more
    editor.set_event("go", None)
    assert editor.events() == {}
    with pytest.raises(EditError, match="name"):
        editor.set_event("", 1.0)

    fps = next(item for item in editor.settings() if item["path"] == ["output", "fps"])
    assert (fps["type"], fps["minimum"], fps["section"]) == ("int", 1, False)
    assert any(item["section"] and item["path"] == ["effects"] for item in editor.settings())
    editor.set_setting(["output", "fps"], 24)
    assert Project.load(root).output.fps == 24
    with pytest.raises(EditError):
        editor.set_setting(["output", "fps"], 0)
    with pytest.raises(EditError):
        editor.set_setting(["output", "speed"], 1)

    (root / "manimx.json").write_text("{")
    with pytest.raises(EditError, match="cannot read"):
        editor.clips()


@pytest.fixture(scope="module")
def renderer() -> StillRenderer:
    try:
        return StillRenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def request(server: PreviewServer, path: str, body: object | None = None, raw: bytes | None = None) -> tuple[int, dict]:
    data = raw if raw is not None else None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(server.url + path.lstrip("/"), data=data)) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_edits_through_the_server(root, renderer) -> None:
    settings = RenderSettings(config(64, 36), fps=10)
    server = PreviewServer(Project.load(root).build_timeline(), settings, renderer=renderer, editor=ProjectEditor(root))
    server.start()
    try:
        status, timeline = request(server, "/api/timeline")
        assert status == 200
        assert timeline["editable"]
        short, waiting = timeline["clips"]
        assert short["entry"]["scene"] == "Short"
        assert waiting["waits"] == ["go"]
        assert timeline["events"] == [{"name": "go", "time": None, "fixed": False}]

        assert request(server, "/api/clip", {"index": 1, "start": 1.0}) == (200, {"ok": True})
        assert request(server, "/api/event", {"name": "go", "time": 2.5}) == (200, {"ok": True})
        assert request(server, "/api/setting", {"path": ["output", "fps"], "value": 30}) == (200, {"ok": True})
        project = Project.load(root)
        assert (project.timeline[1].start, project.events, project.output.fps) == (1.0, {"go": 2.5}, 30)

        status, settings_list = request(server, "/api/settings")
        assert status == 200
        assert any(item["path"] == ["output", "fps"] and item["value"] == 30 for item in settings_list)
        assert request(server, "/api/clip", {"index": 1, "layer": "top"})[0] == 400
        assert request(server, "/api/clip", {"start": 1.0})[1] == {"error": "'index' must be a int"}
        assert request(server, "/api/event", raw=b"{oops")[0] == 400
        assert request(server, "/api/event", [1, 2])[1] == {"error": "the body must be a JSON object"}
        assert request(server, "/api/nothing", {})[0] == 404
    finally:
        server.close()


def test_a_preview_without_an_editor_is_read_only(renderer) -> None:
    server = PreviewServer(Timeline(), RenderSettings(config(64, 36), fps=10), renderer=renderer)
    server.start()
    try:
        assert request(server, "/api/clip", {"index": 0})[0] == 403
        assert request(server, "/api/settings")[0] == 404
        assert request(server, "/api/timeline")[1]["editable"] is False
    finally:
        server.close()
