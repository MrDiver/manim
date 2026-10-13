"""Editing the previewed project live: ProjectEditor and the POST routes
of the preview server."""

from __future__ import annotations

import json
import os
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
from manimx.web import EditError, PreviewServer, ProjectEditor, ProjectReloader

SETTINGS = RenderSettings(config(64, 36), fps=10)


def write(path: Path, text: str) -> None:
    """Write a file with a modification time that always moves on."""
    old = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(text)
    stamp = max(path.stat().st_mtime_ns, old) + 10**9
    os.utime(path, ns=(stamp, stamp))


def short_scene(seconds: float) -> str:
    return f"from manimx import Scene\n\n\nclass Short(Scene):\n    def construct(self):\n        self.wait({seconds})\n"


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """Clips: Short (2 s), Short again after it, and Waiting at 0 s on
    layer 1 (plays 1 s, waits for "go", plays 1 s); one module per scene."""
    monkeypatch.chdir(tmp_path)
    assert main(["new", "demo", "--resolution", "720p", "--manimx-source", "/opt/manimx"]) == 0
    root = tmp_path / "demo"
    write(root / "scenes" / "short.py", short_scene(2))
    write(
        root / "scenes" / "waiting.py",
        "from manimx import Scene\n\n\nclass Waiting(Scene):\n    def construct(self):\n        self.wait(1)\n        self.wait_for('go')\n        self.wait(1)\n",
    )
    data = json.loads((root / "manimx.json").read_text())
    data["scenes"] = {"Short": {"module": "scenes.short", "class": "Short"}, "Waiting": {"module": "scenes.waiting", "class": "Waiting"}}
    data["timeline"] = {"clips": [{"scene": "Short"}, {"scene": "Short"}, {"scene": "Waiting", "start": 0.0, "layer": 1}], "events": {}}
    write(root / "manimx.json", json.dumps(data))
    return root


@pytest.fixture
def editor(root) -> ProjectEditor:
    editor = ProjectEditor(root, lambda project: SETTINGS, save_delay=60.0)
    editor.load()
    return editor


def starts(editor: ProjectEditor) -> list[float]:
    return [clip.start for clip in editor.timeline.clips]


def test_clip_edits_change_the_live_timeline_and_save_later(root, editor) -> None:
    assert starts(editor) == [0.0, 2.0, 0.0]
    saved = (root / "manimx.json").read_text()

    editor.edit_clip(0, {"start": 1.0, "cut": [0.5, None]})
    assert starts(editor) == [1.0, 2.5, 0.0]  # the clip after it follows its new end
    assert (root / "manimx.json").read_text() == saved  # not saved yet
    editor.edit_clip(0, {"speed": 0.5, "layer": 2})
    first = editor.timeline.clips[0]
    assert (first.end - first.start, first.layer) == (pytest.approx(3.0), 2)
    assert editor.clips()[0] == {"scene": "Short", "start": 1.0, "layer": 2, "cut": [0.5, None], "speed": 0.5}

    editor.save()
    entry = Project.load(root).timeline[0]
    assert (entry.start, entry.layer, entry.cut, entry.speed) == (1.0, 2, (0.5, None), 0.5)
    editor.edit_clip(0, {"start": None, "cut": [None, None], "speed": 1})
    assert starts(editor) == [0.0, 2.0, 0.0]

    editor.edit_clips([{"index": 0, "start": 1.0}, {"index": 2, "start": 3.0}])  # dragged together
    assert starts(editor) == [1.0, 3.0, 3.0]

    for changes, message in [
        ({"index": 5, "start": 1.0}, "no clip 5"),
        ({"index": 0, "length": 1.0}, "no length"),
        ({"index": 0, "start": -1}, "at least 0"),
        ({"index": 0, "start": "soon"}, "must be a number"),
        ({"index": 0, "layer": 1.5}, "whole number"),
        ({"index": 0, "cut": [2.0, 1.0]}, "after its in point"),
        ({"index": 0, "cut": [1.0]}, r"\[in, out\]"),
        ({"index": 0, "speed": 0}, "positive"),
    ]:
        with pytest.raises(EditError, match=message):
            editor.edit_clips([changes])


def test_events_and_settings_change_live(root, editor) -> None:
    seen = []
    editor.render_settings = lambda project: seen.append(project.output.fps) or SETTINGS
    waiting = editor.timeline.clips[2]
    assert waiting.end == float("inf")
    editor.set_event("go", 3.0)
    assert waiting.end == pytest.approx(4.0)  # goes on at 3 s, plays 1 s more
    editor.set_event("go", None)
    assert waiting.end == float("inf")
    with pytest.raises(EditError, match="name"):
        editor.set_event("", 1.0)

    fps = next(item for item in editor.settings() if item["path"] == ["output", "fps"])
    assert (fps["type"], fps["minimum"], fps["section"]) == ("int", 1, False)
    editor.set_setting(["output", "fps"], 24)
    assert seen == [24]
    with pytest.raises(EditError):
        editor.set_setting(["output", "fps"], 0)
    with pytest.raises(EditError):
        editor.set_setting(["output", "speed"], 1)
    editor.save()
    assert Project.load(root).output.fps == 24


def test_only_the_scenes_of_a_changed_file_are_prepared_again(root, editor) -> None:
    short, again, waiting = (clip.scene for clip in editor.timeline.clips)
    editor.edit_clip(1, {"start": 5.0})

    write(root / "scenes" / "short.py", short_scene(3))
    editor.files_changed({root / "scenes" / "short.py"})
    clips = editor.timeline.clips
    assert clips[0].scene is not short
    assert clips[1].scene is not again
    assert clips[2].scene is waiting  # its file did not change
    assert clips[0].end == pytest.approx(3.0)
    assert clips[1].start == 5.0  # edits stay

    write(root / "scenes" / "helpers.py", "VALUE = 1\n")  # no scene's own module: any scene may use it
    editor.files_changed({root / "scenes" / "helpers.py"})
    assert editor.timeline.clips[2].scene is not waiting


def test_manimx_json_is_read_again_only_when_someone_else_changed_it(root, editor) -> None:
    timeline = editor.timeline
    editor.edit_clip(0, {"layer": 3})
    editor.save()
    editor.files_changed({root / "manimx.json"})
    assert editor.timeline is timeline  # its own save

    waiting = timeline.clips[2].scene
    data = json.loads((root / "manimx.json").read_text())
    data["timeline"]["clips"][0]["layer"] = 7
    write(root / "manimx.json", json.dumps(data))
    editor.files_changed({root / "manimx.json"})
    assert editor.timeline is not timeline
    assert editor.timeline.clips[0].layer == 7
    assert editor.timeline.clips[2].scene is waiting  # unchanged scenes are reused

    write(root / "scenes" / "short.py", "class Broken(\n")
    server = PreviewServer(Timeline(), SETTINGS, editor=editor)
    editor.files_changed({root / "scenes" / "short.py"})
    assert "SyntaxError" in server.error
    server.close()


def test_the_reloader_hands_changed_files_to_the_editor(root, editor) -> None:
    server = PreviewServer(Timeline(), SETTINGS, editor=editor)
    reloader = ProjectReloader(server, root, lambda project: SETTINGS, editor=editor)
    reloader.reload()
    assert server.timeline is editor.timeline
    write(root / "scenes" / "short.py", short_scene(4))
    assert reloader.check()
    assert editor.timeline.clips[0].end == pytest.approx(4.0)
    server.close()


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
    editor = ProjectEditor(root, lambda project: SETTINGS, save_delay=60.0)
    server = PreviewServer(Timeline(), SETTINGS, renderer=renderer, editor=editor)
    editor.load()
    server.start()
    try:
        status, timeline = request(server, "/api/timeline")
        assert status == 200
        assert timeline["editable"]
        short, _, waiting = timeline["clips"]
        assert short["entry"]["scene"] == "Short"
        assert [wait["event"] for wait in waiting["waits"]] == ["go"]
        assert timeline["events"] == [{"name": "go", "time": None, "fixed": False}]

        version = server.version
        status, answer = request(server, "/api/clip", {"index": 2, "start": 1.0})
        assert status == 200
        assert answer["version"] > version
        assert request(server, "/api/clip", {"clips": [{"index": 0, "start": 0.5}, {"index": 1, "layer": 1}]})[0] == 200
        assert request(server, "/api/event", {"name": "go", "time": 2.5})[0] == 200
        assert request(server, "/api/setting", {"path": ["output", "fps"], "value": 30})[0] == 200
        assert [clip.start for clip in editor.timeline.clips] == [0.5, 2.5, 1.0]
        assert editor.project.events == {"go": 2.5}

        status, settings_list = request(server, "/api/settings")
        assert status == 200
        assert any(item["path"] == ["output", "fps"] and item["value"] == 30 for item in settings_list)
        assert request(server, "/api/clip", {"index": 1, "layer": "top"})[0] == 400
        assert request(server, "/api/clip", {"clips": "all"})[1] == {"error": "'clips' must be a list"}
        assert request(server, "/api/event", raw=b"{oops")[0] == 400
        assert request(server, "/api/event", [1, 2])[1] == {"error": "the body must be a JSON object"}
        assert request(server, "/api/nothing", {})[0] == 404
    finally:
        server.close()


def test_a_preview_without_an_editor_is_read_only(renderer) -> None:
    server = PreviewServer(Timeline(), SETTINGS, renderer=renderer)
    server.start()
    try:
        assert request(server, "/api/clip", {"index": 0})[0] == 403
        assert request(server, "/api/settings")[0] == 404
        assert request(server, "/api/timeline")[1]["editable"] is False
    finally:
        server.close()


def test_clips_are_added_from_scenes_and_removed(root, editor) -> None:
    assert [scene["name"] for scene in editor.scenes()] == ["Short", "Waiting"]
    assert editor.scenes()[0]["clips"] == 2

    index = editor.add_clip("Waiting", 5.0, layer=2)
    clip = editor.timeline.clips[index]
    assert (clip.name, clip.start, clip.layer) == ("Waiting-2", 5.0, 2)
    assert editor.project.timeline[index].scene == "Waiting"
    with pytest.raises(EditError, match="no scene"):
        editor.add_clip("Nothing", 0.0)

    editor.edit_clip(0, {"start": 1.0})
    editor.remove_clips([0])  # the clip after it followed it: it keeps its place
    assert [clip.start for clip in editor.timeline.clips] == [3.0, 0.0, 5.0]
    assert editor.project.timeline[0].start == 3.0
    editor.remove_clips([0, 2])
    assert [clip.name for clip in editor.timeline.clips] == ["Waiting"]
    with pytest.raises(EditError, match="no clip"):
        editor.remove_clips([4])


def test_the_page_adds_removes_and_plays(root, renderer) -> None:
    editor = ProjectEditor(root, lambda project: SETTINGS, save_delay=60.0)
    server = PreviewServer(Timeline(), SETTINGS, renderer=renderer, editor=editor)
    editor.load()
    server.start()
    try:
        status, scenes = request(server, "/api/scenes")
        assert status == 200
        assert [scene["name"] for scene in scenes] == ["Short", "Waiting"]
        assert request(server, "/api/clip/add", {"scene": "Short", "start": 4.0, "layer": 3})[0] == 200
        assert request(server, "/api/clip/remove", {"indices": [0]})[0] == 200
        assert [clip.layer for clip in editor.timeline.clips] == [0, 1, 3]
        waits = request(server, "/api/timeline")[1]["clips"][1]["waits"]
        assert waits == [{"event": "go", "from": 1.0, "time": None}]

        with urllib.request.urlopen(server.url + "api/audio") as response:
            assert response.status == 204  # these scenes make no sound
    finally:
        server.close()

    plain = PreviewServer(Timeline(), SETTINGS, renderer=renderer)
    assert plain.audio() is None
    plain.start()
    try:
        assert request(plain, "/api/scenes")[0] == 404
    finally:
        plain.close()


def test_the_page_gets_the_timelines_sound_as_wav(renderer) -> None:
    import io
    import wave

    import numpy as np
    from manimx.animation import Scene

    class Beep(Scene):
        def construct(self):
            self.play_wave(np.full(2400, 0.5, np.float32), 24_000)
            self.wait(1.0)

    timeline = Timeline()
    timeline.add(Beep().prepare(), start=0.5)
    server = PreviewServer(timeline, SETTINGS, renderer=renderer)
    sound = server.audio()
    assert server.audio() is sound  # mixed once per version
    with wave.open(io.BytesIO(sound)) as file:
        assert (file.getnchannels(), file.getframerate()) == (2, 24_000)
        assert file.getnframes() == pytest.approx(1.5 * 24_000, abs=2)
    server.close()
