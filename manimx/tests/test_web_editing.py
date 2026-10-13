"""Editing the previewed project live: ProjectEditor and the POST routes
of the preview server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from manimx.animation import Timeline
from manimx.cli import main
from manimx.drivers import StillRenderer
from manimx.drivers.offline import RenderSettings
from manimx.examples.navigation import config
from manimx.project import Project
from manimx.animation.timing import NEVER
from manimx.web import EditError, PreviewServer, ProjectEditor, ProjectReloader
from manimx.web.commands import AddClip, ClipEdit, EditClips, parse

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
        assert request(server, "/api/clip", {"clips": "all"})[1] == {"error": "/api/clip does not take {'clips': 'all'}"}
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


def test_every_edit_is_undone_and_redone(root, editor) -> None:
    clips, fps = list(editor.timeline.clips), editor.project.settings.output.fps
    assert editor.history() == {"undo": None, "redo": None}
    editor.edit_clip(0, {"start": 1.0, "cut": [0.5, None]})
    editor.set_event("go", 2.0)
    editor.set_setting(["output", "fps"], 30)
    editor.remove_clips([0, 2])
    index = editor.add_clip("Waiting", 4.0, layer=2)
    assert editor.history() == {"undo": "add Waiting", "redo": None}

    for _ in range(5):
        editor.undo()
    assert editor.timeline.clips == clips  # the very same clips, back in their places
    assert starts(editor) == [0.0, 2.0, 0.0]
    assert [entry.start for entry in editor.project.timeline] == [None, None, 0.0]  # following again
    assert (editor.project.events, editor.project.settings.output.fps, editor.timeline.clips[0].cut_in) == ({}, fps, None)
    assert editor.history() == {"undo": None, "redo": "cut a clip"}
    editor.undo()  # nothing left: nothing happens

    for _ in range(5):
        editor.redo()
    assert [clip.name for clip in editor.timeline.clips] == ["Short-2", "Waiting"]
    assert starts(editor) == [2.5, 4.0]
    assert editor.project.events == {"go": 2.0}
    assert editor.timeline.clips[index].layer == 2
    editor.edit_clip(0, {"layer": 3})
    assert editor.history()["redo"] is None  # a new edit drops what was undone


def test_the_steps_of_a_drag_are_undone_as_one(root, editor) -> None:
    for step in (0.5, 1.0, 1.5):
        editor.run(parse("/api/clip", {"index": 2, "start": step}), gesture="drag-1")
    editor.run(parse("/api/clip", {"index": 2, "start": 3.0}), gesture="drag-2")
    editor.undo()
    assert editor.timeline.clips[2].start == 1.5
    editor.undo()
    assert editor.timeline.clips[2].start == 0.0


def test_a_removed_clip_comes_back_with_its_scene_as_saved_now(root, editor) -> None:
    editor.remove_clips([0])
    write(root / "scenes" / "short.py", short_scene(3))
    editor.files_changed({root / "scenes" / "short.py"})
    editor.undo()
    assert [clip.end for clip in editor.timeline.clips] == [3.0, 6.0, NEVER]


def test_clips_are_split_at_a_time_and_duplicated(root, editor) -> None:
    with pytest.raises(EditError, match="not inside Short-2"):  # 1.5 s is in clip 0 only
        editor.run(parse("/api/clip/split", {"indices": [1, 0], "time": 1.5}))
    assert len(editor.timeline.clips) == 3 and editor.history()["undo"] is None
    editor.run(parse("/api/clip/split", {"indices": [0], "time": 0.5}))
    entries = editor.project.timeline
    assert [(entry.scene, entry.start, entry.cut) for entry in entries[:3]] == [
        ("Short", None, (None, 0.5)),
        ("Short", 0.5, (0.5, None)),
        ("Short", None, None),  # still follows: the second half ends where the clip ended
    ]
    assert [(clip.start, clip.end) for clip in editor.timeline.clips[:3]] == [(0.0, 0.5), (0.5, 2.0), (2.0, 4.0)]
    assert editor.history()["undo"] == "split a clip"

    editor.run(parse("/api/clip/duplicate", {"indices": [0]}))  # after the clips on its track
    copy = editor.timeline.clips[-1]
    assert (copy.start, copy.end, editor.project.timeline[-1].cut) == (4.0, 4.5, (None, 0.5))
    with pytest.raises(EditError, match="never fires"):
        editor.run(parse("/api/clip/duplicate", {"indices": [3]}))  # Waiting holds for "go"

    editor.undo()
    editor.undo()
    assert [(entry.start, entry.cut) for entry in editor.project.timeline] == [(None, None), (None, None), (0.0, None)]
    assert starts(editor) == [0.0, 2.0, 0.0]


def test_copied_clips_are_pasted_as_one_step(root, editor) -> None:
    clips = [{"scene": "Short", "start": 5.0, "layer": 2, "cut": [0.5, None], "speed": 2}, {"scene": "Waiting", "start": 6.0}]
    editor.run(parse("/api/clip/add", {"clips": clips}))
    pasted = editor.timeline.clips[3:]
    assert [(clip.start, clip.layer, clip.cut_in) for clip in pasted] == [(5.0, 2, 0.5), (6.0, 0, None)]
    assert pasted[0].end == pytest.approx(5.75)  # 1.5 s left of the scene at twice the speed
    assert editor.history()["undo"] == "paste 2 clips"
    editor.undo()
    assert len(editor.timeline.clips) == 3
    with pytest.raises(EditError, match="needs a scene"):
        parse("/api/clip/add", {"clips": [{"start": 1.0}]})


def tone(path: Path, seconds: float, rate: int = 8000) -> bytes:
    """A 440 Hz WAV file's bytes."""
    import io

    from manimx.animation import write_wav

    buffer = io.BytesIO()
    write_wav(buffer, 0.5 * np.sin(2 * np.pi * 440 * np.arange(round(seconds * rate)) / rate), rate)
    return buffer.getvalue()


def test_audio_clips_are_uploaded_placed_mixed_and_saved(root, editor) -> None:
    file = editor.save_asset("my voice!.wav", tone(root, 2.0))
    assert file == "assets/audio/my-voice.wav"
    assert editor.save_asset("my voice!.wav", b"") == "assets/audio/my-voice-2.wav"  # never overwritten
    with pytest.raises(EditError, match="not a sound file"):
        editor.save_asset("notes.txt", b"hello")

    editor.run(parse("/api/audio/add", {"file": file, "start": 5.0, "track": 1}))
    editor.run(parse("/api/audio/edit", {"index": 0, "cut": [0.5, 1.5], "volume": 0.5}))
    clip = editor.timeline.audio_clips[0]
    assert (clip.start, clip.end, clip.track) == (5.0, 6.0, 1)
    assert editor.timeline.duration == 6.0  # the audio goes on after the scenes
    mixed = editor.timeline.audio(0.0, 7.0, 8000)
    assert not mixed[: 5 * 8000 - 10].any() and not mixed[6 * 8000 + 10 :].any()
    assert 0.2 < np.abs(mixed[5 * 8000 : 6 * 8000]).max() <= 0.26  # half the tone's loudness
    assert editor.audio_clips()[0] | {"length": 0} == {
        "name": "my-voice", "file": file, "start": 5.0, "end": 6.0, "track": 1, "cut": [0.5, 1.5], "volume": 0.5, "length": 0,
    }
    peaks = editor.waveform(0, 0.0, 2.0, 4)
    assert len(peaks) == 4 and all(low < -0.4 and high > 0.4 for low, high in peaks)

    editor.save()
    project = Project.load(root)
    assert [(entry.file, entry.start, entry.track, entry.cut, entry.volume) for entry in project.audio] == [(file, 5.0, 1, (0.5, 1.5), 0.5)]
    assert project.build_timeline().audio_clips[0].end == 6.0

    editor.run(parse("/api/audio/remove", {"indices": [0]}))
    assert editor.timeline.audio_clips == [] and editor.history()["undo"] == "remove an audio clip"
    editor.undo()
    editor.undo()  # the edit
    assert (editor.timeline.audio_clips[0].cut_out, editor.project.audio[0].volume) == (None, 1.0)
    with pytest.raises(EditError, match="outside the project"):
        editor.run(parse("/api/audio/add", {"file": "../elsewhere.wav", "start": 0}))
    with pytest.raises(EditError, match="audio tracks start at 0"):
        parse("/api/audio/edit", {"index": 0, "track": -1})


def test_the_page_uploads_sound_files(root, renderer) -> None:
    editor = ProjectEditor(root, lambda project: SETTINGS, save_delay=60.0)
    server = PreviewServer(Timeline(), SETTINGS, renderer=renderer, editor=editor)
    editor.load()
    server.start()
    try:
        upload = urllib.request.Request(server.url + "api/asset?name=take.wav", data=tone(root, 1.0), method="POST")
        with urllib.request.urlopen(upload) as response:
            file = json.loads(response.read())["file"]
        assert request(server, "/api/audio/add", {"file": file, "start": 1.0})[0] == 200
        assert request(server, "/api/timeline")[1]["audio"][0]["end"] == 2.0
        assert len(request(server, "/api/waveform?index=0&from=0&to=1&n=8")[1]) == 8
        assert request(server, "/api/waveform?index=3&from=0&to=1&n=8")[0] == 400
        with urllib.request.urlopen(server.url + "api/audio") as response:
            assert response.status == 200  # the audio clip makes a sound
    finally:
        server.close()


def wait_for_render(server: PreviewServer, states: set[str], timeout: float = 180.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = request(server, "/api/render")[1]
        if status["state"] in states:
            return status
        time.sleep(0.2)
    raise AssertionError(f"the render did not end: {status}")


def test_the_page_renders_the_video_and_can_cancel(root, renderer) -> None:
    editor = ProjectEditor(root, lambda project: SETTINGS, save_delay=60.0)
    server = PreviewServer(Timeline(), SETTINGS, renderer=renderer, editor=editor)
    editor.load()
    for path, value in [(["output", "width"], 320), (["output", "height"], 180), (["output", "fps"], 10), (["performance", "workers"], 1)]:
        editor.set_setting(path, value)
    server.start()
    try:
        assert request(server, "/api/render")[1] == {"state": "idle"}
        assert request(server, "/api/render/video")[0] == 404
        assert request(server, "/api/render", {"start": 0, "end": 1.0})[0] == 200
        assert request(server, "/api/render", {})[1] == {"error": "a render is running already"}
        status = wait_for_render(server, {"done", "failed"})
        assert status["state"] == "done", status
        assert status["done"] == status["total"] == 11 and status["encoder"]
        with urllib.request.urlopen(server.url + "api/render/video") as response:
            assert response.headers["Content-Type"] == "video/mp4" and len(response.read()) > 0

        request(server, "/api/render", {"start": 0})
        assert request(server, "/api/render/cancel", {})[0] == 200
        status = wait_for_render(server, {"cancelled", "done", "failed"})
        assert status["state"] == "cancelled" and not Path(status.get("file") or root / "none").exists()
        assert request(server, "/api/render", {"start": "soon"})[0] == 400
    finally:
        server.close()


def test_commands_are_parsed_from_the_pages_requests() -> None:
    assert parse("/api/clip", {"clips": [{"index": 1, "layer": 2}]}) == EditClips((ClipEdit(1, (("layer", 2),)),))
    assert parse("/api/clip/add", {"scene": "Intro"}) == AddClip("Intro", 0.0, 0)
    assert parse("/api/event", {"name": "go", "time": None}).label == "remove go"
    for route, body, message in [
        ("/api/clip/remove", {"indices": [-1]}, "no clip -1"),
        ("/api/event", {"name": ""}, "does not take"),
        ("/api/setting", {"path": [1]}, "does not take"),
        ("/api/clips", {}, "no edit"),
    ]:
        with pytest.raises(EditError, match=message):
            parse(route, body)


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
        status, answer = request(server, "/api/clip/remove", {"indices": [0]})
        assert answer["history"] == {"undo": "remove a clip", "redo": None}
        assert [clip.layer for clip in editor.timeline.clips] == [0, 1, 3]
        assert request(server, "/api/undo", {})[1]["history"]["redo"] == "remove a clip"
        assert request(server, "/api/clip", {"index": 0, "start": 1.0, "gesture": "g"})[0] == 200
        assert request(server, "/api/clip", {"index": 0, "start": 2.0, "gesture": "g"})[0] == 200
        assert request(server, "/api/undo", {})[0] == 200
        assert editor.timeline.clips[0].start == 0.0
        assert request(server, "/api/redo", {})[0] == 200
        assert request(server, "/api/clip/remove", {"indices": [0]})[0] == 200
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
