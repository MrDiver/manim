"""The browser preview: the server's page, timeline and frames."""

from __future__ import annotations

import dataclasses
import io
import json
import socket
import urllib.error
import urllib.request

import numpy as np
import pytest
from manimx.animation import Timeline
from manimx.cli import main
from manimx.drivers.offline import RenderSettings
from manimx.examples.navigation import Navigation, config
from manimx.renderer import FrameComposer, MotionBlur, RenderService
from manimx.web import PreviewServer
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

    monkeypatch.setattr(server.timeline, "frame_at", broken)
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

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        assert main(["ui", "--port", str(taken.getsockname()[1]), "--no-open"]) == 1
    assert "cannot serve on port" in capsys.readouterr().err
