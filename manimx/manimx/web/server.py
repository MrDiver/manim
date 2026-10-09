"""The preview server: a timeline shown in a browser page, frame by frame.

    server = PreviewServer(timeline, settings)
    server.serve_forever()          # http://127.0.0.1:<port>/

The page (``static/``) draws the timeline's clips and segments and lets
the user scrub through it. Everything it shows comes from three kinds of
request, answered by the standard library's HTTP server:

    GET /                    the page and its static files
    GET /api/timeline        duration, frame rate, size, clips, segments
    GET /api/frame?t=&w=     the frame at time t, w pixels wide (JPEG; PNG
                             with &format=png, WebP with &format=webp)

Design decisions:

* **Plain HTTP, the browser asks.** Scrubbing needs no pushes from the
  server: the page asks for one frame at a time and, while a request is
  open, only remembers the newest time it wants next. So the server never
  works through a backlog of outdated times, and it needs nothing beyond
  the standard library. Live presentation and editing may want pushes
  later; the endpoints stay the same.
* **One renderer, one frame at a time.** ``ThreadingHTTPServer`` answers
  requests on threads, but the render service is not thread-safe, so
  every frame is rendered under one lock. Static files and the timeline
  are answered without it.
* **Preview settings.** Frames come from the project's render settings
  without motion blur, scaled to the requested width (at most the output
  width), so a frame costs one render. A frame like the one before is
  held by the composer and not rendered again.
"""

from __future__ import annotations

import dataclasses
import io
import json
import mimetypes
import threading
from collections.abc import Callable
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from ..animation.timing import NEVER, Timeline
from ..drivers.offline import RenderSettings
from ..renderer import MotionBlur, RenderService

MIN_WIDTH = 16
FORMATS = {"jpeg": ("JPEG", "image/jpeg"), "png": ("PNG", "image/png"), "webp": ("WEBP", "image/webp")}


class PreviewServer:
    """Serves ``timeline`` to the preview page (see the module docs)."""

    def __init__(
        self,
        timeline: Timeline,
        settings: RenderSettings,
        title: str = "manimx",
        host: str = "127.0.0.1",
        port: int = 0,
        service: RenderService | None = None,
    ) -> None:
        self.timeline = timeline
        self.settings = settings
        self.title = title
        self._service = service or RenderService.shared()
        effects = tuple(effect for effect in settings.effects if not isinstance(effect, MotionBlur))
        self._composer = self._service.composer(effects)
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), partial(_Handler, preview=self))
        self._httpd.daemon_threads = True
        self._serving = False

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def serve_forever(self) -> None:
        self._serving = True
        try:
            self._httpd.serve_forever()
        finally:
            self._serving = False

    def start(self) -> threading.Thread:
        """Serve on a background thread (for tests and embedding)."""
        thread = threading.Thread(target=self.serve_forever, name="manimx preview", daemon=True)
        self._serving = True  # before the thread runs, so close() right after waits for it
        thread.start()
        return thread

    def close(self) -> None:
        """Stop serving (from another thread) and release the port."""
        if self._serving:
            self._httpd.shutdown()  # waits for serve_forever to return
        self._httpd.server_close()

    # -- answers ------------------------------------------------------------

    def timeline_json(self) -> dict[str, Any]:
        """What the page draws: the timeline's length, clips and segments."""
        config = self.settings.config
        clips = list(self.timeline.clips)
        return {
            "title": self.title,
            "duration": self.timeline.duration,
            "fps": self.settings.fps,
            "width": config.pixel_width,
            "height": config.pixel_height,
            "clips": [
                {"name": clip.name, "start": clip.start, "end": None if clip.end == NEVER else clip.end, "layer": clip.layer}
                for clip in clips
            ],
            "segments": [
                {"clip": clips.index(clip), "index": index, "start": start, "end": None if end == NEVER else end}
                for clip, index, start, end in self.timeline.segments()
            ],
        }

    def frame(self, time: float, width: int | None = None, image_format: str = "jpeg") -> bytes:
        """The frame at ``time`` (clamped to the timeline), ``width`` pixels
        wide (default and at most the output width), encoded.
        """
        if image_format not in FORMATS:
            raise ValueError(f"unknown image format {image_format!r}; use one of {', '.join(FORMATS)}")
        config = self.settings.config
        full_width, full_height = config.pixel_width, config.pixel_height
        width = full_width if width is None else max(MIN_WIDTH, min(int(width), full_width))
        height = max(1, round(width * full_height / full_width))
        time = min(max(float(time), 0.0), self.timeline.duration)
        frame_config = dataclasses.replace(config, pixel_width=width, pixel_height=height, time=time)
        with self._lock:
            pixels = self._composer.compose(self.timeline.frame_at, frame_config)
        pil_format, _ = FORMATS[image_format]
        image = Image.fromarray(pixels, "RGBA")
        if pil_format == "JPEG":
            image = image.convert("RGB")  # JPEG has no alpha
        buffer = io.BytesIO()
        image.save(buffer, pil_format, **({"quality": 85} if pil_format != "PNG" else {}))
        return buffer.getvalue()


class _Handler(BaseHTTPRequestHandler):
    """Routes the page's requests to its :class:`PreviewServer`.

    HTTP/1.1 keeps the browser's connection open between frames, and
    without Nagle's algorithm a response's last packet leaves at once
    instead of waiting for the previous one's acknowledgement.
    """

    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def __init__(self, *args: Any, preview: PreviewServer, **kwargs: Any) -> None:
        self.preview = preview
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
        url = urlparse(self.path)
        routes: dict[str, Callable[[dict[str, list[str]]], None]] = {
            "/api/timeline": self._timeline,
            "/api/frame": self._frame,
        }
        route = routes.get(url.path)
        try:
            if route is not None:
                route(parse_qs(url.query))
            else:
                self._static(url.path)
        except _BadRequest as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # a scene that fails at this time: show it in the page
            self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server's signature
        pass  # one request per scrub step: too many to print

    # -- routes -------------------------------------------------------------

    def _timeline(self, query: dict[str, list[str]]) -> None:
        self._json(self.preview.timeline_json())

    def _frame(self, query: dict[str, list[str]]) -> None:
        time = _number(query, "t", float)
        width = _number(query, "w", int, required=False)
        image_format = query.get("format", ["jpeg"])[0]
        if image_format not in FORMATS:
            raise _BadRequest(f"unknown format {image_format!r}")
        self._send(self.preview.frame(time, width, image_format), FORMATS[image_format][1], cache=False)

    def _static(self, path: str) -> None:
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        static = resources.files(__package__) / "static"
        target = static / name
        if "/" in name or ".." in name or not target.is_file():
            self._json({"error": f"not found: {path}"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self._send(target.read_bytes(), content_type, cache=False)

    # -- responses ----------------------------------------------------------

    def _json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(data).encode(), "application/json", status=status, cache=False)

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK, cache: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _BadRequest(ValueError):
    pass


def _number(query: dict[str, list[str]], name: str, kind: type, required: bool = True) -> Any:
    values = query.get(name)
    if not values:
        if required:
            raise _BadRequest(f"missing parameter {name!r}")
        return None
    try:
        return kind(values[0])
    except ValueError:
        raise _BadRequest(f"parameter {name!r} must be a {kind.__name__}, got {values[0]!r}") from None


__all__ = ["PreviewServer"]
