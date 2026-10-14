"""The preview server: a timeline shown in a browser page, frame by frame.

    server = PreviewServer(timeline, settings)
    server.serve_forever()          # http://127.0.0.1:<port>/

The page (``static/``) draws the timeline's clips and segments and lets
the user scrub through it. Everything it shows comes from three kinds of
request, answered by the standard library's HTTP server:

    GET /                    the page and its static files
    GET /api/timeline        duration, frame rate, size, clips (with their
                             nodes, signals and animations, see tracks.py),
                             segments
    GET /api/frame?t=&w=     the frame at time t, w pixels wide (JPEG; PNG
                             with &format=png, WebP with &format=webp)
    GET /api/frame?f=&v=&s=&w=  frame number f; the answer's X-Version
                             says which version it shows, and when that
                             is v (and s this server's session), the
                             browser may keep it (the same URL is always
                             the same image)
    GET /api/thumbnail?clip=&scene=&at=&s=&w=  clip alone at its scene's
                             playing time `at`, while it shows the scene
                             with that uid (kept by the browser)
    GET /api/labels?t=       the index of every path in its group at time
                             t, with its name and place (see labels.py)
    GET /api/audio           the timeline's sounds mixed, as WAV (204
                             without sounds), for playback in the page
    GET /api/changes?since=  waits until the timeline's version passes
                             `since` (or 25 s), then answers the version
                             and the last build error

With a project editor (editing.py, given by ``manimx ui``) the page
edits the project live: an edit changes the timeline the next frame
shows, and ``manimx.json`` is saved shortly after (see editing.py):

    GET  /api/settings       every setting, its value and limits
    GET  /api/scenes         the project's scenes (for the sidebar)
    POST /api/clip           {"index", and any of "start", "layer",
                             "cut", "speed"}, or {"clips": [...]} of
                             those for clips moved together
    POST /api/clip/add       {"scene", "start", "layer", "cut", "speed"}: a
                             new clip, or {"clips": [...]} of those (paste)
    POST /api/clip/remove    {"indices": [...]}
    POST /api/clip/split     {"indices": [...], "time"}: each in two there
    POST /api/clip/duplicate {"indices": [...]}: a copy after each
    POST /api/event          {"name", "time"} (time null: remove it)
    POST /api/setting        {"path": [...], "value"}
    POST /api/audio/add      {"file", "start", "track", "cut", "volume"}: a
                             sound file of the project on an audio track
                             (or {"clips": [...]})
    POST /api/audio/remove   {"indices": [...]}
    POST /api/audio/edit     {"index", and any of "start", "track", "cut",
                             "volume"}, or {"clips": [...]} of those
    GET  /api/scripts        the project's scripts: text, parts, word
                             times, takes, where each part is placed
    POST /api/script         {"name", "text"}: write a script (new or
                             changed; its words keep their times), or
                             {"name", "delete": true}
    POST /api/script/words   {"script", "part", "words": [{"index",
                             "start", "end"}]}: time words by hand
    POST /api/script/take    {"script", "part", "file", "offset"}: a
                             recorded take of a part (uploaded first);
                             Whisper times its words, answers when done
    POST /api/asset?name=    the body is a sound file (dropped or recorded),
                             saved to assets/audio/; answers its "file"
    GET  /api/waveform?index=&from=&to=&n=  an audio clip's peaks
    POST /api/render         {"start", "end"}: render the video (saved
                             first; full quality, the chosen encoder) in
                             a process of its own; /api/render/cancel
    GET  /api/render         the render's state and progress (rendering.py)
    GET  /api/render/video   the last finished video
    POST /api/undo, /api/redo  undo or redo the last edit

Every edit is a command (commands.py) that can be undone; an edit may
carry a "gesture" id, and the steps of one drag (the same id) are undone
as one. Each answer gives the "history": what undo and redo would do.

The timeline can be replaced while serving (:meth:`PreviewServer.replace`,
used by reload.py when a scene file is saved); every replacement or
failed rebuild (:meth:`PreviewServer.report`) bumps the version, and the
page fetches the timeline again.

Design decisions:

* **Plain HTTP, the browser asks.** Scrubbing needs no pushes from the
  server: the page asks for one frame at a time and, while a request is
  open, only remembers the newest time it wants next. So the server never
  works through a backlog of outdated times, and it needs nothing beyond
  the standard library. Live presentation and editing may want pushes
  later; the endpoints stay the same.
* **One renderer, one frame at a time.** ``ThreadingHTTPServer`` answers
  requests on threads, but the renderer (a
  :class:`~manimx.drivers.StillRenderer`) is not thread-safe, so every
  frame is rendered under one lock. Static files and the timeline
  are answered without it.
* **Preview settings.** Frames come from the project's render settings
  without motion blur, scaled to the requested width (at most the output
  width), so a frame costs one render. A frame like the one before is
  held by the renderer and not rendered again.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import mimetypes
import secrets
import threading
from collections.abc import Callable
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from ..animation import write_wav
from ..animation.timing import NEVER, Clip, Timeline
from ..drivers.offline import RenderSettings
from ..drivers.stills import StillRenderer
from ..renderer import RenderSamples
from .commands import EditError, parse
from .editing import ProjectEditor
from .rendering import RenderJob
from .labels import index_labels
from .tracks import clip_nodes

MIN_WIDTH = 16
AUDIO_RATE = 24_000
"""Sample rate of the page's playback audio (half of a render's: smaller,
fine for previewing)."""
CHANGES_TIMEOUT = 25.0
"""Seconds a /api/changes request waits before answering anyway."""
MAX_ASSET = 512 * 2**20
"""The largest sound file the page may upload."""
IMMUTABLE = "private, max-age=86400, immutable"
"""Cache-Control of a frame asked for by number at the version it shows."""
EDIT_ROUTES = (
    "/api/clip",
    "/api/clip/add",
    "/api/clip/remove",
    "/api/clip/split",
    "/api/clip/duplicate",
    "/api/audio/add",
    "/api/audio/remove",
    "/api/audio/edit",
    "/api/script",
    "/api/script/words",
    "/api/event",
    "/api/setting",
    "/api/undo",
    "/api/redo",
)
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
        renderer: StillRenderer | None = None,
        editor: ProjectEditor | None = None,
    ) -> None:
        self.title = title
        self.session = secrets.token_hex(6)
        """Changes with every server: the page puts it into the URLs of the
        images the browser may keep, since versions and uids start over."""
        self.editor = editor
        """Changes the project when the page edits it (None: read only)."""
        self.render_job = RenderJob(editor.root) if editor is not None else None
        """Renders the project's video for the page (with an editor)."""
        if editor is not None:
            editor.attach(self)
        self._renderer = renderer or StillRenderer()
        self._lock = threading.Lock()
        """Held while rendering or replacing the timeline."""
        self._changed = threading.Condition()
        self.version = 0
        """Bumped by every replace() and report()."""
        self.error: str | None = None
        """The last failed rebuild, until the next successful one."""
        self._closing = False
        self._audio: tuple[int, bytes | None] | None = None
        """The last mixed audio and the version it is of."""
        self._use(timeline, settings)
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

    def replace(self, timeline: Timeline, settings: RenderSettings | None = None) -> None:
        """Show ``timeline`` (and ``settings``) from now on; clears the error."""
        with self._lock:  # bumped inside: a frame's version always says what it shows
            self._use(timeline, settings or self.settings)
            self._bump(error=None)

    def change(self, edit: Callable[[], None]) -> None:
        """Run ``edit`` on the timeline shown (between frames), then tell
        the page."""
        with self._lock:
            edit()
            self._bump(error=self.error)

    def report(self, error: str) -> None:
        """A rebuild failed: keep showing the current timeline, tell the page."""
        self._bump(error=error)

    def wait_for_change(self, since: int, timeout: float = CHANGES_TIMEOUT) -> tuple[int, str | None]:
        """Block until :attr:`version` passes ``since`` (or ``timeout``)."""
        with self._changed:
            self._changed.wait_for(lambda: self.version > since or self._closing, timeout)
            return self.version, self.error

    def _use(self, timeline: Timeline, settings: RenderSettings) -> None:
        self.timeline = timeline
        self.settings = settings
        self._frame_at = timeline.frame_at
        self._effects = tuple(effect for effect in settings.effects if not isinstance(effect, RenderSamples))

    def _bump(self, error: str | None) -> None:
        with self._changed:
            self.version += 1
            self.error = error
            self._changed.notify_all()

    def close(self) -> None:
        """Stop serving (from another thread) and release the port; a render
        running for the page is stopped."""
        if self.render_job is not None:
            self.render_job.cancel()
        with self._changed:
            self._closing = True  # open /api/changes requests answer now
            self._changed.notify_all()
        if self._serving:
            self._httpd.shutdown()  # waits for serve_forever to return
        self._httpd.server_close()

    # -- answers ------------------------------------------------------------

    def timeline_json(self) -> dict[str, Any]:
        """What the page draws: the timeline's length, clips and segments."""
        # The editor's lock is taken before this one when an edit runs: ask it first, not inside.
        history = self.editor.history() if self.editor is not None else None
        with self._lock:  # a Timeline resolves lazily: not while a frame renders
            return self._describe() | {"history": history}

    def _describe(self) -> dict[str, Any]:
        config = self.settings.config
        clips = list(self.timeline.clips)
        entries = self.editor.clips() if self.editor is not None else []
        if len(entries) != len(clips):
            entries = [None] * len(clips)  # not the project's timeline: nothing to edit
        known = self.timeline.event_times()
        fixed = self.timeline.events
        waited = sorted({segment.waits_for for clip in clips for segment in clip.scene.segments if segment.waits_for})
        return {
            "title": self.title,
            "version": self.version,
            "error": self.error,
            "editable": self.editor is not None,
            "session": self.session,
            "events": [{"name": name, "time": known.get(name), "fixed": name in fixed} for name in sorted({*waited, *fixed})],
            "duration": self.timeline.duration,
            "fps": self.settings.fps,
            "width": config.pixel_width,
            "height": config.pixel_height,
            "clips": [
                {
                    "name": clip.name,
                    "start": clip.start,
                    "end": None if clip.end == NEVER else clip.end,
                    "layer": clip.layer,
                    "scene": clip.scene.uid,
                    "length": clip.scene.playing_time,
                    "entry": entry,
                    "waits": _waits(clip, known),
                    "cues": _cues(clip, known),
                    "nodes": clip_nodes(clip),
                }
                for clip, entry in zip(clips, entries, strict=True)
            ],
            "audio": self.editor.audio_clips() if self.editor is not None else [],
            "transcriber": _transcriber(),
            "said": [{"name": name, "time": time} for name, time in sorted(self.timeline.word_events().items(), key=lambda item: item[1])],
            "subtitles": [{"start": item.start, "end": item.end, "text": item.text} for item in self.timeline.subtitles()],
            "segments": [
                {"clip": clips.index(clip), "index": index, "start": start, "end": None if end == NEVER else end}
                for clip, index, start, end in self.timeline.segments()
            ],
        }

    def audio(self) -> bytes | None:
        """The timeline's sounds as WAV (mixed once per version), or None
        without sounds."""
        with self._lock:
            if self._audio is not None and self._audio[0] == self.version:
                return self._audio[1]
            timeline = self.timeline
            if not timeline.audio_clips and not any(clip.placed_sounds() for clip in timeline.clips):
                self._audio = (self.version, None)
                return None
            samples = timeline.audio(0.0, timeline.duration, AUDIO_RATE)
            buffer = io.BytesIO()
            write_wav(buffer, samples, AUDIO_RATE)
            self._audio = (self.version, buffer.getvalue())
            return self._audio[1]

    def labels(self, time: float) -> list[dict[str, Any]]:
        """The index labels of the frame at ``time`` (see labels.py)."""
        with self._lock:
            time = max(float(time), 0.0)  # past the end: the background, nothing plays there
            return index_labels(self.timeline, time, self.settings.config.aspect_ratio)

    def frame(self, time: float, width: int | None = None, image_format: str = "jpeg") -> bytes:
        """The frame at ``time`` (0 or later; past the end, the background), ``width`` pixels
        wide (default and at most the output width), encoded.
        """
        return self.versioned_frame(time, width, image_format)[0]

    def versioned_frame(
        self, time: float | None, width: int | None = None, image_format: str = "jpeg", frame: int | None = None
    ) -> tuple[bytes, int]:
        """:meth:`frame`, at ``time`` or at frame number ``frame``, and the
        version it shows (a frame is the same for the same version)."""
        if image_format not in FORMATS:
            raise ValueError(f"unknown image format {image_format!r}; use one of {', '.join(FORMATS)}")
        with self._lock:
            config = self.settings.config
            full_width, full_height = config.pixel_width, config.pixel_height
            width = full_width if width is None else max(MIN_WIDTH, min(int(width), full_width))
            height = max(1, round(width * full_height / full_width))
            version = self.version
            time = frame / self.settings.fps if frame is not None else time
            time = max(float(time), 0.0)  # past the end: the background, nothing plays there
            frame_config = dataclasses.replace(config, pixel_width=width, pixel_height=height, time=time)
            pixels = self._renderer.render(self._frame_at, frame_config, self._effects)
        return _encode(pixels, image_format), version

    def thumbnail(self, clip: int, scene: int, at: float, width: int) -> bytes | None:
        """Clip ``clip`` alone at its scene's playing time ``at``, ``width``
        pixels wide (JPEG); None unless the clip still shows the prepared
        scene with uid ``scene``. Where a clip is placed does not change its
        thumbnails, so the page keeps them while clips move."""
        with self._lock:
            clips = self.timeline.clips
            if not 0 <= clip < len(clips) or clips[clip].scene.uid != scene:
                return None
            shown = clips[clip]
            config = self.settings.config
            width = max(MIN_WIDTH, min(int(width), config.pixel_width))
            height = max(1, round(width * config.pixel_height / config.pixel_width))
            at = min(max(float(at), 0.0), shown.scene.playing_time)
            compiled = shown.scene.compiled_time_at(at)
            frame_config = dataclasses.replace(config, pixel_width=width, pixel_height=height, time=at)
            pixels = self._renderer.render(lambda _: (shown.scene.plan.frame_at(compiled, global_time=at, uid=shown.uid),), frame_config, ())
        return _encode(pixels, "jpeg")


def _encode(pixels: Any, image_format: str) -> bytes:
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
            "/api/labels": self._labels,
            "/api/thumbnail": self._thumbnail,
            "/api/waveform": self._waveform,
            "/api/changes": self._changes,
            "/api/settings": self._settings,
            "/api/scenes": self._scenes,
            "/api/scripts": self._scripts,
            "/api/audio": self._audio,
            "/api/render": self._render_status,
            "/api/render/video": self._render_video,
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

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        editor = self.preview.editor
        route = urlparse(self.path).path
        if route == "/api/asset" and editor is not None:
            self._asset(editor)
            return
        if route in ("/api/render", "/api/render/cancel") and editor is not None:
            self._render(editor, route)
            return
        if route == "/api/script/take" and editor is not None:
            self._take(editor)
            return
        if route not in EDIT_ROUTES:
            self._json({"error": f"not found: {self.path}"}, HTTPStatus.NOT_FOUND)
            return
        if editor is None:
            self._json({"error": "this preview cannot edit its project"}, HTTPStatus.FORBIDDEN)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as error:
                raise _BadRequest(f"the body is not JSON: {error}") from None
            if not isinstance(body, dict):
                raise _BadRequest("the body must be a JSON object")
            match route:
                case "/api/undo":
                    editor.undo()
                case "/api/redo":
                    editor.redo()
                case _:
                    gesture = body.pop("gesture", None)
                    editor.run(parse(route, body), gesture if isinstance(gesture, str) else None)
            self._json({"ok": True, "version": self.preview.version, "history": editor.history()})
        except (_BadRequest, EditError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _asset(self, editor: ProjectEditor) -> None:
        """POST /api/asset?name=: the body is a sound file to save in the project."""
        try:
            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_ASSET:
                raise _BadRequest(f"a sound file must be between 1 byte and {MAX_ASSET // 2**20} MB")
            self._json({"file": editor.save_asset(name, self.rfile.read(length))})
        except (_BadRequest, EditError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _take(self, editor: ProjectEditor) -> None:
        """POST /api/script/take {"script", "part", "file", "offset"}: a
        recorded take of a script part; Whisper times its words (seconds)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise _BadRequest("the body must be a JSON object")
            script, part, file, offset = body.get("script"), body.get("part"), body.get("file"), body.get("offset", 0.0)
            if not (isinstance(script, str) and isinstance(part, int) and isinstance(file, str) and isinstance(offset, (int, float))):
                raise _BadRequest('a take is {"script", "part", "file", "offset"}')
            problem = editor.record_take(script, part, file, float(offset))
            self._json({"ok": True, "version": self.preview.version, "history": editor.history(), "problem": problem})
        except (_BadRequest, EditError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server's signature
        pass  # one request per scrub step: too many to print

    # -- routes -------------------------------------------------------------

    def _timeline(self, query: dict[str, list[str]]) -> None:
        self._json(self.preview.timeline_json())

    def _frame(self, query: dict[str, list[str]]) -> None:
        frame = _number(query, "f", int, required=False)
        time = None if frame is not None else _number(query, "t", float)
        width = _number(query, "w", int, required=False)
        asked = _number(query, "v", int, required=False)
        image_format = query.get("format", ["jpeg"])[0]
        if image_format not in FORMATS:
            raise _BadRequest(f"unknown format {image_format!r}")
        body, version = self.preview.versioned_frame(time, width, image_format, frame)
        # A frame number at the version it shows is always the same image:
        # the browser keeps it. Anything else may change with the next edit.
        same_server = query.get("s", [None])[0] == self.preview.session
        cache = IMMUTABLE if frame is not None and asked == version and same_server else False
        self._send(body, FORMATS[image_format][1], cache=cache, headers={"X-Version": str(version)})

    def _waveform(self, query: dict[str, list[str]]) -> None:
        editor = self.preview.editor
        if editor is None:
            raise _BadRequest("this preview has no audio clips")
        index, buckets = _number(query, "index", int), _number(query, "n", int)
        start, end = _number(query, "from", float), _number(query, "to", float)
        try:
            self._json(editor.waveform(index, start, end, buckets))
        except EditError as error:
            raise _BadRequest(str(error)) from None

    def _thumbnail(self, query: dict[str, list[str]]) -> None:
        clip, scene, width = (_number(query, name, int) for name in ("clip", "scene", "w"))
        body = self.preview.thumbnail(clip, scene, _number(query, "at", float), width)
        if body is None:
            raise _BadRequest(f"clip {clip} does not show scene {scene} (any more)")
        same_server = query.get("s", [None])[0] == self.preview.session
        self._send(body, "image/jpeg", cache=IMMUTABLE if same_server else False)

    def _labels(self, query: dict[str, list[str]]) -> None:
        self._json(self.preview.labels(_number(query, "t", float)))

    def _settings(self, query: dict[str, list[str]]) -> None:
        if self.preview.editor is None:
            self._json({"error": "this preview has no project settings"}, HTTPStatus.NOT_FOUND)
            return
        self._json(self.preview.editor.settings())

    def _render(self, editor: ProjectEditor, route: str) -> None:
        """POST /api/render {"start", "end"} starts a render, /api/render/cancel stops it."""
        job = self.preview.render_job
        assert job is not None
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if route == "/api/render/cancel":
                job.cancel()
            else:
                start, end = body.get("start") or 0.0, body.get("end")
                if not isinstance(start, (int, float)) or not (end is None or isinstance(end, (int, float))):
                    raise _BadRequest("start and end are seconds")
                editor.save()  # the render reads manimx.json
                job.start(float(start), None if end is None else float(end))
            self._json(job.status())
        except (_BadRequest, EditError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _render_status(self, query: dict[str, list[str]]) -> None:
        if self.preview.render_job is None:
            raise _BadRequest("this preview has no project to render")
        self._json(self.preview.render_job.status())

    def _render_video(self, query: dict[str, list[str]]) -> None:
        video = self.preview.render_job.video() if self.preview.render_job is not None else None
        if video is None:
            self._json({"error": "no finished render"}, HTTPStatus.NOT_FOUND)
            return
        self._send(video.read_bytes(), "video/mp4", cache=False, headers={"Content-Disposition": f'inline; filename="{video.name}"'})

    def _scenes(self, query: dict[str, list[str]]) -> None:
        if self.preview.editor is None:
            self._json({"error": "this preview has no project"}, HTTPStatus.NOT_FOUND)
            return
        self._json(self.preview.editor.scenes())

    def _scripts(self, query: dict[str, list[str]]) -> None:
        if self.preview.editor is None:
            self._json({"error": "this preview has no project"}, HTTPStatus.NOT_FOUND)
            return
        self._json(self.preview.editor.scripts())

    def _audio(self, query: dict[str, list[str]]) -> None:
        sound = self.preview.audio()
        if sound is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send(sound, "audio/wav", cache=False)

    def _changes(self, query: dict[str, list[str]]) -> None:
        version, error = self.preview.wait_for_change(_number(query, "since", int))
        self._json({"version": version, "error": error})

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

    def _send(
        self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK, cache: bool | str = True, headers: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache is False:
            self.send_header("Cache-Control", "no-store")
        elif isinstance(cache, str):
            self.send_header("Cache-Control", cache)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


class _BadRequest(ValueError):
    pass


def _waits(clip: Clip, known: dict[str, float]) -> list[dict[str, Any]]:
    """Every wait of ``clip`` inside its cut: the event, where the clip
    starts waiting (the end of the segment before, None if that is not
    placed) and the event's time (None while it never fires)."""
    waits = []
    previous_end: float | None = None
    for segment, timing in zip(clip.scene.segments, clip.timings, strict=True):
        if clip._compiled_range(segment) is None:
            continue
        if segment.waits_for:
            waits.append({"event": segment.waits_for, "from": previous_end, "time": known.get(segment.waits_for)})
        previous_end = None if timing is None else timing.end
    return waits


def _transcriber() -> bool:
    """Whether Whisper (the optional faster-whisper) can transcribe here."""
    return importlib.util.find_spec("faster_whisper") is not None


def _cues(clip: Clip, known: dict[str, float]) -> list[dict[str, Any]]:
    """Every cue of ``clip``'s scene inside its cut: its name, the event
    it syncs to and that event's time (None while it never fires)."""
    cues = []
    for segment in clip.scene.segments:
        compiled = clip._compiled_range(segment)
        for cue in segment.cues:
            if compiled is not None and compiled[0] <= segment.offset + cue.time <= compiled[1]:
                event = clip.event_of(cue.name)
                cues.append({"name": cue.name, "event": event, "time": known.get(event)})
    return cues


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
