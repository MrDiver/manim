"""Frame sinks: where rendered frames go.

A sink receives frames through :meth:`FrameSink.write`, as ``(H, W, 4)``
uint8 RGBA arrays, or as NV12 frames when it asks for them
(:meth:`FrameSink.pixel_format`; see :class:`FrameFormat`). Drivers call it from one writer thread, separate
from rendering, so a slow encoder or disk never stalls the GPU pipeline
(until the bounded queue in front of it fills up: backpressure instead of
unbounded memory).

``ordered`` sinks (a video stream) get frames strictly in index order;
the driver reorders frames that finish out of order. Unordered sinks
(image files) get them as soon as they are done.

The array passed to ``write`` is only valid during the call (it may be a
view into shared memory); copy it to keep it.
"""

from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from ..animation.audio import write_wav
from .encoders import X264, H264Encoder


PixelFormat = Literal["rgba", "nv12"]
"""How a frame's pixels arrive: "rgba", an ``(H, W, 4)`` uint8 array of
sRGB values with straight alpha; "nv12", an ``(H * 3 // 2, W)`` uint8
array of the Y'CbCr 4:2:0 planes most video encoders take: ``H`` rows of
luma, then ``H // 2`` rows of interleaved (Cb, Cr) pairs, one per 2x2
pixels, in BT.709 limited range and without alpha. The GPU converts to
NV12, so a third of the bytes are read back and the encoder converts
nothing."""


@dataclass(frozen=True, slots=True, eq=False)
class FrameFormat:
    width: int
    height: int
    fps: float
    audio: np.ndarray | None = None
    """Stereo float32 samples for exactly the rendered frames, or None."""
    sample_rate: int = 48_000
    pixel_format: PixelFormat = "rgba"
    """What :meth:`FrameSink.write` gets, as the sink asked for it."""


class FrameSink:
    """Base class: override :meth:`write`, optionally :meth:`open` and :meth:`close`."""

    ordered: bool = True

    def pixel_format(self, width: int, height: int) -> PixelFormat:
        """The pixels this sink wants for frames of this size (asked
        before :meth:`open`; the format it is opened with says what it gets)."""
        return "rgba"

    def open(self, format: FrameFormat) -> None:
        """Called once before the first frame."""

    def write(self, index: int, image: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Called once after the last frame (also after errors)."""


@dataclass(eq=False)
class FrameList(FrameSink):
    """Keeps every frame in memory (tests, previews, thumbnails)."""

    frames: list[np.ndarray] = field(default_factory=list)

    def write(self, index: int, image: np.ndarray) -> None:
        self.frames.append(image.copy())


BT709 = ("-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709")
"""The color tags of NV12 frames (see :data:`PixelFormat`), given to
ffmpeg for its input and its output."""


@dataclass(eq=False)
class VideoFile(FrameSink):
    """Encodes frames into a video file with ffmpeg (from ``PATH``).

    The defaults give H.264 in yuv420p from x264, which plays everywhere;
    H.264 is encoded from NV12 frames, converted on the GPU (BT.709, tagged
    so in the file), when the size is even;
    ``encoder`` uses another H.264 encoder, e.g. the GPU's (see
    encoders.py: ``choose_encoder("auto")``). ``codec_args`` replaces the
    codec settings, e.g. ``["-c:v", "prores_ks", "-profile:v", "4444",
    "-pix_fmt", "yuva444p10le"]`` for a .mov with alpha.
    """

    path: str | Path
    codec_args: list[str] = field(default_factory=lambda: list(X264.codec_args))
    encoder: H264Encoder | None = None
    """An H.264 encoder: its codec settings replace ``codec_args``."""
    audio_args: list[str] = field(default_factory=lambda: ["-c:a", "aac", "-b:a", "192k"])
    ffmpeg: str = "ffmpeg"
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _audio_directory: tempfile.TemporaryDirectory | None = field(default=None, init=False, repr=False)

    def pixel_format(self, width: int, height: int) -> PixelFormat:
        h264 = self.encoder is not None or self.codec_args == list(X264.codec_args)
        return "nv12" if h264 and width % 2 == 0 and height % 2 == 0 else "rgba"

    def open(self, format: FrameFormat) -> None:
        encoder = self.encoder if self.encoder is None or self.encoder.fits(format.width, format.height) else X264
        input_args, codec_args = (encoder.input_args, encoder.codec_args) if encoder else ((), self.codec_args)
        colors = BT709 if format.pixel_format == "nv12" else ()
        command = [
            self.ffmpeg, "-y", "-loglevel", "error", *input_args,
            "-f", "rawvideo", "-pix_fmt", format.pixel_format, *colors, "-s", f"{format.width}x{format.height}", "-r", str(format.fps),
            "-i", "-",
        ]  # fmt: skip
        if format.audio is not None:
            # The mix is known before the first frame, so ffmpeg muxes it
            # in the same pass; a temporary WAV keeps stdin for video.
            self._audio_directory = tempfile.TemporaryDirectory(prefix="manimx-audio-")
            audio_path = Path(self._audio_directory.name) / "audio.wav"
            write_wav(audio_path, format.audio, format.sample_rate)
            command += ["-i", str(audio_path), "-map", "0:v", "-map", "1:a", *self.audio_args]
        command += [*codec_args, *colors, str(self.path)]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, index: int, image: np.ndarray) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(memoryview(np.ascontiguousarray(image)).cast("B"))

    def close(self) -> None:
        if self._process is None:
            return
        assert self._process.stdin is not None and self._process.stderr is not None
        self._process.stdin.close()
        errors = self._process.stderr.read().decode(errors="replace")
        code = self._process.wait()
        self._process = None
        if self._audio_directory is not None:
            self._audio_directory.cleanup()
            self._audio_directory = None
        if code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {code}: {errors.strip()}")


@dataclass(eq=False)
class Window(FrameSink):
    """Shows frames in a window as they arrive, with ffplay (from ``PATH``).

    ffplay plays the stream at the format's frame rate, so rendering
    faster than real time is held back to real time (the pipe fills up);
    slower rendering plays as fast as frames come. Sounds are not played.
    Closing the window stops the render.
    """

    title: str = "manimx"
    ffplay: str = "ffplay"
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def open(self, format: FrameFormat) -> None:
        command = [
            self.ffplay, "-loglevel", "error", "-autoexit", "-window_title", self.title,
            "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{format.width}x{format.height}",
            "-framerate", str(format.fps), "-i", "-",
        ]  # fmt: skip
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, index: int, image: np.ndarray) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(memoryview(np.ascontiguousarray(image)).cast("B"))
        except BrokenPipeError:
            raise RuntimeError("the window was closed") from None

    def close(self) -> None:
        if self._process is None:
            return
        assert self._process.stdin is not None
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass  # closed by the user
        self._process.wait()
        self._process = None


@dataclass(eq=False)
class Tee(FrameSink):
    """Passes every frame to several sinks, e.g. a video file and a window.

    Frames arrive in order if any of the sinks needs them in order.
    """

    sinks: tuple[FrameSink, ...]

    def __post_init__(self) -> None:
        self.ordered = any(sink.ordered for sink in self.sinks)

    def pixel_format(self, width: int, height: int) -> PixelFormat:
        """What every sink wants, or RGBA when they differ."""
        wanted = {sink.pixel_format(width, height) for sink in self.sinks}
        return wanted.pop() if len(wanted) == 1 else "rgba"

    def open(self, format: FrameFormat) -> None:
        for sink in self.sinks:
            sink.open(format)

    def write(self, index: int, image: np.ndarray) -> None:
        for sink in self.sinks:
            sink.write(index, image)

    def close(self) -> None:
        errors = []
        for sink in self.sinks:  # close every sink, even after an error
            try:
                sink.close()
            except Exception as error:  # noqa: BLE001 - the first one is raised below
                errors.append(error)
        if errors:
            raise errors[0]


@dataclass(eq=False)
class ImageSequence(FrameSink):
    """Writes one image file per frame, ``pattern.format(index)`` inside
    ``directory``; PNG encoding runs on ``threads`` threads (zlib releases
    the GIL, so they really run in parallel)."""

    directory: str | Path
    pattern: str = "frame_{:05d}.png"
    threads: int = 4
    ordered = False
    _pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _writes: list[Future] = field(default_factory=list, init=False, repr=False)

    audio_name: str = "audio.wav"
    """Written next to the frames when the source has sounds."""

    def open(self, format: FrameFormat) -> None:
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        if format.audio is not None:
            write_wav(Path(self.directory) / self.audio_name, format.audio, format.sample_rate)
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="png")

    def write(self, index: int, image: np.ndarray) -> None:
        assert self._pool is not None
        path = Path(self.directory) / self.pattern.format(index)
        self._writes.append(self._pool.submit(Image.fromarray(image.copy(), "RGBA").save, path))

    def close(self) -> None:
        if self._pool is None:
            return
        self._pool.shutdown(wait=True)
        self._pool = None
        writes, self._writes = self._writes, []
        for write in writes:
            write.result()  # raise the first write error, if any
