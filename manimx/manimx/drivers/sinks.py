"""Frame sinks: where rendered frames go.

A sink receives frames as ``(H, W, 4)`` uint8 RGBA arrays through
:meth:`FrameSink.write`. Drivers call it from one writer thread, separate
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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class FrameFormat:
    width: int
    height: int
    fps: float


class FrameSink:
    """Base class: override :meth:`write`, optionally :meth:`open` and :meth:`close`."""

    ordered: bool = True

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


@dataclass(eq=False)
class VideoFile(FrameSink):
    """Encodes frames into a video file with ffmpeg (from ``PATH``).

    The defaults give H.264 in yuv420p, which plays everywhere.
    ``codec_args`` replaces the codec settings, e.g. ``["-c:v",
    "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]`` for
    a .mov with alpha.
    """

    path: str | Path
    codec_args: list[str] = field(
        default_factory=lambda: ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    )
    ffmpeg: str = "ffmpeg"
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def open(self, format: FrameFormat) -> None:
        command = [
            self.ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{format.width}x{format.height}", "-r", str(format.fps),
            "-i", "-", *self.codec_args, str(self.path),
        ]  # fmt: skip
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
        if code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {code}: {errors.strip()}")


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

    def open(self, format: FrameFormat) -> None:
        Path(self.directory).mkdir(parents=True, exist_ok=True)
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
