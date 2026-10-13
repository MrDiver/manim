"""The offline render driver: every frame of a source into a sink, as fast
as the machine allows.

    render("manimx.examples.navigation:Navigation", VideoFile("nav.mp4"),
           RenderSettings(RenderConfig(), fps=60), workers=4)

Rendering a frame has three parts that can all run at the same time:
evaluating the scene and encoding GPU commands (CPU), drawing (GPU), and
writing the result (an encoder process or PNG threads). The driver
overlaps them on three levels:

1. **In a process**, a :class:`~.core_pipeline.CoreFramePipeline` keeps
   several frames in flight, so the CPU prepares the next frames while
   the GPU draws the earlier ones.
2. **Writing** happens on a separate thread behind a bounded queue, so a
   slow sink only slows rendering once the queue is full.
3. **Across processes** (``workers > 1``), each worker prepares its own
   copy of the source and renders chunks of consecutive frames with its
   own GPU device. This scales the Python part, which a single process
   cannot (the GIL, for the Python a scene runs per frame). It is
   correct only because evaluation is pure:
   ``frame_at(t)`` gives the same frame in every process (rule 4 of the
   render contract). Consecutive frames per chunk keep each worker's
   renderer caches warm.

Workers hand frames to the main process through shared memory, in slots
owned per worker. A worker renders its frames in increasing order and
only waits for its own slots, so in-order writing can never deadlock:
the lowest missing frame always belongs to a worker whose held slots
contain only lower, already writable frames.

Sources are given as ``"module:attr"`` (or any picklable callable) so
worker processes can rebuild them. ``attr`` may be a Scene class (it is
prepared), a factory function, a prepared scene, a timeline or a
:class:`FrameSource`.
"""

from __future__ import annotations

import importlib
import math
import multiprocessing
import pickle
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from ..renderer.config import Frame, RenderConfig
from ..renderer.effects import Effect
from ..voice.transcript import Subtitle
from .sinks import FrameFormat, FrameSink, PixelFormat

Progress = Callable[[int, int], None]
"""``progress(frames written, total frames)``, called from the writer thread."""


@dataclass(frozen=True, eq=False)
class FrameSource:
    """Something to render: a function of time and how long it lasts."""

    frame_at: Callable[[float], Frame]
    """What to render at a time: the views of a timeline or prepared scene
    (``frame_at``)."""
    duration: float
    audio: Callable[[float, float, int], np.ndarray] | None = None
    """``audio(start, end, sample_rate)``: stereo float32 samples, if any."""
    subtitles: Callable[[], list[Subtitle]] | None = None
    """``subtitles()``: what is said when (a timeline's transcripts), if anything."""


@dataclass(frozen=True)
class RenderSettings:
    config: RenderConfig = field(default_factory=RenderConfig)
    fps: float = 60.0
    effects: tuple[Effect, ...] = ()
    """Compositor effects (motion blur, bloom, tone mapping, shaders)."""
    start: float = 0.0
    end: float | None = None
    """Last time to render (default: the source's duration)."""
    in_flight: int = 3
    """Frames on the GPU at once, per process."""
    chunk: int = 8
    """Consecutive frames a worker process renders at a time."""
    sample_rate: int = 48_000
    audio: bool = True
    """Pass the source's sounds to the sink (if it has any)."""

    def frame_times(self, duration: float) -> list[float]:
        end = duration if self.end is None else self.end
        count = math.floor((end - self.start) * self.fps + 1e-9) + 1
        return [self.start + index / self.fps for index in range(max(count, 0))]


@dataclass(frozen=True)
class RenderStats:
    frames: int
    seconds: float

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else math.inf


def load_source(spec: Any) -> FrameSource:
    """A :class:`FrameSource` from ``"module:attr"``, a Scene class, a
    factory, a prepared scene or a timeline: its ``frame_at``.
    """
    from ..animation.scene import PreparedScene, Scene
    from ..animation.timing import Timeline

    target = spec
    if isinstance(spec, str):
        module, _, attr = spec.partition(":")
        if not attr:
            raise ValueError(f"expected 'module:attr', got {spec!r}")
        target = getattr(importlib.import_module(module), attr)
    if isinstance(target, type) and issubclass(target, Scene):
        target = target().prepare()
    elif callable(target) and not hasattr(target, "frame_at"):
        target = target()
    if isinstance(target, FrameSource):
        return target
    if not isinstance(target, (PreparedScene, Timeline)):
        raise TypeError(f"cannot render {target!r}: give a Scene, a prepared scene or a timeline")
    return FrameSource(target.frame_at, target.duration, target.audio, getattr(target, "subtitles", None))


def _format(source: FrameSource, sink: FrameSink, settings: RenderSettings, times: list[float]) -> FrameFormat:
    """The sink's format: its pixels, and the audio and subtitles of exactly the rendered frames."""
    config = settings.config
    width, height = config.pixel_width, config.pixel_height
    audio = None
    subtitles: tuple[Subtitle, ...] = ()
    if times:
        start, end = times[0], times[0] + len(times) / settings.fps
        if settings.audio and source.audio is not None:
            samples = source.audio(start, end, settings.sample_rate)
            audio = samples if np.any(samples) else None  # no sounds: no audio track
        if source.subtitles is not None:
            subtitles = tuple(
                Subtitle(max(item.start, start) - start, min(item.end, end) - start, item.text)
                for item in source.subtitles()
                if item.end > start and item.start < end
            )
    return FrameFormat(width, height, settings.fps, audio, settings.sample_rate, sink.pixel_format(width, height), subtitles)


def _frame_shape(format: FrameFormat) -> tuple[int, ...]:
    """The shape of a frame's array (see :data:`~.sinks.PixelFormat`)."""
    if format.pixel_format == "nv12":
        return (format.height * 3 // 2, format.width)
    return (format.height, format.width, 4)


class RenderCancelled(Exception):
    """``cancelled()`` said yes: the render stopped early."""


def render(
    source: Any,
    sink: FrameSink,
    settings: RenderSettings | None = None,
    workers: int = 1,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> RenderStats:
    """Render every frame of ``source`` into ``sink``; see the module docs.
    ``cancelled`` is asked between frames; when it says yes, the render
    stops (the sink is closed) and :class:`RenderCancelled` is raised."""
    settings = settings or RenderSettings()
    stop = cancelled or (lambda: False)
    started = time.perf_counter()
    if workers <= 1:
        count = _render_here(load_source(source), sink, settings, progress, stop)
    else:
        count = _render_in_workers(source, sink, settings, workers, progress, stop)
    return RenderStats(count, time.perf_counter() - started)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


class _Writer:
    """Feeds a sink from its own thread, in index order if it is ordered."""

    def __init__(self, sink: FrameSink, total: int, progress: Progress | None, capacity: int) -> None:
        self.sink = sink
        self.total = total
        self.progress = progress
        self._queue: queue.Queue = queue.Queue(capacity)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="frame writer", daemon=True)
        self._thread.start()

    def put(self, index: int, image: np.ndarray, release: Callable[[], None] | None = None) -> None:
        self.raise_error()
        self._queue.put((index, image, release))

    def finish(self) -> None:
        self.stop()
        self.raise_error()

    def stop(self) -> None:
        """Write what is queued and end the thread (idempotent)."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("writing frames failed") from self._error

    def _run(self) -> None:
        waiting: dict[int, tuple[np.ndarray, Callable[[], None] | None]] = {}
        next_index = written = 0
        while (item := self._queue.get()) is not None:
            if self._error is not None:
                _release(item[2])  # keep draining so producers never block
                continue
            index, image, release = item
            try:
                if not self.sink.ordered:
                    self._write(index, image, release)
                    written += 1
                else:
                    waiting[index] = (image, release)
                    while next_index in waiting:
                        self._write(next_index, *waiting.pop(next_index))
                        next_index += 1
                        written += 1
                if self.progress is not None:
                    self.progress(written, self.total)
            except BaseException as error:  # noqa: BLE001 - reported by finish()
                self._error = error
                for _, pending_release in waiting.values():
                    _release(pending_release)
                waiting.clear()

    def _write(self, index: int, image: np.ndarray, release: Callable[[], None] | None) -> None:
        try:
            self.sink.write(index, image)
        finally:
            del image
            _release(release)


def _release(release: Callable[[], None] | None) -> None:
    if release is not None:
        release()


# --------------------------------------------------------------------------
# One process
# --------------------------------------------------------------------------


def _pipeline(settings: RenderSettings, pixel_format: PixelFormat) -> Any:
    from .core_pipeline import CoreFramePipeline

    return CoreFramePipeline(settings.effects, settings.config, settings.in_flight, pixel_format=pixel_format)


def _render_here(
    source: FrameSource, sink: FrameSink, settings: RenderSettings, progress: Progress | None, cancelled: Callable[[], bool]
) -> int:
    times = settings.frame_times(source.duration)
    format = _format(source, sink, settings, times)
    pipeline = _pipeline(settings, format.pixel_format)
    sink.open(format)
    try:
        writer = _Writer(sink, len(times), progress, capacity=2 * settings.in_flight + 4)
        for index, t in enumerate(times):
            writer.raise_error()  # stop early if the sink failed
            if cancelled():
                pipeline.flush()
                writer.stop()
                raise RenderCancelled
            future = pipeline.submit(index, t, source.frame_at)
            future.add_done_callback(lambda done, index=index: writer.put(index, done.result()))
        pipeline.flush()
        writer.finish()
    finally:
        sink.close()
    return len(times)


# --------------------------------------------------------------------------
# Worker processes
# --------------------------------------------------------------------------


def _render_in_workers(
    spec: Any, sink: FrameSink, settings: RenderSettings, workers: int, progress: Progress | None, cancelled: Callable[[], bool]
) -> int:
    try:
        pickle.dumps(spec)
    except Exception as error:
        raise TypeError("worker processes need a picklable source, e.g. 'module:attr'") from error
    source = load_source(spec)
    times = settings.frame_times(source.duration)
    format = _format(source, sink, settings, times)
    shape = _frame_shape(format)
    frame_bytes = math.prod(shape)
    slots = settings.in_flight + 2

    context = multiprocessing.get_context("spawn")  # GPU devices do not survive fork
    tasks = context.Queue()
    for first in range(0, len(times), settings.chunk):
        tasks.put((first, min(first + settings.chunk, len(times))))
    for _ in range(workers):
        tasks.put(None)
    done = context.Queue()
    memories = [SharedMemory(create=True, size=frame_bytes * slots) for _ in range(workers)]
    free = [context.Queue() for _ in range(workers)]
    for worker_free in free:
        for slot in range(slots):
            worker_free.put(slot)
    processes = [
        context.Process(
            target=_worker_main,
            args=(worker, spec, settings, format.pixel_format, memories[worker].name, frame_bytes, tasks, free[worker], done),
            name=f"render worker {worker}",
            daemon=True,
        )
        for worker in range(workers)
    ]

    sink.open(format)
    writer: _Writer | None = None
    try:
        for process in processes:
            process.start()
        writer = _Writer(sink, len(times), progress, capacity=workers * slots + 1)
        running = workers
        while running:
            if cancelled():
                raise RenderCancelled
            try:
                message = done.get(timeout=0.2)
            except queue.Empty:
                continue
            if message[0] == "frame":
                _, worker, index, slot = message
                image = np.ndarray(shape, np.uint8, buffer=memories[worker].buf, offset=slot * frame_bytes)
                writer.put(index, image, lambda queue=free[worker], slot=slot: queue.put(slot))
                del image
                writer.raise_error()  # the sink failed: stop the workers
            elif message[0] == "done":
                running -= 1
            else:
                raise RuntimeError(f"render worker {message[1]} failed:\n{message[2]}")
        writer.finish()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        if writer is not None:
            writer.stop()  # no frame views into shared memory remain after this
        try:
            sink.close()
        finally:
            for memory in memories:
                memory.close()
                memory.unlink()
    return len(times)


def _worker_main(
    worker: int,
    spec: Any,
    settings: RenderSettings,
    pixel_format: PixelFormat,
    memory_name: str,
    frame_bytes: int,
    tasks: Any,
    free: Any,
    done: Any,
) -> None:  # pragma: no cover - runs in a child process
    memory = SharedMemory(name=memory_name)
    try:
        source = load_source(spec)
        times = settings.frame_times(source.duration)
        pipeline = _pipeline(settings, pixel_format)

        def deliver(index: int, image: np.ndarray) -> None:
            slot = free.get()  # waits while all of this worker's slots are queued
            target = np.ndarray(image.shape, np.uint8, buffer=memory.buf, offset=slot * frame_bytes)
            target[...] = image
            del target
            done.put(("frame", worker, index, slot))

        while (chunk := tasks.get()) is not None:
            for index in range(*chunk):
                future = pipeline.submit(index, times[index], source.frame_at)
                future.add_done_callback(lambda result, index=index: deliver(index, result.result()))
        pipeline.flush()
        done.put(("done", worker))
    except BaseException:
        done.put(("error", worker, traceback.format_exc()))
    finally:
        memory.close()
