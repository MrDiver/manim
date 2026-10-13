"""Pipelined frame rendering on the core's renderer.

:class:`CoreFramePipeline` renders frames: views, each a render object
tree (manimx.robjects) and the camera looking at it, as a timeline's or
prepared scene's ``frame_at`` returns them. The core
(``manimx._core.FrameRenderer``) gets each tree with its
``RenderConfig`` and packs, renders, composites and reads back every
frame in Rust, with up to ``in_flight`` frames on the GPU, so the CPU
prepares the next frames while the GPU draws the earlier ones. Python
only asks for the views of each sub-frame and sets their configs.

Frames come back as RGBA, or with ``pixel_format="nv12"`` as the NV12
planes a video encoder takes, converted on the GPU (a third of the bytes
to read back; see :class:`~.sinks.FrameFormat`).

A frame that shows exactly what the frame before showed is not rendered
(the core compares the trees and configs of every view); its future gets
the previous image.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np

from .. import _core
from ..renderer.config import Frame, RenderConfig, ViewState
from ..renderer.effects import Bloom, Effect, RenderSamples, ShaderEffect, ToneMap
from .sinks import PixelFormat

class CoreFramePipeline:
    """Renders frames of the core with up to ``in_flight`` of them on the GPU."""

    def __init__(
        self, effects: tuple[Effect, ...], config: RenderConfig, in_flight: int = 3, profile: bool = False, pixel_format: PixelFormat = "rgba"
    ) -> None:
        self.effects = tuple(effects)
        self.config = config
        self.in_flight = in_flight
        tone_map = next((effect for effect in effects if isinstance(effect, ToneMap)), ToneMap())
        self.renderer = _core.FrameRenderer(
            config,
            tone_map.operator.value,
            [_effect_tuple(effect) for effect in effects if isinstance(effect, (Bloom, ShaderEffect))],
            in_flight,
            profile,
            pixel_format,
        )
        self._samples = next((effect for effect in effects if isinstance(effect, RenderSamples)), RenderSamples(1))
        self._futures: dict[int, Future[np.ndarray]] = {}
        """Futures of the frames on the GPU, by submission number (the
        caller's indices need not be unique)."""
        self._submitted = 0
        self._last: Future[np.ndarray] | None = None

    def submit(self, index: int, time: float, scene: Frame | Callable[[float], Frame]) -> Future[np.ndarray]:
        """Queue the frame at ``time``; the future gets its pixels: an
        ``(H, W, 4)`` uint8 RGBA array, or an NV12 frame.
        """
        sub_frames = self._sub_frames(scene, time)
        future: Future[np.ndarray] = Future()
        future.set_running_or_notify_cancel()
        number = self._submitted
        self._submitted += 1
        held = self.renderer.render(number, sub_frames)
        if held and self._last is not None:
            # The same picture as the frame before: no render, no readback.
            self._last.add_done_callback(lambda done: _pass_on(done, future))
        else:
            self._futures[number] = future
        self._last = future
        self._deliver(self.renderer.finished())
        return future

    def render(self, scene: Frame | Callable[[float], Frame], time: float) -> np.ndarray:
        """The frame at ``time``, waiting for it: ``(H, W, 4)`` uint8 RGBA."""
        future = self.submit(0, time, scene)
        self.flush()
        return future.result()

    def flush(self) -> None:
        """Wait for every submitted frame."""
        self._deliver(self.renderer.finished(wait=True))

    def gpu_timings(self) -> dict[str, float]:
        """GPU milliseconds per pass since the last call (with ``profile``)."""
        return self.renderer.gpu_timings()

    def _deliver(self, finished: list[tuple[int, np.ndarray]]) -> None:
        for number, image in finished:
            self._futures.pop(number).set_result(image)

    def _sub_frames(self, scene: Frame | Callable[[float], Frame], time: float) -> list[tuple[float, list[tuple]]]:
        """``(weight, views)`` of each render of the frame at ``time``."""
        samples = self._samples
        moving = callable(scene) and samples.shutter > 0
        times = samples.sample_times(time) if moving else [(time, 1 / samples.count)] * samples.count
        still = None if moving else (scene(time) if callable(scene) else scene)
        return [
            (weight, self._views(scene(t) if still is None else still, t, samples.count, index, samples.jitter(index)))
            for index, (t, weight) in enumerate(times)
        ]

    def _views(self, frame: Frame, time: float, share: int = 1, index: int = 0, jitter: tuple[float, float] = (0.0, 0.0)) -> list[tuple]:
        """Each view as every renderer takes it: the render object tree and
        the ``RenderConfig`` to see it with (the view's camera and camera
        views, the time, the render's share of the shadow taps and its
        camera offset).
        """
        views = [frame] if isinstance(frame, ViewState) else list(frame)
        config = dataclasses.replace(self.config, time=time, shadow_share=share, shadow_index=index, jitter=jitter)
        return [(view.root, dataclasses.replace(config, camera=view.camera or self.config.camera, camera_views=view.camera_views)) for view in views]


def _effect_tuple(effect: Bloom | ShaderEffect) -> tuple:
    if isinstance(effect, Bloom):
        return ("bloom", effect.threshold, effect.knee, effect.intensity, effect.radius, effect.levels)
    return ("shader", effect.code, effect.stage, list(effect.values))


def _pass_on(done: Future[np.ndarray], future: Future[np.ndarray]) -> None:
    """Complete ``future`` like ``done`` (a held frame repeats its image)."""
    error = done.exception()
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(done.result())


__all__ = ["CoreFramePipeline"]
