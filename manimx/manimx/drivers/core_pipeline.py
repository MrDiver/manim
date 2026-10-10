"""Pipelined frame rendering on the core's renderer.

:class:`CoreFramePipeline` does what :class:`~.pipeline.FramePipeline`
with a :class:`~manimx.renderer.FrameComposer` does, for frames of the
core (views whose root is a ``PlannedFrame``): the core
(``manimx._core.FrameRenderer``, see CORE.md) packs, renders, composites
and reads back every frame in Rust, with up to ``in_flight`` frames on
the GPU. Python only asks the timeline for the views of each sub-frame
and builds each camera's Frame record.

A frame that shows exactly what the frame before showed is not rendered
(the core compares keys of every view); its future gets the previous
image, as in :class:`~.pipeline.FramePipeline`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from .. import _core
from ..renderer.config import Frame, RenderConfig, ViewState
from ..renderer.effects import Bloom, Effect, MotionBlur, ShaderEffect, ToneMap
from ..renderer.webgpu.core_packer import config_values
from ..renderer.webgpu.frame import camera_frame
from ..robjects.planned_frame import PlannedFrame
from .pipeline import _pass_on

SHADER_FOLDER = Path(__file__).resolve().parents[1] / "renderer" / "webgpu" / "shaders"


class CoreFramePipeline:
    """Renders frames of the core with up to ``in_flight`` of them on the GPU."""

    def __init__(self, effects: tuple[Effect, ...], config: RenderConfig, in_flight: int = 3, profile: bool = False) -> None:
        self.effects = tuple(effects)
        self.config = config
        self.in_flight = in_flight
        tone_map = next((effect for effect in effects if isinstance(effect, ToneMap)), ToneMap())
        self.renderer = _core.FrameRenderer(
            str(SHADER_FOLDER),
            config.pixel_width,
            config.pixel_height,
            config.msaa_samples,
            config.shadow_resolution,
            tuple(float(c) for c in config.background_rgba),
            tone_map.operator.value,
            [_effect_tuple(effect) for effect in effects if isinstance(effect, (Bloom, ShaderEffect))],
            in_flight,
            profile,
        )
        self._blur = next((effect for effect in effects if isinstance(effect, MotionBlur)), None)
        self._futures: dict[int, Future[np.ndarray]] = {}
        """Futures of the frames on the GPU, by submission number (the
        caller's indices need not be unique)."""
        self._submitted = 0
        self._last: Future[np.ndarray] | None = None

    def submit(self, index: int, time: float, scene: Frame | Callable[[float], Frame]) -> Future[np.ndarray]:
        """Queue the frame at ``time``; the future gets its ``(H, W, 4)``
        uint8 RGBA pixels.
        """
        times = self._sub_frames(scene, time)
        sub_frames = [(weight, self._views(scene(t) if callable(scene) else scene, t, len(times), index)) for index, (t, weight) in enumerate(times)]
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

    def flush(self) -> None:
        """Wait for every submitted frame."""
        self._deliver(self.renderer.finished(wait=True))

    def gpu_timings(self) -> dict[str, float]:
        """GPU milliseconds per pass since the last call (with ``profile``)."""
        return self.renderer.gpu_timings()

    def _deliver(self, finished: list[tuple[int, bytes]]) -> None:
        height, width = self.config.pixel_height, self.config.pixel_width
        for number, data in finished:
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
            self._futures.pop(number).set_result(image)

    def _sub_frames(self, scene: object, time: float) -> list[tuple[float, float]]:
        blur = self._blur
        if blur is None or not callable(scene) or blur.shutter == 0:
            return [(time, 1.0)]
        return blur.sample_times(time)

    def _views(self, frame: Frame, time: float, share: int = 1, index: int = 0) -> list[tuple]:
        """Each view as the core takes it: graph, state, key, config values,
        the camera's Frame record, the config values and Frame record of
        each of its camera views, and whether its picture depends on the
        time.
        """
        views = [frame] if isinstance(frame, ViewState) else list(frame)
        described = []
        for view in views:
            if not isinstance(view.root, PlannedFrame):
                raise TypeError("CoreFramePipeline renders frames of the core (PlannedFrame views) only")
            config = dataclasses.replace(
                self.config, time=time, shadow_share=share, shadow_index=index, camera=view.camera or self.config.camera, camera_views=view.camera_views
            )
            root = view.root
            pictures = [
                (config_values(picture), camera_frame(picture, 0).tobytes())
                for picture in (
                    dataclasses.replace(config, camera=camera_view.camera, view_aspect=camera_view.aspect_ratio, hide_camera_views=True)
                    for camera_view in view.camera_views
                )
            ]
            described.append((root.graph, root.state, root.key, config_values(config), camera_frame(config, 0).tobytes(), pictures, root.uses_time))
        return described


def _effect_tuple(effect: Bloom | ShaderEffect) -> tuple:
    if isinstance(effect, Bloom):
        return ("bloom", effect.threshold, effect.knee, effect.intensity, effect.radius, effect.levels)
    return ("shader", effect.code, effect.stage, list(effect.values))


__all__ = ["CoreFramePipeline"]
