"""Pipelined frame rendering: keep the GPU busy while the CPU prepares the
next frames.

A synchronous render waits for each frame to finish and be read back
before the CPU starts the next one, so the CPU and the GPU take turns.
:class:`FramePipeline` overlaps them instead::

    CPU   [state_at + encode 0][state_at + encode 1][state_at + encode 2][wait 0] ...
    GPU                        [frame 0            ][frame 1            ]...

Each submitted frame is copied into its own staging buffer (one per frame
in flight) and mapped asynchronously; the CPU only waits when all staging
buffers are busy, and then for the oldest frame, which is usually done
by then. The render targets themselves are shared: WebGPU executes
submissions in order, so the next frame's passes never overwrite a target
before the copy of the previous frame has read it.

Results arrive as :class:`concurrent.futures.Future` objects, completed
in submission order; attach callbacks to put frames wherever they go.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import wgpu

from ..objects import RNode
from ..renderer.config import RenderConfig, ViewState
from ..renderer.webgpu import FrameComposer

_ROW_ALIGNMENT = 256  # bytes_per_row of texture-to-buffer copies


@dataclass(eq=False)
class _InFlight:
    index: int
    staging: wgpu.GPUBuffer
    mapped: object  # GPUPromise of the mapping
    future: Future[np.ndarray]


class FramePipeline:
    """Renders frames with up to ``in_flight`` of them on the GPU at once."""

    def __init__(self, composer: FrameComposer, config: RenderConfig, in_flight: int = 3) -> None:
        if in_flight < 1:
            raise ValueError("a pipeline needs at least one frame in flight")
        self.composer = composer
        self.config = config
        self.in_flight = in_flight
        self._device = composer.renderer.context.device
        width = config.pixel_width
        self._row_bytes = (4 * width + _ROW_ALIGNMENT - 1) // _ROW_ALIGNMENT * _ROW_ALIGNMENT
        size = self._row_bytes * config.pixel_height
        self._free = [
            self._device.create_buffer(
                label=f"readback {i}", size=size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
            )
            for i in range(in_flight)
        ]
        self._pending: deque[_InFlight] = deque()

    def submit(
        self, index: int, time: float, scene: RNode | ViewState | Callable[[float], RNode | ViewState]
    ) -> Future[np.ndarray]:
        """Queue the frame at ``time``; the future gets its ``(H, W, 4)``
        uint8 RGBA pixels. Blocks only while all staging buffers are busy."""
        if not self._free:
            self._finish_oldest()
        staging = self._free.pop()
        config = dataclasses.replace(self.config, time=time)
        texture = self.composer.compose_texture(scene, config)
        encoder = self._device.create_command_encoder(label="readback")
        encoder.copy_texture_to_buffer(
            {"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
            {"buffer": staging, "offset": 0, "bytes_per_row": self._row_bytes, "rows_per_image": config.pixel_height},
            (config.pixel_width, config.pixel_height, 1),
        )
        self._device.queue.submit([encoder.finish()])
        future: Future[np.ndarray] = Future()
        future.set_running_or_notify_cancel()
        self._pending.append(_InFlight(index, staging, staging.map_async(wgpu.MapMode.READ), future))
        return future

    def flush(self) -> None:
        """Wait for every submitted frame."""
        while self._pending:
            self._finish_oldest()

    def _finish_oldest(self) -> None:
        frame = self._pending.popleft()
        try:
            frame.mapped.sync_wait()  # type: ignore[attr-defined]
            data = np.frombuffer(frame.staging.read_mapped(copy=False), dtype=np.uint8)
            height, width = self.config.pixel_height, self.config.pixel_width
            # Drop the row padding; the copy also frees the mapped memory.
            image = data.reshape(height, self._row_bytes)[:, : 4 * width].reshape(height, width, 4).copy()
            frame.staging.unmap()
        except Exception as error:  # pragma: no cover - device loss and the like
            frame.future.set_exception(error)
        else:
            frame.future.set_result(image)
        finally:
            self._free.append(frame.staging)
