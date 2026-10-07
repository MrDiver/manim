"""Device creation and small GPU resource helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wgpu

SHADER_DIRECTORY = Path(__file__).parent / "shaders"

COLOR_FORMAT = wgpu.TextureFormat.rgba8unorm
"""Colors are stored and blended in sRGB values, like Manim's Cairo renderer."""
DEPTH_FORMAT = wgpu.TextureFormat.depth32float

REQUIRED_FEATURES = ("indirect-first-instance",)
"""``first_instance`` in indirect draws selects the per-draw instance records."""


@dataclass(frozen=True, slots=True)
class GPUContext:
    device: wgpu.GPUDevice

    @staticmethod
    def create() -> GPUContext:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        missing = [f for f in REQUIRED_FEATURES if f not in adapter.features]
        if missing:
            raise RuntimeError(f"GPU adapter lacks required features: {missing}")
        device = adapter.request_device_sync(required_features=list(REQUIRED_FEATURES))
        return GPUContext(device)

    def shader(self, filename: str) -> wgpu.GPUShaderModule:
        code = (SHADER_DIRECTORY / filename).read_text(encoding="utf-8")
        return self.device.create_shader_module(label=filename, code=code)


class GrowableBuffer:
    """A GPU buffer that is reused across frames and grows when needed."""

    MIN_SIZE = 256

    def __init__(self, context: GPUContext, label: str, usage: int) -> None:
        self._device = context.device
        self._label = label
        self._usage = usage | wgpu.BufferUsage.COPY_DST
        self.buffer = self._allocate(self.MIN_SIZE)

    def upload(self, data: np.ndarray) -> bool:
        """Write ``data`` to the start of the buffer.

        Returns ``True`` if the buffer had to be reallocated, in which case
        bind groups referencing it must be recreated.
        """
        size = max(self.MIN_SIZE, data.nbytes)
        grown = size > self.buffer.size
        if grown:
            self.buffer.destroy()
            # Grow geometrically so a slowly growing scene reallocates rarely.
            self.buffer = self._allocate(max(size, 2 * self.buffer.size))
        if data.nbytes:
            self._device.queue.write_buffer(self.buffer, 0, np.ascontiguousarray(data))
        return grown

    def _allocate(self, size: int) -> wgpu.GPUBuffer:
        size = (size + 3) // 4 * 4  # buffer sizes must be multiples of 4
        return self._device.create_buffer(label=self._label, size=size, usage=self._usage)
