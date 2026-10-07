"""Device creation and small GPU resource helpers."""

from __future__ import annotations

import re
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
        return self.device.create_shader_module(label=filename, code=load_wgsl(filename))


_INCLUDE = re.compile(r'^#include "([\w./-]+)"\s*$', re.MULTILINE)


def load_wgsl(filename: str, included: set[str] | None = None) -> str:
    """WGSL source with ``#include "file.wgsl"`` lines replaced by that file.

    WGSL has no module system; this lets pipelines share structs and
    functions (frame uniforms, material evaluation, lighting) so each piece
    can later move to another pass, e.g. lighting into a deferred pass.
    Each file is included at most once per shader.
    """
    if included is None:
        included = set()
    included.add(filename)
    code = (SHADER_DIRECTORY / filename).read_text(encoding="utf-8")

    def include(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in included:
            return f"// {name} already included"
        return load_wgsl(name, included)

    return _INCLUDE.sub(include, code)


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
