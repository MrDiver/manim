"""Optional GPU timing per render pass, using WebGPU timestamp queries.

    renderer = WGPURenderer(profile=True)
    renderer.render(scene, config)
    renderer.profiler.read()   # {"shadows": 0.41, "geometry": 0.22, ...} ms

Every profiled pass gets a timestamp pair; passes with the same name (for
example one per shadow view) are summed. Reading waits for the GPU, so the
profiler is meant for benchmarks and tuning, not for every frame.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import wgpu

_CAPACITY = 512  # timestamps, i.e. 256 passes per read


class GpuProfiler:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self.enabled = "timestamp-query" in device.features
        self._names: list[str] = []
        if not self.enabled:
            return
        self._queries = device.create_query_set(type=wgpu.QueryType.timestamp, count=_CAPACITY)
        size = _CAPACITY * 8
        self._resolve = device.create_buffer(size=size, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
        self._readback = device.create_buffer(size=size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

    def pass_timing(self, name: str) -> dict | None:
        """``timestamp_writes`` for a render pass descriptor, or None."""
        if not self.enabled or 2 * len(self._names) + 2 > _CAPACITY:
            return None
        index = 2 * len(self._names)
        self._names.append(name)
        return {
            "query_set": self._queries,
            "beginning_of_pass_write_index": index,
            "end_of_pass_write_index": index + 1,
        }

    def read(self) -> dict[str, float]:
        """Milliseconds per pass name since the last read (waits for the GPU)."""
        if not self.enabled or not self._names:
            self._names = []
            return {}
        count = 2 * len(self._names)
        encoder = self._device.create_command_encoder(label="profiler")
        encoder.resolve_query_set(self._queries, 0, count, self._resolve, 0)
        encoder.copy_buffer_to_buffer(self._resolve, 0, self._readback, 0, count * 8)
        self._device.queue.submit([encoder.finish()])
        self._readback.map_sync(wgpu.MapMode.READ, 0, count * 8)
        stamps = np.frombuffer(self._readback.read_mapped(0, count * 8), dtype=np.uint64).astype(np.int64)
        self._readback.unmap()

        timings: dict[str, float] = defaultdict(float)
        for i, name in enumerate(self._names):
            timings[name] += max(0, int(stamps[2 * i + 1] - stamps[2 * i])) / 1e6  # ns -> ms
        self._names = []
        return dict(timings)
