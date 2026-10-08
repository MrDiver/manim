"""Device creation and small GPU resource helpers."""

from __future__ import annotations

import re
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wgpu

SHADER_DIRECTORY = Path(__file__).parent / "shaders"

COLOR_FORMAT = wgpu.TextureFormat.rgba8unorm
"""Colors are stored and blended in sRGB values, like Manim's Cairo renderer."""
DEPTH_FORMAT = wgpu.TextureFormat.depth32float
SHADOW_FORMAT = wgpu.TextureFormat.depth32float
HDR_FORMAT = wgpu.TextureFormat.rgba16float
"""Linear radiance of lit meshes, before tone mapping."""
GBUFFER_FORMATS = (
    wgpu.TextureFormat.rgba8unorm_srgb,  # base color, occlusion
    wgpu.TextureFormat.rgba16float,  # world normal, roughness
    wgpu.TextureFormat.rgba16float,  # emissive radiance, metallic
    wgpu.TextureFormat.r16float,  # clip scope (selects the lights), -1 for none
)
"""Color targets of the G-buffer, see shaders/gbuffer.wgsl."""

REQUIRED_FEATURES = ("indirect-first-instance",)
"""``first_instance`` in indirect draws selects the per-draw instance records."""
OPTIONAL_FEATURES = ("timestamp-query",)
"""Used when available: GPU pass timings for the profiler."""


@dataclass(frozen=True, slots=True)
class GPUContext:
    device: wgpu.GPUDevice

    @staticmethod
    def create() -> GPUContext:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        missing = [f for f in REQUIRED_FEATURES if f not in adapter.features]
        if missing:
            raise RuntimeError(f"GPU adapter lacks required features: {missing}")
        optional = [f for f in OPTIONAL_FEATURES if f in adapter.features]
        device = adapter.request_device_sync(required_features=[*REQUIRED_FEATURES, *optional])
        return GPUContext(device)

    def shader(self, filename: str, replacements: dict[str, str] | None = None) -> wgpu.GPUShaderModule:
        """Compile a shader file (after #include expansion and the given
        plain-text ``replacements``, used for small variants)."""
        code = load_wgsl(filename)
        for old, new in (replacements or {}).items():
            code = code.replace(old, new)
        return self.device.create_shader_module(label=filename, code=code)


_INCLUDE = re.compile(r'^#include "([\w./-]+)"\s*$', re.MULTILINE)


def load_wgsl(filename: str, included: set[str] | None = None) -> str:
    """WGSL source of a shader file with its ``#include`` lines expanded.

    WGSL has no module system; ``#include "file.wgsl"`` lets pipelines share
    structs and functions (frame uniforms, material evaluation, lighting).
    Each file is included at most once per shader.
    """
    if included is None:
        included = set()
    included.add(filename)
    return expand_includes((SHADER_DIRECTORY / filename).read_text(encoding="utf-8"), included)


def expand_includes(code: str, included: set[str] | None = None) -> str:
    """Expand ``#include`` lines in WGSL source given as a string."""
    if included is None:
        included = set()

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


class ResourceCache:
    """Views and bind groups that live as long as what they refer to.

    Creating a view or a bind group is cheap on the GPU but costs a round
    of Python-to-native calls; frames that bind the same textures and
    buffers (every frame, every motion-blur sub-frame) reuse them instead.

    * Views are cached per texture in a weak-key map: they go away with
      their texture.
    * Bind groups are keyed by their layout and the identity of every
      resource (and buffer range). An entry holds its resources, so an id
      cannot be reused while the entry exists; the least recently used
      entries are dropped beyond ``capacity`` (e.g. after resizes).
    """

    def __init__(self, device: wgpu.GPUDevice, capacity: int = 512) -> None:
        self._device = device
        self._capacity = capacity
        self._views: weakref.WeakKeyDictionary[wgpu.GPUTexture, dict[tuple, wgpu.GPUTextureView]] = weakref.WeakKeyDictionary()
        self._bind_groups: OrderedDict[tuple, tuple[wgpu.GPUBindGroup, tuple]] = OrderedDict()

    def view(self, texture: wgpu.GPUTexture | wgpu.GPUTextureView, **descriptor: object) -> wgpu.GPUTextureView:
        """A view of ``texture`` (a view passes through unchanged)."""
        if isinstance(texture, wgpu.GPUTextureView):
            return texture
        views = self._views.setdefault(texture, {})
        key = tuple(sorted(descriptor.items()))
        view = views.get(key)
        if view is None:
            view = views[key] = texture.create_view(**descriptor)
        return view

    def bind_group(self, layout: wgpu.GPUBindGroupLayout, resources: list) -> wgpu.GPUBindGroup:
        """A bind group with ``resources`` as bindings 0, 1, 2, ..."""
        key = (id(layout), *(_resource_key(resource) for resource in resources))
        entry = self._bind_groups.get(key)
        if entry is not None:
            self._bind_groups.move_to_end(key)
            return entry[0]
        group = self._device.create_bind_group(
            layout=layout, entries=[{"binding": i, "resource": r} for i, r in enumerate(resources)]
        )
        self._bind_groups[key] = (group, (layout, *resources))
        if len(self._bind_groups) > self._capacity:
            self._bind_groups.popitem(last=False)
        return group


def _resource_key(resource: object) -> tuple:
    if isinstance(resource, dict):  # a buffer binding
        return (id(resource["buffer"]), resource.get("offset", 0), resource.get("size"))
    return (id(resource),)
