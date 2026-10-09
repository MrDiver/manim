"""Shadow map rendering: one depth layer per shadow view.

For every view from shadow_views.py, the casters' depth is rendered into
one layer of a ``depth32float`` 2D array texture, using the normal mesh and
vector pipelines with a frame uniform whose "camera" is the light. The
lighting pass then samples the array (see shaders/shadows.wgsl).

Layers whose views say they are unchanged (``ShadowViews.changed``) keep
their depth from the previous render and get no pass, unless the maps
were just recreated or the caller asks for all of them.
"""

from __future__ import annotations

import numpy as np
import wgpu

from .frame import FRAME_DTYPE
from .gpu import SHADOW_FORMAT, GPUContext, GrowableBuffer
from .mesh_pass import MeshPass
from .profiler import GpuProfiler
from .shadow_views import SHADOW_VIEW_DTYPE, ShadowViews
from .vector_pass import VectorPass

# Frames for the shadow views live in one uniform buffer; bind group
# offsets must be multiples of 256 bytes.
_FRAME_STRIDE = 512
assert FRAME_DTYPE.itemsize <= _FRAME_STRIDE
_STRIDED_FRAME = np.dtype({"names": ["frame"], "formats": [FRAME_DTYPE], "itemsize": _FRAME_STRIDE})


class ShadowPass:
    def __init__(self, context: GPUContext) -> None:
        self._context = context
        self._frames = GrowableBuffer(context, "shadow frames", wgpu.BufferUsage.UNIFORM)
        self.views = GrowableBuffer(context, "shadow views", wgpu.BufferUsage.STORAGE)
        self.sampler = context.device.create_sampler(
            compare=wgpu.CompareFunction.less_equal,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
        self._maps: wgpu.GPUTexture | None = None
        self.maps_view: wgpu.GPUTextureView | None = None
        self._layers: list[wgpu.GPUTextureView] = []
        self._count = 0
        self._draw_layers: list[int] = []
        self._ensure_maps(1, 1)

    def prepare(self, shadows: ShadowViews, resolution: int, all_layers: bool = False) -> bool:
        """Upload the views; returns True if the map texture was recreated.

        ``all_layers`` renders every layer, also those the views mark
        unchanged (the maps hold another packer's depth).
        """
        self._count = len(shadows)
        self._scopes = list(shadows.scopes)
        frames = np.zeros(max(1, self._count), _STRIDED_FRAME)
        frames["frame"][: self._count] = shadows.frames
        self._frames.upload(frames)
        self.views.upload(shadows.views if self._count else np.zeros(1, SHADOW_VIEW_DTYPE))
        if self._count == 0:
            self._draw_layers = []
            return False
        recreated = self._ensure_maps(resolution, self._count)
        changed = shadows.changed
        if recreated or all_layers or changed is None:
            self._draw_layers = list(range(self._count))
        else:
            self._draw_layers = [layer for layer in range(self._count) if changed[layer]]
        return recreated

    def draw(self, encoder: wgpu.GPUCommandEncoder, meshes: MeshPass, vectors: VectorPass, profiler: GpuProfiler) -> None:
        for layer in self._draw_layers:
            render_pass = encoder.begin_render_pass(
                color_attachments=[],
                depth_stencil_attachment={
                    "view": self._layers[layer],
                    "depth_clear_value": 1.0,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                },
                timestamp_writes=profiler.pass_timing("shadows"),
            )
            offset = layer * _FRAME_STRIDE
            scope = self._scopes[layer]
            meshes.draw_shadow(render_pass, self._frames.buffer, offset, scope)
            vectors.draw_shadow(render_pass, self._frames.buffer, offset, scope)
            render_pass.end()

    def _ensure_maps(self, resolution: int, layers: int) -> bool:
        maps = self._maps
        if maps is not None and maps.width == resolution and maps.depth_or_array_layers >= layers:
            return False
        if maps is not None:
            layers = max(layers, maps.depth_or_array_layers)
        self._maps = self._context.device.create_texture(
            label="shadow maps",
            size=(resolution, resolution, layers),
            format=SHADOW_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.maps_view = self._maps.create_view(dimension=wgpu.TextureViewDimension.d2_array)
        self._layers = [
            self._maps.create_view(dimension=wgpu.TextureViewDimension.d2, base_array_layer=i, array_layer_count=1)
            for i in range(layers)
        ]
        return True
