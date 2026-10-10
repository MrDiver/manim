"""Shadow map rendering: one depth layer per shadow view.

For every view from shadow_views.py, the casters' depth is rendered into
one layer of a depth-stencil 2D array texture, using the normal mesh and
vector pipelines with a frame uniform whose "camera" is the light. The
lighting pass then samples its depth (see shaders/shadows.wgsl).

Casters inside clipping groups cast only the shadow of what their masks
let through, as seen from the light: before a clip scope's casters, the
masks between the light and them are drawn into the stencil, outermost
first. (A light limited to a clip scope is inside it, as are the casters
it lights: that scope's mask, and those around it, do not apply.) The outermost
writes a reference value, each nested one counts it up where the masks
around it are; the casters then draw where the stencil holds the last
value. References only grow within a layer, so values left by earlier
clip scopes never match.

Layers whose views say they are unchanged (``ShadowViews.changed``) keep
their depth from the previous render and get no pass, unless the maps
were just recreated or the caller asks for all of them.
"""

from __future__ import annotations

import numpy as np
import wgpu

from ..scene_graph import NO_CLIP
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
_STENCIL_MAX = 255
"""The largest stencil value (8 bits)."""
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
        self._scope_parents: list[int] = []
        self._draw_layers: list[int] = []
        self._ensure_maps(1, 1)

    def prepare(self, shadows: ShadowViews, resolution: int, scope_parents: np.ndarray, all_layers: bool = False) -> bool:
        """Upload the views; returns True if the map texture was recreated.

        ``scope_parents`` holds each clip scope's parent (NO_CLIP for none).
        ``all_layers`` renders every layer, also those the views mark
        unchanged (the maps hold another packer's depth).
        """
        self._scope_parents = [int(parent) for parent in scope_parents]
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
            render_pass = self._begin(encoder, layer, profiler)
            offset = layer * _FRAME_STRIDE
            scope = self._scopes[layer]
            frames = self._frames.buffer
            reference = 0
            around_light = len(self._mask_chain(scope))
            for clip in sorted({*meshes.shadow_clips(scope), *vectors.shadow_clips(scope)}):
                # The light is inside its own clip scope and the ones around
                # it: only the masks between it and the casters cut them.
                chain = self._mask_chain(clip)[around_light:]
                if chain:
                    if reference + len(chain) > _STENCIL_MAX:
                        # Out of stencil values: start the layer's stencil over.
                        render_pass.end()
                        render_pass = self._begin(encoder, layer, profiler, clear_depth=False)
                        reference = 0
                    reference += 1
                    render_pass.set_stencil_reference(reference)
                    vectors.draw_shadow_mask(render_pass, frames, offset, chain[0], inner=False)
                    for inner in chain[1:]:
                        render_pass.set_stencil_reference(reference)
                        vectors.draw_shadow_mask(render_pass, frames, offset, inner, inner=True)
                        reference += 1
                    render_pass.set_stencil_reference(reference)
                meshes.draw_shadow(render_pass, frames, offset, scope, clip, clipped=bool(chain))
                vectors.draw_shadow(render_pass, frames, offset, scope, clip, clipped=bool(chain))
            render_pass.end()

    def _begin(self, encoder: wgpu.GPUCommandEncoder, layer: int, profiler: GpuProfiler, clear_depth: bool = True) -> wgpu.GPURenderPassEncoder:
        return encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": self._layers[layer],
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear if clear_depth else wgpu.LoadOp.load,
                "depth_store_op": wgpu.StoreOp.store,
                "stencil_clear_value": 0,
                "stencil_load_op": wgpu.LoadOp.clear,
                "stencil_store_op": wgpu.StoreOp.discard,
            },
            timestamp_writes=profiler.pass_timing("shadows"),
        )

    def _mask_chain(self, clip: int) -> list[int]:
        """The clip scopes ``clip`` is inside, outermost first (with ``clip``)."""
        chain = []
        while clip != NO_CLIP:
            chain.append(clip)
            clip = self._scope_parents[clip]
        return chain[::-1]

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
        # The lighting pass samples the depth; the stencil only serves the passes.
        self.maps_view = self._maps.create_view(
            format=wgpu.TextureFormat.depth32float, dimension=wgpu.TextureViewDimension.d2_array, aspect=wgpu.TextureAspect.depth_only
        )
        self._layers = [
            self._maps.create_view(dimension=wgpu.TextureViewDimension.d2, base_array_layer=i, array_layer_count=1)
            for i in range(layers)
        ]
        return True
