"""GPU side of the vector pipeline: buffers, pipelines and draw encoding.

Per frame, :meth:`VectorPass.prepare` uploads a :class:`VectorBatch` into
persistent storage buffers and one indirect-draw buffer laid out as

    [opaque draws][transparent draws][mask 0 draws][mask 1 draws]...

Every list is then issued with a single ``multi_draw_indirect`` call:
:meth:`draw_mask` renders one clip scope's mask into its own pass, and
:meth:`draw_frame` renders the opaque and the transparent lists into the
frame.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import multi_draw_indirect

from ..scene_graph import NO_CLIP
from .frame import FRAME_DTYPE
from .gpu import (
    COLOR_FORMAT,
    DEPTH_FORMAT,
    HDR_FORMAT,
    GPUContext,
    GrowableBuffer,
    ShadowStencil,
    shadow_depth_stencil,
)
from .vector_batch import DRAW_INDIRECT_DTYPE, VectorBatch

_STORAGE = wgpu.BufferUsage.STORAGE

# "Over" blending for premultiplied colors (see fs_main in vector.wgsl). In
# mask passes the alpha channel accumulates the union of all mask shapes.
_ALPHA_BLEND = {
    "color": {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
    "alpha": {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
}


class Role(Enum):
    """What a pipeline variant is used for."""

    OPAQUE = "opaque"
    """Frame pass, no blending; overlap is resolved by the depth test."""
    TRANSPARENT = "transparent"
    """Frame pass with blending, drawn back to front after the opaque list."""
    MASK = "mask"
    """Clip mask pass: blending, no depth buffer."""
    SHADOW = "shadow"
    """Shadow map pass: depth only, seen from a light (stencil variants by
    ShadowStencil)."""


class VectorPass:
    def __init__(self, context: GPUContext, frame_uniform: wgpu.GPUBuffer) -> None:
        self._context = context
        self._frame_uniform = frame_uniform
        self._buffers = {
            "objects": GrowableBuffer(context, "vector objects", _STORAGE),
            "instances": GrowableBuffer(context, "vector instances", _STORAGE),
            "fill_vertices": GrowableBuffer(context, "fill vertices", _STORAGE),
            "fill_colors": GrowableBuffer(context, "fill colors", _STORAGE),
            "stroke_points": GrowableBuffer(context, "stroke points", _STORAGE),
            "stroke_curves": GrowableBuffer(context, "stroke curves", _STORAGE),
        }
        self._indirect = GrowableBuffer(context, "vector draws", wgpu.BufferUsage.INDIRECT)
        # Shader and layout depend on whether the meshes' depth buffer is
        # multisampled; they are (re)built by _use_variant().
        self._multisampled: bool | None = None
        self._layout: wgpu.GPUBindGroupLayout | None = None
        self._shader: wgpu.GPUShaderModule | None = None
        self._pipelines: dict[tuple[Role, int], wgpu.GPURenderPipeline] = {}
        self._bind_group: wgpu.GPUBindGroup | None = None
        self._mask_layers: wgpu.GPUTextureView | None = None
        self._mesh_depth: wgpu.GPUTextureView | None = None
        self._camera_views: wgpu.GPUTextureView | None = None
        self._camera_view_sampler = context.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear
        )
        # (first draw, draw count) of each list in the indirect buffer.
        self._opaque = (0, 0)
        self._transparent = (0, 0)
        self._masks: list[tuple[int, int]] = []
        self._shadows: dict[tuple[int, int], tuple[int, int]] = {}
        """(first, count) of each (light scope, clip scope) run of casters."""
        self._shadow_groups: dict[tuple[int, int], wgpu.GPUBindGroup] = {}

    # -- per frame ----------------------------------------------------------

    def prepare(
        self,
        batch: VectorBatch,
        mask_layers: wgpu.GPUTextureView,
        mesh_depth: wgpu.GPUTextureView,
        multisampled: bool,
        full_upload: bool = False,
        camera_views: wgpu.GPUTextureView | None = None,
    ) -> None:
        """Upload the frame's batch. Must be called before drawing.

        ``mask_layers`` is the 2D array texture holding one layer per clip
        scope; content draws sample it, mask draws write to a copy of it.
        ``mesh_depth`` is the meshes' depth buffer that hides shapes behind
        meshes (see ``hidden_by_meshes`` in vector.wgsl). ``full_upload``
        ignores the batch's dirty ranges, for buffers that hold another
        packer's records. ``camera_views`` holds the pictures of camera
        views, one per layer.
        """
        changed = self._use_variant(multisampled)
        grown = False
        for name, buffer in self._buffers.items():
            ranges = None if full_upload or batch.dirty is None else batch.dirty[name]
            data = getattr(batch, name)
            grown |= buffer.upload(data) if ranges is None else buffer.write(data, ranges)

        shadow_scopes = list(batch.shadow_draws)
        lists = [batch.opaque_draws, batch.transparent_draws, *batch.shadow_draws.values(), *batch.mask_draws]
        self._indirect.upload(np.concatenate(lists))
        ranges, first = [], 0
        for draws in lists:
            ranges.append((first, len(draws)))
            first += len(draws)
        self._opaque, self._transparent = ranges[:2]
        self._shadows = shadow_runs(shadow_scopes, ranges[2 : 2 + len(shadow_scopes)], batch.shadow_clips)
        self._masks = ranges[2 + len(shadow_scopes) :]

        camera_views = camera_views or self._camera_views or self._empty_camera_views()
        inputs_changed = mask_layers is not self._mask_layers or mesh_depth is not self._mesh_depth or camera_views is not self._camera_views
        if grown or changed or self._bind_group is None or inputs_changed:
            self._mask_layers, self._mesh_depth, self._camera_views = mask_layers, mesh_depth, camera_views
            self._bind_group = self._create_bind_group(self._frame_uniform, 0)
            self._shadow_groups.clear()

    def draw_mask(self, render_pass: wgpu.GPURenderPassEncoder, sample_count: int, scope: int) -> None:
        """Draw the mask shapes of clip scope ``scope``."""
        render_pass.set_bind_group(0, self._bind_group)
        self._draw(render_pass, Role.MASK, sample_count, self._masks[scope])

    def draw_frame(self, render_pass: wgpu.GPURenderPassEncoder, sample_count: int) -> None:
        """Draw all opaque, then all transparent shapes into the frame."""
        render_pass.set_bind_group(0, self._bind_group)
        self._draw(render_pass, Role.OPAQUE, sample_count, self._opaque)
        self._draw(render_pass, Role.TRANSPARENT, sample_count, self._transparent)

    def shadow_clips(self, scope: int) -> list[int]:
        """The clip scopes of light scope ``scope``'s shadow casters."""
        return [clip for light_scope, clip in self._shadows if light_scope == scope]

    def draw_shadow(self, render_pass: wgpu.GPURenderPassEncoder, frames: wgpu.GPUBuffer, offset: int, scope: int, clip: int, clipped: bool) -> None:
        """Draw the depth of light scope ``scope``'s casters in clip scope
        ``clip``, seen from the frame at ``offset`` (``clipped``: where the
        stencil equals the reference)."""
        self._use_shadow_frame(render_pass, frames, offset)
        stencil = ShadowStencil.CLIPPED if clipped else ShadowStencil.NONE
        self._draw(render_pass, (Role.SHADOW, stencil), 1, self._shadows.get((scope, clip), (0, 0)))

    def draw_shadow_mask(self, render_pass: wgpu.GPURenderPassEncoder, frames: wgpu.GPUBuffer, offset: int, clip: int, inner: bool) -> None:
        """Draw clip scope ``clip``'s mask into the stencil, seen from the
        frame at ``offset``: the reference where it covers, or (``inner``)
        one more where it covers the reference."""
        self._use_shadow_frame(render_pass, frames, offset)
        stencil = ShadowStencil.INNER_MASK if inner else ShadowStencil.MASK
        self._draw(render_pass, (Role.SHADOW, stencil), 1, self._masks[clip])

    def _use_shadow_frame(self, render_pass: wgpu.GPURenderPassEncoder, frames: wgpu.GPUBuffer, offset: int) -> None:
        key = (id(frames), offset)
        if key not in self._shadow_groups:
            self._shadow_groups[key] = self._create_bind_group(frames, offset)
        render_pass.set_bind_group(0, self._shadow_groups[key])

    def _draw(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        role: Role | tuple[Role, ShadowStencil],
        sample_count: int,
        draws: tuple[int, int],
    ) -> None:
        first, count = draws
        if count == 0:
            return
        render_pass.set_pipeline(self._pipeline(role, sample_count))
        multi_draw_indirect(
            render_pass,
            self._indirect.buffer,
            offset=first * DRAW_INDIRECT_DTYPE.itemsize,
            count=count,
        )

    # -- setup --------------------------------------------------------------

    def _pipeline(self, role: Role | tuple[Role, ShadowStencil], sample_count: int) -> wgpu.GPURenderPipeline:
        key = (role, sample_count)
        if key not in self._pipelines:
            if isinstance(role, tuple):
                self._pipelines[key] = self._create_shadow_pipeline(role[1])
            else:
                self._pipelines[key] = self._create_pipeline(role, sample_count)
        return self._pipelines[key]

    def _create_pipeline(self, role: Role, sample_count: int) -> wgpu.GPURenderPipeline:
        device = self._context.device
        target = {"format": COLOR_FORMAT}
        if role is not Role.OPAQUE:
            target["blend"] = _ALPHA_BLEND
        depth_stencil = None
        if role is not Role.MASK:
            depth_stencil = {
                "format": DEPTH_FORMAT,
                # Transparent draws also write depth: with the strict "less"
                # test an object can never blend over itself (e.g. where its
                # stroke ribbon overlaps a joint), only over earlier objects.
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            }
        entry_point = "fs_mask" if role is Role.MASK else "fs_main"
        return device.create_render_pipeline(
            label=f"vector {role.value} x{sample_count}",
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            vertex={"module": self._shader, "entry_point": "vs_main", "buffers": []},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                # Fills and strokes are flat and visible from both sides.
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil=depth_stencil,
            multisample={"count": sample_count},
            fragment={"module": self._shader, "entry_point": entry_point, "targets": [target]},
        )

    def _create_shadow_pipeline(self, stencil: ShadowStencil) -> wgpu.GPURenderPipeline:
        device = self._context.device
        return device.create_render_pipeline(
            label=f"vector shadow {stencil.value}",
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            vertex={"module": self._shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            depth_stencil=shadow_depth_stencil(stencil),
            fragment={"module": self._shader, "entry_point": "fs_shadow", "targets": []},
        )

    def _empty_camera_views(self) -> wgpu.GPUTextureView:
        """A one-pixel picture, for frames drawn without a renderer's
        camera view texture."""
        texture = self._context.device.create_texture(
            size=(1, 1, 1), format=HDR_FORMAT, usage=wgpu.TextureUsage.TEXTURE_BINDING
        )
        return texture.create_view(dimension=wgpu.TextureViewDimension.d2_array)

    def _use_variant(self, multisampled: bool) -> bool:
        """Build shader, layout and pipelines for the mesh depth's sample
        count; returns True if they changed."""
        if multisampled == self._multisampled:
            return False
        replacements = {} if multisampled else {"texture_depth_multisampled_2d": "texture_depth_2d"}
        self._multisampled = multisampled
        self._shader = self._context.shader("vector.wgsl", replacements)
        self._layout = self._create_bind_group_layout(multisampled)
        self._pipelines.clear()
        return True

    def _create_bind_group_layout(self, multisampled: bool) -> wgpu.GPUBindGroupLayout:
        vertex, fragment = wgpu.ShaderStage.VERTEX, wgpu.ShaderStage.FRAGMENT
        entries = [
            {
                "binding": 0,
                # The fragment shader reads the layer depth step from the frame.
                "visibility": vertex | fragment,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }
        ]
        for binding in range(1, 1 + len(self._buffers)):
            entries.append(
                {
                    "binding": binding,
                    "visibility": vertex,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                }
            )
        entries.append(
            {
                "binding": 1 + len(self._buffers),
                "visibility": fragment,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2_array,
                },
            }
        )
        entries.append(
            {
                "binding": 2 + len(self._buffers),
                "visibility": fragment,
                "texture": {"sample_type": wgpu.TextureSampleType.depth, "multisampled": multisampled},
            }
        )
        entries.append(
            {
                "binding": 3 + len(self._buffers),
                "visibility": fragment,
                "texture": {"sample_type": wgpu.TextureSampleType.float, "view_dimension": wgpu.TextureViewDimension.d2_array},
            }
        )
        entries.append({"binding": 4 + len(self._buffers), "visibility": fragment, "sampler": {"type": wgpu.SamplerBindingType.filtering}})
        return self._context.device.create_bind_group_layout(entries=entries)

    def _create_bind_group(self, frames: wgpu.GPUBuffer, offset: int) -> wgpu.GPUBindGroup:
        # Binding order matches vector.wgsl: frame, storage buffers, masks,
        # mesh depth, camera views and their sampler.
        entries = [{"binding": 0, "resource": {"buffer": frames, "offset": offset, "size": FRAME_DTYPE.itemsize}}]
        buffers = [b.buffer for b in self._buffers.values()]
        entries += [
            {"binding": 1 + i, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}}
            for i, buffer in enumerate(buffers)
        ]
        entries.append({"binding": 1 + len(buffers), "resource": self._mask_layers})
        entries.append({"binding": 2 + len(buffers), "resource": self._mesh_depth})
        entries.append({"binding": 3 + len(buffers), "resource": self._camera_views})
        entries.append({"binding": 4 + len(buffers), "resource": self._camera_view_sampler})
        return self._context.device.create_bind_group(layout=self._layout, entries=entries)


def shadow_runs(scopes: list[int], ranges: list[tuple[int, int]], clips: dict[int, list[tuple[int, int]]]) -> dict[tuple[int, int], tuple[int, int]]:
    """(first, count) per (light scope, clip scope) of shadow casters: each
    light scope's range split into its runs of equal clip scopes."""
    runs: dict[tuple[int, int], tuple[int, int]] = {}
    for scope, (first, count) in zip(scopes, ranges, strict=True):
        for clip, run in clips.get(scope, [(NO_CLIP, count)] if count else []):
            runs[(scope, clip)] = (first, run)
            first += run
    return runs
