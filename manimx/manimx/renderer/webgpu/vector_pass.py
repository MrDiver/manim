"""GPU side of the vector pipeline: buffers, pipelines and draw encoding.

Per frame, :meth:`VectorPass.prepare` uploads a :class:`VectorBatch` into
persistent storage buffers. :meth:`VectorPass.draw` then issues exactly two
``multi_draw_indirect`` calls, one for opaque and one for transparent draws.
"""

from __future__ import annotations

import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import multi_draw_indirect

from .gpu import COLOR_FORMAT, DEPTH_FORMAT, GPUContext, GrowableBuffer
from .vector_batch import DRAW_INDIRECT_DTYPE, VectorBatch

_STORAGE = wgpu.BufferUsage.STORAGE

# Standard "over" blending for straight (non-premultiplied) alpha.
_ALPHA_BLEND = {
    "color": {
        "src_factor": wgpu.BlendFactor.src_alpha,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
    "alpha": {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
}


class VectorPass:
    def __init__(self, context: GPUContext, frame_uniform: wgpu.GPUBuffer) -> None:
        self._context = context
        self._frame_uniform = frame_uniform
        self._buffers = {
            "objects": GrowableBuffer(context, "vector objects", _STORAGE),
            "instances": GrowableBuffer(context, "vector instances", _STORAGE),
            "fill_vertices": GrowableBuffer(context, "fill vertices", _STORAGE),
            "stroke_points": GrowableBuffer(context, "stroke points", _STORAGE),
            "stroke_curves": GrowableBuffer(context, "stroke curves", _STORAGE),
        }
        self._indirect = GrowableBuffer(context, "vector draws", wgpu.BufferUsage.INDIRECT)
        self._layout = self._create_bind_group_layout()
        self._shader = context.shader("vector.wgsl")
        self._pipelines: dict[tuple[int, bool], wgpu.GPURenderPipeline] = {}
        self._bind_group: wgpu.GPUBindGroup | None = None
        self._opaque_count = 0
        self._transparent_count = 0

    # -- per frame ----------------------------------------------------------

    def prepare(self, batch: VectorBatch) -> None:
        """Upload the frame's batch. Must be called before :meth:`draw`."""
        grown = False
        for name, buffer in self._buffers.items():
            grown |= buffer.upload(getattr(batch, name))
        # Opaque draws first, transparent ones right after (see draw()).
        self._indirect.upload(np.concatenate([batch.opaque_draws, batch.transparent_draws]))
        self._opaque_count = len(batch.opaque_draws)
        self._transparent_count = len(batch.transparent_draws)
        if grown or self._bind_group is None:
            self._bind_group = self._create_bind_group()

    def draw(self, render_pass: wgpu.GPURenderPassEncoder, sample_count: int) -> None:
        render_pass.set_bind_group(0, self._bind_group)
        stride = DRAW_INDIRECT_DTYPE.itemsize
        if self._opaque_count:
            render_pass.set_pipeline(self._pipeline(sample_count, transparent=False))
            multi_draw_indirect(render_pass, self._indirect.buffer, offset=0, count=self._opaque_count)
        if self._transparent_count:
            render_pass.set_pipeline(self._pipeline(sample_count, transparent=True))
            multi_draw_indirect(
                render_pass,
                self._indirect.buffer,
                offset=self._opaque_count * stride,
                count=self._transparent_count,
            )

    # -- setup --------------------------------------------------------------

    def _pipeline(self, sample_count: int, transparent: bool) -> wgpu.GPURenderPipeline:
        key = (sample_count, transparent)
        if key not in self._pipelines:
            self._pipelines[key] = self._create_pipeline(sample_count, transparent)
        return self._pipelines[key]

    def _create_pipeline(self, sample_count: int, transparent: bool) -> wgpu.GPURenderPipeline:
        device = self._context.device
        target = {"format": COLOR_FORMAT}
        if transparent:
            target["blend"] = _ALPHA_BLEND
        return device.create_render_pipeline(
            label=f"vector {'transparent' if transparent else 'opaque'} x{sample_count}",
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            vertex={"module": self._shader, "entry_point": "vs_main", "buffers": []},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                # Fills and strokes are flat and visible from both sides.
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": DEPTH_FORMAT,
                # Transparent draws also write depth: with the strict "less"
                # test an object can never blend over itself (e.g. where its
                # stroke ribbon overlaps a joint), only over earlier objects.
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": sample_count},
            fragment={"module": self._shader, "entry_point": "fs_main", "targets": [target]},
        )

    def _create_bind_group_layout(self) -> wgpu.GPUBindGroupLayout:
        visibility = wgpu.ShaderStage.VERTEX
        entries = [
            {
                "binding": 0,
                # The fragment shader reads the layer depth step from the frame.
                "visibility": visibility | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }
        ]
        for binding in range(1, 1 + len(self._buffers)):
            entries.append(
                {
                    "binding": binding,
                    "visibility": visibility,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                }
            )
        return self._context.device.create_bind_group_layout(entries=entries)

    def _create_bind_group(self) -> wgpu.GPUBindGroup:
        # Binding order matches vector.wgsl: frame, then the storage buffers.
        buffers = [self._frame_uniform, *(b.buffer for b in self._buffers.values())]
        return self._context.device.create_bind_group(
            layout=self._layout,
            entries=[
                {"binding": i, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}}
                for i, buffer in enumerate(buffers)
            ],
        )

