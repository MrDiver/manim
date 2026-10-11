"""Deferred lighting pass: G-buffer + lights -> linear HDR "lit" layer.

Fullscreen triangles shade the G-buffer: once per pixel where all MSAA
samples hold one surface, once per sample on edges, which a first
triangle marks in a stencil (see shaders/deferred_lighting.wgsl). Lights live in one storage buffer, so the
cost grows with lights x covered samples, independent of how many meshes
or how much overdraw produced the G-buffer.
"""

from __future__ import annotations

import numpy as np
import wgpu

from .gpu import EDGE_FORMAT, HDR_FORMAT, GPUContext, GrowableBuffer
from .shadow_pass import ShadowPass


_EDGE = 1
"""The stencil value of a pixel on an edge."""
_FIRST_SAMPLE = "const FIRST_SAMPLE: vec2<f32> = vec2<f32>(0.375, 0.125);"


class LightingPass:
    def __init__(self, context: GPUContext, frame_uniform: wgpu.GPUBuffer) -> None:
        self._context = context
        self._frame_uniform = frame_uniform
        self._lights = GrowableBuffer(context, "lights", wgpu.BufferUsage.STORAGE)
        self._scope_parents = GrowableBuffer(context, "scope parents", wgpu.BufferUsage.STORAGE)
        self._variants: dict[int, tuple[wgpu.GPUBindGroupLayout, list[wgpu.GPURenderPipeline]]] = {}
        self._bind_group: wgpu.GPUBindGroup | None = None
        self._inputs: tuple[wgpu.GPUTextureView, ...] = ()
        self._sample_count = 0

    def prepare(
        self,
        lights: np.ndarray,
        scope_parents: np.ndarray,
        gbuffer: tuple[wgpu.GPUTextureView, ...],
        sample_count: int,
        shadows: ShadowPass,
        shadows_changed: bool,
    ) -> None:
        """Upload lights and scope parents; ``gbuffer`` is (albedo, normal,
        emissive, scope, depth)."""
        grown = self._lights.upload(lights)
        grown |= self._scope_parents.upload(scope_parents if len(scope_parents) else np.full(1, -1, np.int32))
        inputs = (*gbuffer, shadows.maps_view, shadows.views.buffer)
        if grown or shadows_changed or self._bind_group is None or inputs != self._inputs or sample_count != self._sample_count:
            self._inputs, self._sample_count = inputs, sample_count
            self._bind_group = self._context.device.create_bind_group(
                layout=self._variant(sample_count)[0],
                entries=[
                    {"binding": 0, "resource": {"buffer": self._frame_uniform, "offset": 0, "size": self._frame_uniform.size}},
                    {"binding": 1, "resource": {"buffer": self._lights.buffer, "offset": 0, "size": self._lights.buffer.size}},
                    {"binding": 2, "resource": gbuffer[0]},
                    {"binding": 3, "resource": gbuffer[1]},
                    {"binding": 4, "resource": gbuffer[2]},
                    {"binding": 9, "resource": gbuffer[3]},
                    {"binding": 5, "resource": gbuffer[4]},
                    {"binding": 10, "resource": {"buffer": self._scope_parents.buffer, "offset": 0, "size": self._scope_parents.buffer.size}},
                    {"binding": 6, "resource": shadows.maps_view},
                    {"binding": 7, "resource": shadows.sampler},
                    {"binding": 8, "resource": {"buffer": shadows.views.buffer, "offset": 0, "size": shadows.views.buffer.size}},
                ],
            )

    def draw(self, render_pass: wgpu.GPURenderPassEncoder) -> None:
        """Light into ``render_pass``, whose stencil (with MSAA) is the
        edge stencil (``RenderTargets.lit_attachments``), cleared."""
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.set_stencil_reference(_EDGE)
        for pipeline in self._variant(self._sample_count)[1]:
            render_pass.set_pipeline(pipeline)
            render_pass.draw(3)  # one fullscreen triangle

    def _variant(self, sample_count: int) -> tuple[wgpu.GPUBindGroupLayout, list[wgpu.GPURenderPipeline]]:
        """Layout and pipelines for multisampled G-buffers (marking edges,
        per pixel, per sample on edges) or single-sampled ones.

        A multisampled texture binding cannot have one sample, so without
        MSAA the shader reads plain textures; textureLoad's third argument
        then is the mip level 0 instead of the sample index 0, and every
        pixel is shaded once.
        """
        if sample_count not in self._variants:
            device = self._context.device
            multisampled = sample_count > 1
            replacements = {"const GBUFFER_SAMPLES: i32 = 4;": f"const GBUFFER_SAMPLES: i32 = {sample_count};"}
            if not multisampled:
                replacements |= {
                    "texture_depth_multisampled_2d": "texture_depth_2d",
                    "texture_multisampled_2d<f32>": "texture_2d<f32>",
                    _FIRST_SAMPLE: "const FIRST_SAMPLE: vec2<f32> = vec2<f32>(0.5);",
                }
            shader = self._context.shader("deferred_lighting.wgsl", replacements)
            layout = self._create_layout(multisampled)
            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[layout])

            def pipeline(entry_point: str, edges: str | None, mark: bool = False) -> wgpu.GPURenderPipeline:
                """``edges``: the stencil test (None: no stencil); ``mark``:
                write the edge value instead of color."""
                stencil = None
                write = 0 if mark else wgpu.ColorWrite.ALL
                if edges is not None:
                    operation = wgpu.StencilOperation.replace if mark else wgpu.StencilOperation.keep
                    face = {"compare": edges, "pass_op": operation}
                    stencil = {"format": EDGE_FORMAT, "stencil_front": face, "stencil_back": face}
                return device.create_render_pipeline(
                    label=f"deferred lighting x{sample_count} ({entry_point})",
                    layout=pipeline_layout,
                    vertex={"module": shader, "entry_point": "vs_fullscreen", "buffers": []},
                    primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
                    depth_stencil=stencil,
                    multisample={"count": sample_count},
                    fragment={"module": shader, "entry_point": entry_point, "targets": [{"format": HDR_FORMAT, "write_mask": write}]},
                )

            if multisampled:
                compare = wgpu.CompareFunction
                pipelines = [
                    pipeline("fs_mark_edges", compare.always, mark=True),
                    pipeline("fs_light_pixel", compare.not_equal),
                    pipeline("fs_light_sample", compare.equal),
                ]
            else:
                pipelines = [pipeline("fs_light_pixel", None)]
            self._variants[sample_count] = (layout, pipelines)
        return self._variants[sample_count]

    def _create_layout(self, multisampled: bool) -> wgpu.GPUBindGroupLayout:
        fragment = wgpu.ShaderStage.FRAGMENT
        gbuffer = {"sample_type": wgpu.TextureSampleType.unfilterable_float, "multisampled": multisampled}
        return self._context.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": fragment, "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": fragment, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": fragment, "texture": gbuffer},
                {"binding": 3, "visibility": fragment, "texture": gbuffer},
                {"binding": 4, "visibility": fragment, "texture": gbuffer},
                {
                    "binding": 5,
                    "visibility": fragment,
                    "texture": {"sample_type": wgpu.TextureSampleType.depth, "multisampled": multisampled},
                },
                {
                    "binding": 6,
                    "visibility": fragment,
                    "texture": {"sample_type": wgpu.TextureSampleType.depth, "view_dimension": wgpu.TextureViewDimension.d2_array},
                },
                {"binding": 7, "visibility": fragment, "sampler": {"type": wgpu.SamplerBindingType.comparison}},
                {"binding": 8, "visibility": fragment, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 9, "visibility": fragment, "texture": gbuffer},
                {"binding": 10, "visibility": fragment, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            ]
        )
