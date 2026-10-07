"""Deferred lighting pass: G-buffer + lights -> linear HDR "lit" layer.

A single fullscreen triangle shades every MSAA sample of the G-buffer (see
shaders/deferred_lighting.wgsl). Lights live in one storage buffer, so the
cost grows with lights x covered samples, independent of how many meshes
or how much overdraw produced the G-buffer.
"""

from __future__ import annotations

import numpy as np
import wgpu

from .gpu import HDR_FORMAT, GPUContext, GrowableBuffer
from .shadow_pass import ShadowPass


class LightingPass:
    def __init__(self, context: GPUContext, frame_uniform: wgpu.GPUBuffer) -> None:
        self._context = context
        self._frame_uniform = frame_uniform
        self._lights = GrowableBuffer(context, "lights", wgpu.BufferUsage.STORAGE)
        self._scope_parents = GrowableBuffer(context, "scope parents", wgpu.BufferUsage.STORAGE)
        self._variants: dict[int, tuple[wgpu.GPUBindGroupLayout, wgpu.GPURenderPipeline]] = {}
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
        render_pass.set_pipeline(self._variant(self._sample_count)[1])
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.draw(3)  # one fullscreen triangle

    def _variant(self, sample_count: int) -> tuple[wgpu.GPUBindGroupLayout, wgpu.GPURenderPipeline]:
        """Layout and pipeline for multisampled or single-sampled G-buffers.

        A multisampled texture binding cannot have one sample, so without
        MSAA the shader reads plain textures; textureLoad's third argument
        then is the mip level 0 instead of the sample index 0.
        """
        if sample_count not in self._variants:
            device = self._context.device
            multisampled = sample_count > 1
            replacements = {} if multisampled else {
                "texture_depth_multisampled_2d": "texture_depth_2d",
                "texture_multisampled_2d<f32>": "texture_2d<f32>",
            }
            shader = self._context.shader("deferred_lighting.wgsl", replacements)
            layout = self._create_layout(multisampled)
            pipeline = device.create_render_pipeline(
                label=f"deferred lighting x{sample_count}",
                layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
                vertex={"module": shader, "entry_point": "vs_fullscreen", "buffers": []},
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
                multisample={"count": sample_count},
                fragment={"module": shader, "entry_point": "fs_light", "targets": [{"format": HDR_FORMAT}]},
            )
            self._variants[sample_count] = (layout, pipeline)
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
