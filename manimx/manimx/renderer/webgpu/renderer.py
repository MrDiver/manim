"""WebGPU renderer: scene tree + config in, image layers out.

    renderer = WGPURenderer()
    image = renderer.render(scene, RenderConfig())   # (H, W, 4) uint8

The renderer has no scene state of its own. Each call flattens the given
tree, reuses cached per-object data (fill meshes, packed vertices) where
the object's versioned fields are unchanged, uploads everything into shared
buffers and encodes these passes:

    mask passes    one per clip scope (parents first): mask shapes into a
                   scratch target, copied into the scope's mask layer
    shadow passes  one per shadow view: casters' depth seen from a light
    geometry pass  Shape3D meshes -> multisampled G-buffer + depth
    lighting pass  every G-buffer sample x every light -> "lit" HDR layer
    vector pass    Shape2D and Curve3D -> "vector" layer, depth-tested
                   against the meshes; opaque draws, then sorted
                   transparent draws

:meth:`WGPURenderer.render_layers` stops there and returns both layers as
GPU textures. A :class:`~.composer.FrameComposer` turns layers into a frame
(bloom, tone mapping, compositing, motion blur); :meth:`WGPURenderer.render`
is a composer with default settings plus a read-back.
"""

from __future__ import annotations

import numpy as np
import wgpu

from ...objects import RenderObject
from ..base import Renderer
from ..config import RenderConfig
from ..scene_graph import flatten_scene
from .frame import FRAME_DTYPE, camera_frame
from .gpu import GPUContext
from .lighting_pass import LightingPass
from .mesh_batch import MeshBatch
from .mesh_pass import MeshPass
from .profiler import GpuProfiler
from .shadow_pass import ShadowPass
from .shadow_views import build_shadow_views
from .targets import RenderLayers, RenderTargets
from .vector_batch import VectorBatch
from .vector_pass import VectorPass

class WGPURenderer(Renderer):
    def __init__(self, context: GPUContext | None = None, profile: bool = False) -> None:
        super().__init__()
        self.context = context or GPUContext.create()
        self.profiler = GpuProfiler(self.context.device)
        self.profiler.enabled &= profile
        device = self.context.device
        self._frame_uniform = device.create_buffer(
            label="frame",
            size=FRAME_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._vector_pass = VectorPass(self.context, self._frame_uniform)
        self._mesh_pass = MeshPass(self.context, self._frame_uniform)
        self._lighting_pass = LightingPass(self.context, self._frame_uniform)
        self._shadow_pass = ShadowPass(self.context)
        self._targets: RenderTargets | None = None
        self._composer = None

    @property
    def frame_uniform(self) -> wgpu.GPUBuffer:
        """Camera, viewport and time of the latest render (frame.wgsl)."""
        return self._frame_uniform

    def render(self, root: RenderObject, config: RenderConfig) -> np.ndarray:
        """Render ``root`` to ``(H, W, 4)`` uint8 RGBA with default compositing."""
        if self._composer is None:
            from .composer import FrameComposer  # the composer imports this module

            self._composer = FrameComposer(self)
        return self._composer.compose(root, config)

    def render_layers(self, root: RenderObject, config: RenderConfig) -> RenderLayers:
        """Render ``root`` into the lit and vector layers (GPU textures)."""
        scene = flatten_scene(root)
        vectors = VectorBatch.build(scene, self.cache, config.camera.position)
        meshes = MeshBatch.build(scene, self.cache, config)
        targets = self._render_targets(config, mask_count=len(scene.clips))
        samples = config.msaa_samples

        device = self.context.device
        device.queue.write_buffer(self._frame_uniform, 0, camera_frame(config, len(meshes.lights)))
        self._vector_pass.prepare(vectors, targets.mask_layers_view, targets.gbuffer_views[-1], samples > 1)
        self._mesh_pass.prepare(meshes, targets.mask_layers_view)
        shadows = build_shadow_views(scene, meshes.light_items, meshes.lights, self.cache, config)
        shadows_changed = self._shadow_pass.prepare(shadows, config.shadow_resolution)
        scope_parents = np.array([scope.parent for scope in scene.clips], dtype=np.int32)
        self._lighting_pass.prepare(meshes.lights, scope_parents, targets.gbuffer_views, samples, self._shadow_pass, shadows_changed)

        encoder = device.create_command_encoder(label="frame")
        self._shadow_pass.draw(encoder, self._mesh_pass, self._vector_pass, self.profiler)
        for scope in range(len(scene.clips)):
            mask_pass = encoder.begin_render_pass(
                color_attachments=[targets.mask_attachment()], timestamp_writes=self.profiler.pass_timing("masks")
            )
            self._vector_pass.draw_mask(mask_pass, samples, scope)
            mask_pass.end()
            targets.copy_mask_to_layer(encoder, scope)

        geometry_pass = encoder.begin_render_pass(
            color_attachments=targets.gbuffer_attachments(),
            depth_stencil_attachment=targets.depth_attachment(),
            timestamp_writes=self.profiler.pass_timing("geometry"),
        )
        self._mesh_pass.draw(geometry_pass, samples)
        geometry_pass.end()

        lighting_pass = encoder.begin_render_pass(
            color_attachments=[targets.lit_attachment()], timestamp_writes=self.profiler.pass_timing("lighting")
        )
        self._lighting_pass.draw(lighting_pass)
        lighting_pass.end()

        vector_pass = encoder.begin_render_pass(
            color_attachments=[targets.vector_attachment()],
            depth_stencil_attachment=targets.vector_depth_attachment(),
            timestamp_writes=self.profiler.pass_timing("vectors"),
        )
        self._vector_pass.draw_frame(vector_pass, samples)
        vector_pass.end()

        device.queue.submit([encoder.finish()])
        return targets.layers

    def _render_targets(self, config: RenderConfig, mask_count: int) -> RenderTargets:
        size = (config.pixel_width, config.pixel_height, config.msaa_samples)
        targets = self._targets
        if targets is None or targets.size != size or targets.mask_capacity < mask_count:
            capacity = max(mask_count, 2 * targets.mask_capacity if targets else 1)
            self._targets = targets = RenderTargets(self.context.device, *size, mask_capacity=capacity)
        return targets
