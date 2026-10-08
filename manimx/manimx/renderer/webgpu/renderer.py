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

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import wgpu

from ...objects import RenderObject, Shape3D
from ..base import Renderer
from ..config import RenderConfig
from ..effects import Effect
from ..scene_graph import flatten_scene
from .frame import FRAME_DTYPE, camera_frame
from .gpu import GPUContext, ResourceCache
from .lighting_pass import LightingPass
from .mesh_batch import MeshBatch
from .mesh_pass import MeshPass
from .profiler import GpuProfiler
from .shadow_pass import ShadowPass
from .shadow_views import build_shadow_views
from .targets import RenderLayers, RenderTargets
from .vector_batch import VectorBatch
from .vector_pass import VectorPass

if TYPE_CHECKING:
    from .composer import FrameComposer

CACHE_MAX_AGE = 1024
"""Renders after which unused per-object cache entries are dropped."""

_shared: WGPURenderer | None = None
_shared_lock = threading.Lock()


class WGPURenderer(Renderer):
    """The WebGPU renderer, meant to be used as a long-lived service.

    Everything it builds is kept across calls: the device, pipelines,
    buffers, render targets, views and bind groups, per-object data
    (tessellations, stroke records) and composers per effect chain. A
    process should therefore render through one instance,
    :meth:`shared`, instead of creating renderers per call. Rendering is
    not thread-safe; one thread renders at a time.
    """

    @classmethod
    def shared(cls) -> WGPURenderer:
        """The process's renderer service, created on first use."""
        global _shared
        with _shared_lock:
            if _shared is None:
                _shared = cls()
            return _shared

    def __init__(self, context: GPUContext | None = None, profile: bool = False) -> None:
        super().__init__()
        self.context = context or GPUContext.create()
        self.profiler = GpuProfiler(self.context.device)
        self.resources = ResourceCache(self.context.device)
        """Views and bind groups shared by the renderer and its composers."""
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
        self._empty_3d: RenderTargets | None = None
        """Targets whose G-buffer and lit layer are known to be empty."""
        self._composers: dict[tuple, Any] = {}

    @property
    def frame_uniform(self) -> wgpu.GPUBuffer:
        """Camera, viewport and time of the latest render (frame.wgsl)."""
        return self._frame_uniform

    def render(self, root: RenderObject, config: RenderConfig) -> np.ndarray:
        """Render ``root`` to ``(H, W, 4)`` uint8 RGBA with default compositing."""
        return self.composer().compose(root, config)

    def composer(self, effects: Sequence[Effect] = ()) -> FrameComposer:
        """The composer for an effect chain, kept like everything else (its
        pipelines and textures survive across calls)."""
        key = tuple(effects)
        composer = self._composers.get(key)
        if composer is None:
            from .composer import FrameComposer  # the composer imports this module

            composer = self._composers[key] = FrameComposer(self, key)
        return composer

    def render_layers(self, root: RenderObject, config: RenderConfig, encoder: wgpu.GPUCommandEncoder | None = None) -> RenderLayers:
        """Render ``root`` into the lit and vector layers (GPU textures).

        With ``encoder`` the passes are recorded into it and the caller
        submits (the composer adds its passes to the same submission);
        without, they are submitted right away.
        """
        if self.cache.next_generation() % CACHE_MAX_AGE == 0:
            self.cache.sweep(CACHE_MAX_AGE)
        scene = flatten_scene(root)
        vectors = VectorBatch.build(scene, self.cache, config.camera.position)
        targets = self._render_targets(config, mask_count=len(scene.clips))
        samples = config.msaa_samples
        # Lights, shadows and the deferred passes only affect meshes. A
        # frame without meshes skips them; its G-buffer (whose depth the
        # vector pass tests against) and lit layer only need to be empty,
        # and stay empty as long as the next frames have no meshes either.
        has_meshes = any(isinstance(item.obj, Shape3D) for item in scene.items)
        clear_3d = not has_meshes and self._empty_3d is not targets
        meshes = MeshBatch.build(scene, self.cache, config) if has_meshes else None

        device = self.context.device
        device.queue.write_buffer(self._frame_uniform, 0, camera_frame(config, len(meshes.lights) if meshes else 0))
        self._vector_pass.prepare(vectors, targets.mask_layers_view, targets.gbuffer_views[-1], samples > 1)
        if meshes is not None:
            self._mesh_pass.prepare(meshes, targets.mask_layers_view)
            shadows = build_shadow_views(scene, meshes.light_items, meshes.lights, self.cache, config)
            shadows_changed = self._shadow_pass.prepare(shadows, config.shadow_resolution)
            scope_parents = np.array([scope.parent for scope in scene.clips], dtype=np.int32)
            self._lighting_pass.prepare(meshes.lights, scope_parents, targets.gbuffer_views, samples, self._shadow_pass, shadows_changed)

        own_encoder = encoder is None
        if encoder is None:
            encoder = device.create_command_encoder(label="frame")
        if meshes is not None:
            self._shadow_pass.draw(encoder, self._mesh_pass, self._vector_pass, self.profiler)
        for scope in range(len(scene.clips)):
            mask_pass = encoder.begin_render_pass(
                color_attachments=[targets.mask_attachment()], timestamp_writes=self.profiler.pass_timing("masks")
            )
            self._vector_pass.draw_mask(mask_pass, samples, scope)
            mask_pass.end()
            targets.copy_mask_to_layer(encoder, scope)

        if meshes is not None or clear_3d:
            geometry_pass = encoder.begin_render_pass(
                color_attachments=targets.gbuffer_attachments(),
                depth_stencil_attachment=targets.depth_attachment(),
                timestamp_writes=self.profiler.pass_timing("geometry"),
            )
            if meshes is not None:
                self._mesh_pass.draw(geometry_pass, samples)
            geometry_pass.end()

            lighting_pass = encoder.begin_render_pass(
                color_attachments=[targets.lit_attachment()], timestamp_writes=self.profiler.pass_timing("lighting")
            )
            if meshes is not None:
                self._lighting_pass.draw(lighting_pass)
            lighting_pass.end()
        self._empty_3d = targets if meshes is None else None

        vector_pass = encoder.begin_render_pass(
            color_attachments=[targets.vector_attachment()],
            depth_stencil_attachment=targets.vector_depth_attachment(),
            timestamp_writes=self.profiler.pass_timing("vectors"),
        )
        self._vector_pass.draw_frame(vector_pass, samples)
        vector_pass.end()

        if own_encoder:
            device.queue.submit([encoder.finish()])
        return targets.layers

    def _render_targets(self, config: RenderConfig, mask_count: int) -> RenderTargets:
        size = (config.pixel_width, config.pixel_height, config.msaa_samples)
        targets = self._targets
        if targets is None or targets.size != size or targets.mask_capacity < mask_count:
            capacity = max(mask_count, 2 * targets.mask_capacity if targets else 1)
            self._targets = targets = RenderTargets(self.context.device, *size, mask_capacity=capacity)
        return targets
