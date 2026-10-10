"""WebGPU renderer: a frame packet in, image layers out.

The renderer is the GPU stage of the render chain (see packer.py): it
takes a :class:`~.packer.FramePacket`, uploads its records into
persistent buffers and encodes these passes:

    mask passes    one per clip scope (parents first): mask shapes into a
                   scratch target, copied into the scope's mask layer
    shadow passes  one per shadow view: casters' depth seen from a light
    geometry pass  RShape3D meshes -> multisampled G-buffer + depth
    lighting pass  every G-buffer sample x every light -> "lit" HDR layer
    vector pass    RShape2D and RCurve3D -> "vector" layer, depth-tested
                   against the meshes; opaque draws, then sorted
                   transparent draws

It never sees a render node and decides nothing about the scene: what to
draw, and what changed, is in the packet. :meth:`WGPURenderer.render_layers`
returns both layers as GPU textures; a :class:`~.composer.FrameComposer`
turns them into a frame. :class:`~.service.RenderService` runs the whole
chain.
"""

from __future__ import annotations

import wgpu

from ..config import RenderConfig
from .frame import FRAME_DTYPE
from .gpu import HDR_FORMAT, GPUContext, ResourceCache
from .lighting_pass import LightingPass
from .mesh_pass import MeshPass
from .packer import FramePacket
from .profiler import GpuProfiler
from .shadow_pass import ShadowPass
from .targets import RenderLayers, RenderTargets
from .vector_pass import VectorPass


class WGPURenderer:
    """The GPU stage: packets in, layers out (see the module docs).

    Everything it builds is kept across calls: the device, pipelines,
    buffers, render targets, views and bind groups. Rendering is not
    thread-safe; one thread renders at a time.
    """

    def __init__(self, context: GPUContext | None = None, profile: bool = False) -> None:
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
        self._source: object = None
        """The packer of the previous packet: the vector buffers hold its
        records, so its next packet only uploads what changed."""
        self._targets: RenderTargets | None = None
        self._empty_3d: RenderTargets | None = None
        """Targets whose G-buffer and lit layer are known to be empty."""
        self._camera_views: wgpu.GPUTexture | None = None
        self._camera_views_view: wgpu.GPUTextureView | None = None
        self.ensure_camera_views(1, (1, 1))

    @property
    def frame_uniform(self) -> wgpu.GPUBuffer:
        """Camera, viewport and time of the latest render (frame.wgsl)."""
        return self._frame_uniform

    def ensure_camera_views(self, count: int, size: tuple[int, int]) -> None:
        """Make the camera view texture hold at least ``count`` layers of
        ``size`` pixels (composers render the pictures into it, the vector
        pass samples it; see ``RenderConfig.camera_views``)."""
        texture = self._camera_views
        if texture is not None and texture.size[2] >= count and tuple(texture.size[:2]) == tuple(size):
            return
        self._camera_views = self.context.device.create_texture(
            label="camera views",
            size=(size[0], size[1], max(count, 1)),
            format=HDR_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._camera_views_view = self._camera_views.create_view(dimension=wgpu.TextureViewDimension.d2_array)

    def camera_view_layer(self, layer: int) -> wgpu.GPUTextureView:
        """Layer ``layer`` of the camera view texture, to render a picture into."""
        assert self._camera_views is not None
        return self.resources.view(self._camera_views, dimension=wgpu.TextureViewDimension.d2, base_array_layer=layer, array_layer_count=1)

    def render_layers(self, packet: FramePacket, encoder: wgpu.GPUCommandEncoder | None = None) -> RenderLayers:
        """Render ``packet`` into the lit and vector layers (GPU textures).

        With ``encoder`` the passes are recorded into it and the caller
        submits (the composer adds its passes to the same submission);
        without, they are submitted right away.
        """
        config = packet.config
        meshes = packet.meshes
        targets = self._render_targets(config, mask_count=packet.mask_count)
        samples = config.msaa_samples
        # A frame without meshes skips the 3D passes; its G-buffer (whose
        # depth the vector pass tests against) and lit layer only need to
        # be empty, and stay empty as long as the next frames have no
        # meshes either.
        clear_3d = meshes is None and self._empty_3d is not targets

        device = self.context.device
        device.queue.write_buffer(self._frame_uniform, 0, packet.frame)
        full_upload = packet.source is not self._source
        self._source = packet.source
        self._vector_pass.prepare(
            packet.vectors, targets.mask_layers_view, targets.gbuffer_views[-1], samples > 1, full_upload, self._camera_views_view
        )
        if meshes is not None:
            self._mesh_pass.prepare(meshes, targets.mask_layers_view, full_upload)
            shadows_changed = self._shadow_pass.prepare(packet.shadows, config.shadow_resolution, all_layers=full_upload)
            self._lighting_pass.prepare(meshes.lights, packet.scope_parents, targets.gbuffer_views, samples, self._shadow_pass, shadows_changed)

        own_encoder = encoder is None
        if encoder is None:
            encoder = device.create_command_encoder(label="frame")
        if meshes is not None:
            self._shadow_pass.draw(encoder, self._mesh_pass, self._vector_pass, self.profiler)
        for scope in range(packet.mask_count):
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
            capacity = targets.mask_capacity if targets else 1
            if capacity < mask_count:
                capacity = max(mask_count, 2 * capacity)  # grow in steps, not by one
            self._targets = targets = RenderTargets(self.context.device, *size, mask_capacity=capacity)
        return targets
