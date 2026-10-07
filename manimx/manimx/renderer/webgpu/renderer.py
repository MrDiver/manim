"""WebGPU renderer: scene tree + config in, image out.

    renderer = WGPURenderer()
    image = renderer.render(scene, RenderConfig())   # (H, W, 4) uint8

The renderer has no scene state of its own. Each call flattens the given
tree, reuses cached per-object data (fill meshes) where the object's
versioned fields are unchanged, uploads everything into shared buffers and
encodes the passes:

    for each clip scope (parents first):
        mask pass: draw the mask shapes into a scratch target,
                   copy the result into the scope's layer of the mask array
    frame pass: clear -> lit meshes (Shape3D) -> opaque vector draws
                -> transparent vector draws (sorted); clipped content
                multiplies its alpha by its mask layer

:meth:`WGPURenderer.render_texture` returns the resolved GPU texture without
reading it back, which is the hook for a later compositor.
"""

from __future__ import annotations

import numpy as np
import wgpu

from ...objects import RenderObject
from ..base import Renderer
from ..config import RenderConfig
from ..scene_graph import flatten_scene
from .gpu import COLOR_FORMAT, DEPTH_FORMAT, GPUContext
from .mesh_batch import MeshBatch
from .mesh_pass import MeshPass
from .vector_batch import VectorBatch
from .vector_pass import VectorPass

# Mirrors `struct Frame` in shaders/frame.wgsl.
FRAME_DTYPE = np.dtype(
    [
        ("view", "<f4", (4, 4)),
        ("projection", "<f4", (4, 4)),
        ("view_projection", "<f4", (4, 4)),
        ("inverse_view", "<f4", (4, 4)),
        ("inverse_projection", "<f4", (4, 4)),
        ("camera_position", "<f4", 4),
        ("viewport", "<f4", 4),
        ("time", "<f4"),
        ("layer_depth_step", "<f4"),
        ("light_count", "<u4"),
        ("exposure", "<f4"),
        ("ambient", "<f4", 4),
    ]
)
assert FRAME_DTYPE.itemsize == 384


class WGPURenderer(Renderer):
    def __init__(self, context: GPUContext | None = None) -> None:
        super().__init__()
        self.context = context or GPUContext.create()
        device = self.context.device
        self._frame_uniform = device.create_buffer(
            label="frame",
            size=FRAME_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._vector_pass = VectorPass(self.context, self._frame_uniform)
        self._mesh_pass = MeshPass(self.context, self._frame_uniform)
        self._targets: _RenderTargets | None = None

    def render(self, root: RenderObject, config: RenderConfig) -> np.ndarray:
        """Render ``root`` and read the image back as ``(H, W, 4)`` uint8 RGBA."""
        texture = self.render_texture(root, config)
        data = self.context.device.queue.read_texture(
            {"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
            {"offset": 0, "bytes_per_row": 4 * config.pixel_width, "rows_per_image": config.pixel_height},
            (config.pixel_width, config.pixel_height, 1),
        )
        return np.frombuffer(data, dtype=np.uint8).reshape(config.pixel_height, config.pixel_width, 4).copy()

    def render_texture(self, root: RenderObject, config: RenderConfig) -> wgpu.GPUTexture:
        """Render ``root`` into a single-sampled RGBA texture on the GPU."""
        scene = flatten_scene(root)
        vectors = VectorBatch.build(scene, self.cache, config.camera.position)
        meshes = MeshBatch.build(scene, self.cache, config)
        targets = self._render_targets(config, mask_count=len(scene.clips))

        device = self.context.device
        self._vector_pass.prepare(vectors, targets.mask_layers_view)
        light_count = self._mesh_pass.prepare(meshes, targets.mask_layers_view)
        device.queue.write_buffer(self._frame_uniform, 0, _frame_uniform_data(config, light_count))

        encoder = device.create_command_encoder(label="frame")
        for scope in range(len(scene.clips)):
            mask_pass = encoder.begin_render_pass(color_attachments=[targets.mask_attachment()])
            self._vector_pass.draw_mask(mask_pass, config.msaa_samples, scope)
            mask_pass.end()
            targets.copy_mask_to_layer(encoder, scope)

        frame_pass = encoder.begin_render_pass(
            color_attachments=[targets.color_attachment(config.background_rgba)],
            depth_stencil_attachment=targets.depth_attachment(),
        )
        self._mesh_pass.draw(frame_pass, config.msaa_samples)
        self._vector_pass.draw_frame(frame_pass, config.msaa_samples)
        frame_pass.end()
        device.queue.submit([encoder.finish()])
        return targets.output

    def _render_targets(self, config: RenderConfig, mask_count: int) -> _RenderTargets:
        size = (config.pixel_width, config.pixel_height, config.msaa_samples)
        targets = self._targets
        if targets is None or targets.size != size or targets.mask_capacity < mask_count:
            capacity = max(mask_count, 2 * targets.mask_capacity if targets else 1)
            self._targets = targets = _RenderTargets(self.context.device, *size, mask_capacity=capacity)
        return targets


def _frame_uniform_data(config: RenderConfig, light_count: int) -> np.ndarray:
    camera = config.camera
    view = camera.view_matrix()
    projection = camera.projection_matrix(config.aspect_ratio)
    frame = np.zeros((), FRAME_DTYPE)
    # numpy matrices are row-major, WGSL's are column-major: transpose.
    frame["view"] = view.T
    frame["projection"] = projection.T
    frame["view_projection"] = (projection @ view).T
    frame["inverse_view"] = np.linalg.inv(view).T
    frame["inverse_projection"] = np.linalg.inv(projection).T
    frame["camera_position"] = (*camera.position, 1.0)
    frame["viewport"] = (
        config.pixel_width,
        config.pixel_height,
        1 / config.pixel_width,
        1 / config.pixel_height,
    )
    frame["time"] = config.time
    frame["layer_depth_step"] = config.layer_depth_step
    frame["light_count"] = light_count
    frame["exposure"] = config.exposure
    frame["ambient"][:3] = config.ambient_linear
    return frame


class _RenderTargets:
    """Frame attachments, the resolved output, and the clip mask textures."""

    def __init__(
        self, device: wgpu.GPUDevice, width: int, height: int, samples: int, mask_capacity: int
    ) -> None:
        self.size = (width, height, samples)
        self.mask_capacity = mask_capacity
        self.output = device.create_texture(
            label="output",
            size=(width, height, 1),
            format=COLOR_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._msaa_color = (
            device.create_texture(
                label="msaa color",
                size=(width, height, 1),
                format=COLOR_FORMAT,
                sample_count=samples,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            if samples > 1
            else None
        )
        self._depth = device.create_texture(
            label="depth",
            size=(width, height, 1),
            format=DEPTH_FORMAT,
            sample_count=samples,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        # One layer per clip scope, sampled by clipped content. Mask passes
        # cannot render into it directly while it is bound for sampling, so
        # they render into a scratch target that is copied into the layer.
        self._mask_layers = device.create_texture(
            label="mask layers",
            size=(width, height, mask_capacity),
            format=COLOR_FORMAT,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.mask_layers_view = self._mask_layers.create_view(
            dimension=wgpu.TextureViewDimension.d2_array
        )
        self._mask_scratch = device.create_texture(
            label="mask scratch",
            size=(width, height, 1),
            format=COLOR_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self._mask_msaa = (
            device.create_texture(
                label="mask msaa",
                size=(width, height, 1),
                format=COLOR_FORMAT,
                sample_count=samples,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            if samples > 1
            else None
        )

    def mask_attachment(self) -> dict:
        """Transparent black scratch target; mask shapes accumulate alpha."""
        return _attachment(self._mask_scratch, self._mask_msaa, (0.0, 0.0, 0.0, 0.0))

    def copy_mask_to_layer(self, encoder: wgpu.GPUCommandEncoder, layer: int) -> None:
        width, height, _ = self.size
        encoder.copy_texture_to_texture(
            {"texture": self._mask_scratch, "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": self._mask_layers, "mip_level": 0, "origin": (0, 0, layer)},
            (width, height, 1),
        )

    def color_attachment(self, clear: tuple[float, float, float, float]) -> dict:
        return _attachment(self.output, self._msaa_color, clear)

    def depth_attachment(self) -> dict:
        return {
            "view": self._depth.create_view(),
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        }


def _attachment(
    output: wgpu.GPUTexture,
    msaa: wgpu.GPUTexture | None,
    clear: tuple[float, float, float, float],
) -> dict:
    """Color attachment that clears, renders, and ends up in ``output``."""
    attachment = {
        "clear_value": clear,
        "load_op": wgpu.LoadOp.clear,
        "store_op": wgpu.StoreOp.store,
    }
    if msaa is None:
        attachment["view"] = output.create_view()
    else:
        # Render into the multisampled texture, resolve into the output.
        attachment["view"] = msaa.create_view()
        attachment["resolve_target"] = output.create_view()
        attachment["store_op"] = wgpu.StoreOp.discard
    return attachment
