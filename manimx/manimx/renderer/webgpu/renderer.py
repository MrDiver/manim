"""WebGPU renderer: scene tree + config in, image out.

    renderer = WGPURenderer()
    image = renderer.render(scene, RenderConfig())   # (H, W, 4) uint8

The renderer has no scene state of its own. Each call flattens the given
tree, reuses cached per-object data (fill meshes) where the object's
versioned fields are unchanged, uploads everything into shared buffers and
encodes one render pass:

    clear -> opaque vector draws -> transparent vector draws (sorted)

:meth:`WGPURenderer.render_texture` returns the resolved GPU texture without
reading it back, which is the hook for a later compositor.
"""

from __future__ import annotations

import numpy as np
import wgpu

from ...objects import RenderObject
from ..base import Renderer
from ..config import RenderConfig
from ..scene_graph import flatten
from .gpu import COLOR_FORMAT, DEPTH_FORMAT, GPUContext
from .vector_batch import VectorBatch
from .vector_pass import VectorPass

# Mirrors `struct Frame` in the shaders.
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
        ("_pad", "<f4", 2),
    ]
)
assert FRAME_DTYPE.itemsize == 368


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
        items = flatten(root)
        batch = VectorBatch.build(items, self.cache, config.camera.position)

        device = self.context.device
        device.queue.write_buffer(self._frame_uniform, 0, _frame_uniform_data(config))
        self._vector_pass.prepare(batch)

        targets = self._render_targets(config)
        encoder = device.create_command_encoder(label="frame")
        render_pass = encoder.begin_render_pass(
            color_attachments=[targets.color_attachment(config.background_rgba)],
            depth_stencil_attachment=targets.depth_attachment(),
        )
        self._vector_pass.draw(render_pass, config.msaa_samples)
        render_pass.end()
        device.queue.submit([encoder.finish()])
        return targets.output

    def _render_targets(self, config: RenderConfig) -> _RenderTargets:
        size = (config.pixel_width, config.pixel_height, config.msaa_samples)
        if self._targets is None or self._targets.size != size:
            self._targets = _RenderTargets(self.context.device, *size)
        return self._targets


def _frame_uniform_data(config: RenderConfig) -> np.ndarray:
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
    return frame


class _RenderTargets:
    """Multisampled color + depth attachments and the resolved output."""

    def __init__(self, device: wgpu.GPUDevice, width: int, height: int, samples: int) -> None:
        self.size = (width, height, samples)
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

    def color_attachment(self, clear: tuple[float, float, float, float]) -> dict:
        attachment = {
            "clear_value": clear,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
        }
        if self._msaa_color is None:
            attachment["view"] = self.output.create_view()
        else:
            # Render into the multisampled texture, resolve into the output.
            attachment["view"] = self._msaa_color.create_view()
            attachment["resolve_target"] = self.output.create_view()
            attachment["store_op"] = wgpu.StoreOp.discard
        return attachment

    def depth_attachment(self) -> dict:
        return {
            "view": self._depth.create_view(),
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        }
