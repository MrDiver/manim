"""All textures one frame renders into, recreated when the size changes.

    masks       one RGBA layer per clip scope (+ a scratch target to draw into)
    G-buffer    3 color targets + depth, multisampled (see gbuffer.wgsl)
    lit         multisampled HDR target of the lighting pass -> resolved,
                with a stencil marking the pixels on edges
    vector      multisampled premultiplied RGBA of 2D shapes and curves,
                depth-tested against the G-buffer depth -> resolved

The resolved ``lit`` and ``vector`` textures are the renderer's output
(:class:`RenderLayers`); the frame composer combines them into an image.
"""

from __future__ import annotations

from dataclasses import dataclass

import wgpu

from .gpu import COLOR_FORMAT, DEPTH_FORMAT, EDGE_FORMAT, GBUFFER_FORMATS, HDR_FORMAT

_ATTACHMENT = wgpu.TextureUsage.RENDER_ATTACHMENT
_SAMPLED = wgpu.TextureUsage.TEXTURE_BINDING


@dataclass(frozen=True, slots=True)
class RenderLayers:
    """What one render produces, as resolved single-sample GPU textures.

    The textures belong to the renderer and are overwritten by its next
    render, so consume them (e.g. composite them) before rendering again.
    """

    lit: wgpu.GPUTexture
    """Linear HDR radiance of lit meshes, premultiplied by mesh coverage."""
    vector: wgpu.GPUTexture
    """sRGB-valued, premultiplied 2D shapes and curves (Manim colors, exact)."""
    width: int
    height: int
    mesh_depth: wgpu.GPUTextureView | None = None
    """The meshes' depth buffer (multisampled like the frame)."""
    vector_depth: wgpu.GPUTextureView | None = None
    """The 2D shapes' and curves' depth buffer (multisampled like the frame)."""
    samples: int = 1


class RenderTargets:
    def __init__(self, device: wgpu.GPUDevice, width: int, height: int, samples: int, mask_capacity: int) -> None:
        self.size = (width, height, samples)
        self.mask_capacity = mask_capacity

        def texture(label: str, format: str, sample_count: int = samples, usage: int = _ATTACHMENT, layers: int = 1) -> wgpu.GPUTexture:
            return device.create_texture(
                label=label, size=(width, height, layers), format=format, sample_count=sample_count, usage=usage
            )

        # G-buffer: sampled per sample by the lighting pass.
        self.gbuffer = [texture(f"gbuffer {i}", f, usage=_ATTACHMENT | _SAMPLED) for i, f in enumerate(GBUFFER_FORMATS)]
        self.depth = texture("depth", DEPTH_FORMAT, usage=_ATTACHMENT | _SAMPLED)
        # The vector pass orders shapes with biased depth in its own buffer
        # and tests against the meshes' depth (above) explicitly.
        self.vector_depth = texture("vector depth", DEPTH_FORMAT, usage=_ATTACHMENT | _SAMPLED)

        # Resolved outputs, read by the composer.
        self.lit = texture("lit", HDR_FORMAT, sample_count=1, usage=_ATTACHMENT | _SAMPLED | wgpu.TextureUsage.COPY_SRC)
        self.vector = texture("vector", COLOR_FORMAT, sample_count=1, usage=_ATTACHMENT | _SAMPLED | wgpu.TextureUsage.COPY_SRC)
        self._lit_msaa = texture("lit msaa", HDR_FORMAT) if samples > 1 else None
        self._edges = texture("edges", EDGE_FORMAT) if samples > 1 else None
        self._vector_msaa = texture("vector msaa", COLOR_FORMAT) if samples > 1 else None

        # One layer per clip scope, sampled by clipped content. Mask passes
        # cannot render into it while it is bound for sampling, so they
        # render into a scratch target that is copied into the layer.
        self._mask_layers = texture("mask layers", COLOR_FORMAT, sample_count=1, usage=_SAMPLED | wgpu.TextureUsage.COPY_DST, layers=mask_capacity)
        self.mask_layers_view = self._mask_layers.create_view(dimension=wgpu.TextureViewDimension.d2_array)
        self._mask_scratch = texture("mask scratch", COLOR_FORMAT, sample_count=1, usage=_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self._mask_msaa = texture("mask msaa", COLOR_FORMAT) if samples > 1 else None

        # Color targets in GBUFFER_FORMATS order, then the depth buffer.
        self.gbuffer_views = (*(t.create_view() for t in self.gbuffer), self.depth.create_view())

        # Attachments are built once: every frame renders into the same
        # views (creating views per frame was measurable Python overhead).
        self._mask_attachment = _color(self._mask_scratch, self._mask_msaa)
        self._lit_attachment = _color(self.lit, self._lit_msaa)
        self._edge_attachment = None
        if self._edges is not None:
            self._edge_attachment = {
                "view": self._edges.create_view(),
                "stencil_clear_value": 0,
                "stencil_load_op": wgpu.LoadOp.clear,
                "stencil_store_op": wgpu.StoreOp.discard,
            }
        self._vector_attachment = _color(self.vector, self._vector_msaa)
        self._gbuffer_attachments = [
            {"view": view, "clear_value": (0, 0, 0, 0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}
            for view in self.gbuffer_views[:-1]
        ]
        self._depth_attachment = _depth(self.gbuffer_views[-1])
        self.vector_depth_view = self.vector_depth.create_view()
        self._vector_depth_attachment = _depth(self.vector_depth_view)

    @property
    def layers(self) -> RenderLayers:
        width, height, samples = self.size
        return RenderLayers(self.lit, self.vector, width, height, self.gbuffer_views[-1], self.vector_depth_view, samples)

    # -- attachments ----------------------------------------------------------

    def mask_attachment(self) -> dict:
        """Transparent scratch target; mask shapes accumulate alpha."""
        return self._mask_attachment

    def copy_mask_to_layer(self, encoder: wgpu.GPUCommandEncoder, layer: int) -> None:
        width, height, _ = self.size
        encoder.copy_texture_to_texture(
            {"texture": self._mask_scratch, "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": self._mask_layers, "mip_level": 0, "origin": (0, 0, layer)},
            (width, height, 1),
        )

    def gbuffer_attachments(self) -> list[dict]:
        return self._gbuffer_attachments

    def lit_attachment(self) -> dict:
        return self._lit_attachment

    def edge_attachment(self) -> dict | None:
        """The lighting pass's stencil (None without MSAA)."""
        return self._edge_attachment

    def vector_attachment(self) -> dict:
        return self._vector_attachment

    def depth_attachment(self) -> dict:
        """The meshes' depth buffer, written by the geometry pass."""
        return self._depth_attachment

    def vector_depth_attachment(self) -> dict:
        """The vector pass's own depth buffer (painter's-order layers)."""
        return self._vector_depth_attachment


def _depth(view: wgpu.GPUTextureView) -> dict:
    return {
        "view": view,
        "depth_clear_value": 1.0,
        "depth_load_op": wgpu.LoadOp.clear,
        "depth_store_op": wgpu.StoreOp.store,
    }


def _color(output: wgpu.GPUTexture, msaa: wgpu.GPUTexture | None) -> dict:
    """Color attachment that clears to transparent and ends up in ``output``."""
    attachment = {"clear_value": (0, 0, 0, 0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}
    if msaa is None:
        attachment["view"] = output.create_view()
    else:
        # Render into the multisampled texture, resolve into the output.
        attachment["view"] = msaa.create_view()
        attachment["resolve_target"] = output.create_view()
        attachment["store_op"] = wgpu.StoreOp.discard
    return attachment
