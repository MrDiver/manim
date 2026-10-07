"""All textures one frame renders into, recreated when the size changes.

    masks       one RGBA layer per clip scope (+ a scratch target to draw into)
    G-buffer    3 color targets + depth, multisampled (see gbuffer.wgsl)
    lit         multisampled HDR target of the lighting pass -> resolved
    vector      multisampled premultiplied RGBA of 2D shapes and curves,
                depth-tested against the G-buffer depth -> resolved

The resolved ``lit`` and ``vector`` textures are the renderer's output
(:class:`RenderLayers`); the frame composer combines them into an image.
"""

from __future__ import annotations

from dataclasses import dataclass

import wgpu

from .gpu import COLOR_FORMAT, DEPTH_FORMAT, GBUFFER_FORMATS, HDR_FORMAT

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

        # Resolved outputs, read by the composer.
        self.lit = texture("lit", HDR_FORMAT, sample_count=1, usage=_ATTACHMENT | _SAMPLED | wgpu.TextureUsage.COPY_SRC)
        self.vector = texture("vector", COLOR_FORMAT, sample_count=1, usage=_ATTACHMENT | _SAMPLED | wgpu.TextureUsage.COPY_SRC)
        self._lit_msaa = texture("lit msaa", HDR_FORMAT) if samples > 1 else None
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

    @property
    def layers(self) -> RenderLayers:
        width, height, _ = self.size
        return RenderLayers(self.lit, self.vector, width, height)

    # -- attachments ----------------------------------------------------------

    def mask_attachment(self) -> dict:
        """Transparent scratch target; mask shapes accumulate alpha."""
        return _color(self._mask_scratch, self._mask_msaa)

    def copy_mask_to_layer(self, encoder: wgpu.GPUCommandEncoder, layer: int) -> None:
        width, height, _ = self.size
        encoder.copy_texture_to_texture(
            {"texture": self._mask_scratch, "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": self._mask_layers, "mip_level": 0, "origin": (0, 0, layer)},
            (width, height, 1),
        )

    def gbuffer_attachments(self) -> list[dict]:
        return [
            {"view": view, "clear_value": (0, 0, 0, 0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}
            for view in self.gbuffer_views[:-1]
        ]

    def lit_attachment(self) -> dict:
        return _color(self.lit, self._lit_msaa)

    def vector_attachment(self) -> dict:
        return _color(self.vector, self._vector_msaa)

    def depth_attachment(self, clear: bool) -> dict:
        """The shared depth buffer: cleared by the geometry pass, then reused
        by the vector pass so 2D shapes are hidden behind meshes."""
        return {
            "view": self.gbuffer_views[-1],
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear if clear else wgpu.LoadOp.load,
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
