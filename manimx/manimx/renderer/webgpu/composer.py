"""The frame composer: drives the renderer and turns its layers into frames.

    composer = FrameComposer(RenderService.shared(), [MotionBlur(samples=16), Bloom(), ToneMap(ACES)])
    image = composer.compose(lambda t: scene_at(t), RenderConfig(time=2.0))

The composer is the last stage of the render chain (see packer.py) and
drives the others: temporal effects decide how often and at which times
the service's flattener, packer and renderer are asked for layers, image
effects process those layers. Turning an effect off simply means the renderer is
called less often or a pass is skipped.

Per frame::

    for each sub-frame time t (one, or the MotionBlur samples):
        for each view of scene_at(t) (one per active clip, lowest first):
            layers = service.render_layers(view's tree, config with view's camera)
            lit = lit-stage effects (Bloom, custom shaders) applied to layers.lit
            image = composite(tone_map(lit), layers.vector) over image
                    (the first view goes over the background)
        accumulator += weight * image
    frame = frame-stage effects applied to the accumulator
    output = encode to sRGB

With one view (the usual case) the image goes straight into the
accumulator; several views are stacked in a texture first, so that motion
blur averages whole images.

All intermediate images are linear and premultiplied; see composite.wgsl
for how 3D and 2D content are combined so that 2D colors stay exact.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from ...robjects import RGroup, RNode
from ..config import Camera, Frame, RenderConfig, ViewState
from ..effects import Bloom, Effect, MotionBlur, ShaderEffect, ToneMap
from .gpu import COLOR_FORMAT, HDR_FORMAT, GPUContext, ResourceCache, expand_includes
from .profiler import GpuProfiler

if TYPE_CHECKING:
    from .service import RenderService

SceneSource = Frame | Callable[[float], Frame]
"""A fixed frame, or a function returning the frame at a given time; a
:class:`ViewState`'s camera replaces the config's."""

_SAMPLED = wgpu.TextureUsage.TEXTURE_BINDING
_ATTACHMENT = wgpu.TextureUsage.RENDER_ATTACHMENT
_UNIFORM_STRIDE = 256  # minimum uniform buffer offset alignment

_ADDITIVE = {
    "color": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one, "operation": wgpu.BlendOperation.add},
    "alpha": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one, "operation": wgpu.BlendOperation.add},
}
_OVER = {  # premultiplied source over destination
    "color": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add},
    "alpha": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add},
}

COMPOSITE_PARAMS_DTYPE = np.dtype(
    [("background", "<f4", 4), ("tone_map", "<u4"), ("weight", "<f4"), ("_pad", "<f4", 2)]
)
BLOOM_PARAMS_DTYPE = np.dtype(
    [("texel", "<f4", 2), ("threshold", "<f4"), ("knee", "<f4"), ("intensity", "<f4"), ("radius", "<f4"), ("_pad", "<f4", 2)]
)


class FrameComposer:
    def __init__(self, service: RenderService, effects: Sequence[Effect] = ()) -> None:
        self.service = service
        renderer = service.renderer
        self.effects = tuple(effects)
        _validate(self.effects)
        self._context = renderer.context
        self._resources = renderer.resources
        self._passes = _Fullscreen(self._context, self._resources)
        self._textures = _TexturePool(self._context.device)
        self._bloom = _BloomProcessor(self._context, self._passes, self._textures, renderer.profiler)
        self._shaders = _ShaderProcessor(self._context, renderer.frame_uniform, self._textures, self._resources)
        self._composite_params = self._context.device.create_buffer(
            size=_UNIFORM_STRIDE * 64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._slot = 0

    def compose(self, scene: SceneSource, config: RenderConfig) -> np.ndarray:
        """The frame at ``config.time`` as ``(H, W, 4)`` uint8 straight-alpha RGBA."""
        texture = self.compose_texture(scene, config)
        data = self._context.device.queue.read_texture(
            {"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
            {"offset": 0, "bytes_per_row": 4 * config.pixel_width, "rows_per_image": config.pixel_height},
            (config.pixel_width, config.pixel_height, 1),
        )
        return np.frombuffer(data, dtype=np.uint8).reshape(config.pixel_height, config.pixel_width, 4).copy()

    def compose_texture(self, scene: SceneSource, config: RenderConfig) -> wgpu.GPUTexture:
        """The frame as an RGBA8 GPU texture (sRGB values, straight alpha)."""
        device = self._context.device
        size = (config.pixel_width, config.pixel_height)
        accumulator = self._textures.get("accumulator", size, HDR_FORMAT)

        for index, (time, weight) in enumerate(self._sub_frames(scene, config.time)):
            views = _views(scene(time) if callable(scene) else scene)
            sub_config = dataclasses.replace(config, time=time)
            if len(views) == 1:
                # One submission per sub-frame: the renderer's passes and the
                # composite share an encoder. Buffer writes for this sub-frame
                # land before it; the previous one is already submitted.
                encoder = device.create_command_encoder(label="sub-frame")
                lit, vector = self._render_view(encoder, views[0], sub_config)
                self._composite(encoder, lit, vector, accumulator, config, weight, first=index == 0)
                device.queue.submit([encoder.finish()])
                continue
            # Several views: each is rendered and submitted on its own (the
            # renderer's buffers hold one view at a time) and put over the
            # stack, which is then added to the accumulator.
            stack = self._textures.get("view stack", size, HDR_FORMAT)
            for position, view in enumerate(views):
                encoder = device.create_command_encoder(label="view")
                lit, vector = self._render_view(encoder, view, sub_config)
                self._composite(encoder, lit, vector, stack, config, 1.0, first=position == 0, background=position == 0, blend=_OVER)
                device.queue.submit([encoder.finish()])
            encoder = device.create_command_encoder(label="accumulate views")
            self._passes.run(
                encoder,
                "composite.wgsl",
                "fs_accumulate",
                accumulator,
                [self._params(config, weight, background=False), self._resources.view(stack)],
                blend=_ADDITIVE,
                clear=index == 0,
                timing=self.service.renderer.profiler.pass_timing("composite"),
            )
            device.queue.submit([encoder.finish()])

        encoder = device.create_command_encoder(label="finish frame")
        frame = accumulator
        for effect in self.effects:
            if isinstance(effect, ShaderEffect) and effect.stage == "frame":
                frame = self._shaders.apply(encoder, frame, effect, "frame")
        output = self._textures.get("output", size, COLOR_FORMAT, extra_usage=wgpu.TextureUsage.COPY_SRC)
        self._passes.run(
            encoder, "finalize.wgsl", "fs_finalize", output, [self._resources.view(frame)], timing=self.service.renderer.profiler.pass_timing("finalize")
        )
        device.queue.submit([encoder.finish()])
        return output

    def _sub_frames(self, scene: SceneSource, time: float) -> list[tuple[float, float]]:
        blur = next((e for e in self.effects if isinstance(e, MotionBlur)), None)
        if blur is None or not callable(scene) or blur.shutter == 0:
            return [(time, 1.0)]  # a fixed scene looks the same at every time
        return blur.sample_times(time)

    def _render_view(
        self, encoder: wgpu.GPUCommandEncoder, view: tuple[RNode, Camera | None], config: RenderConfig
    ) -> tuple[wgpu.GPUTexture, wgpu.GPUTexture]:
        """Render one view into the renderer's layers; returns the lit layer
        after the lit-stage effects, and the vector layer.
        """
        root, camera = view
        if camera is not None:
            config = dataclasses.replace(config, camera=camera)
        layers = self.service.render_layers(root, config, encoder)
        lit = layers.lit
        for effect in self.effects:
            if isinstance(effect, Bloom):
                lit = self._bloom.apply(encoder, lit, effect)
            elif isinstance(effect, ShaderEffect) and effect.stage == "lit":
                lit = self._shaders.apply(encoder, lit, effect, "lit")
        return lit, layers.vector

    def _params(self, config: RenderConfig, weight: float, background: bool) -> dict:
        """Composite parameters in the next uniform slot, as a binding."""
        tone_map = next((e for e in self.effects if isinstance(e, ToneMap)), ToneMap())
        params = np.zeros((), COMPOSITE_PARAMS_DTYPE)
        params["background"] = config.background_rgba if background else (0.0, 0.0, 0.0, 0.0)
        params["tone_map"] = tone_map.operator.value
        params["weight"] = weight
        self._slot = (self._slot + 1) % 64
        offset = self._slot * _UNIFORM_STRIDE
        self._context.device.queue.write_buffer(self._composite_params, offset, params)
        return {"buffer": self._composite_params, "offset": offset, "size": COMPOSITE_PARAMS_DTYPE.itemsize}

    def _composite(
        self,
        encoder: wgpu.GPUCommandEncoder,
        lit: wgpu.GPUTexture,
        vector: wgpu.GPUTexture,
        target: wgpu.GPUTexture,
        config: RenderConfig,
        weight: float,
        first: bool,
        background: bool = True,
        blend: dict = _ADDITIVE,
    ) -> None:
        self._passes.run(
            encoder,
            "composite.wgsl",
            "fs_composite",
            target,
            [self._params(config, weight, background), self._resources.view(lit), self._resources.view(vector)],
            blend=blend,
            clear=first,  # the first sub-frame (or view) starts a new image
            timing=self.service.renderer.profiler.pass_timing("composite"),
        )


def _views(frame: Frame) -> list[tuple[RNode, Camera | None]]:
    """``(tree, camera)`` per view of a frame, lowest first; a camera of
    None means ``RenderConfig.camera``. An empty frame shows the background.
    """
    if isinstance(frame, RNode):
        return [(frame, None)]
    if isinstance(frame, ViewState):
        return [(frame.root, frame.camera)]
    views = [(view.root, view.camera) for view in frame]
    return views or [(RGroup(), None)]


def _validate(effects: tuple[Effect, ...]) -> None:
    for kind in (MotionBlur, ToneMap):
        if sum(isinstance(e, kind) for e in effects) > 1:
            raise ValueError(f"at most one {kind.__name__} per composer")
    for effect in effects:
        if not isinstance(effect, Effect):
            raise TypeError(f"not an effect: {effect!r}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _TexturePool:
    """Named textures reused across frames, recreated when the size changes."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._textures: dict[str, wgpu.GPUTexture] = {}

    def get(
        self, name: str, size: tuple[int, int], format: str, mip_levels: int = 1, extra_usage: int = 0
    ) -> wgpu.GPUTexture:
        texture = self._textures.get(name)
        if texture is None or (texture.width, texture.height, texture.mip_level_count) != (*size, mip_levels):
            texture = self._device.create_texture(
                label=name,
                size=(*size, 1),
                format=format,
                mip_level_count=mip_levels,
                usage=_SAMPLED | _ATTACHMENT | extra_usage,
            )
            self._textures[name] = texture
        return texture


class _Fullscreen:
    """Runs fullscreen-triangle passes; pipelines are cached per shader,
    entry point, target format and blend mode."""

    def __init__(self, context: GPUContext, resources: ResourceCache) -> None:
        self._context = context
        self.resources = resources
        self._resources = resources
        self._pipelines: dict[tuple, tuple[wgpu.GPURenderPipeline, wgpu.GPUBindGroupLayout]] = {}
        self.linear_clamp = context.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )

    def pipeline(self, shader: str, entry_point: str, format: str, blend: dict | None = None) -> tuple[wgpu.GPURenderPipeline, wgpu.GPUBindGroupLayout]:
        """The pipeline and its bind group layout (kept: each
        ``get_bind_group_layout`` call returns a new object)."""
        key = (shader, entry_point, format, None if blend is None else str(blend))
        if key not in self._pipelines:
            module = self._context.shader(shader)
            target = {"format": format}
            if blend is not None:
                target["blend"] = blend
            pipeline = self._context.device.create_render_pipeline(
                label=f"{shader}:{entry_point}",
                layout=wgpu.AutoLayoutMode.auto,
                vertex={"module": module, "entry_point": "vs_fullscreen", "buffers": []},
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
                fragment={"module": module, "entry_point": entry_point, "targets": [target]},
            )
            self._pipelines[key] = (pipeline, pipeline.get_bind_group_layout(0))
        return self._pipelines[key]

    def run(
        self,
        encoder: wgpu.GPUCommandEncoder,
        shader: str,
        entry_point: str,
        target: wgpu.GPUTexture | wgpu.GPUTextureView,
        resources: list,
        blend: dict | None = None,
        clear: bool = True,
        timing: dict | None = None,
    ) -> None:
        """Draw one fullscreen pass; ``resources`` are bindings 0, 1, 2, ..."""
        view = self._resources.view(target)
        pipeline, layout = self.pipeline(shader, entry_point, view.texture.format, blend)
        bind_group = self._resources.bind_group(layout, resources)
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "clear_value": (0, 0, 0, 0),
                    "load_op": wgpu.LoadOp.clear if clear else wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            timestamp_writes=timing,
        )
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(3)
        render_pass.end()


class _BloomProcessor:
    """GPU side of :class:`Bloom`, see shaders/bloom.wgsl."""

    def __init__(self, context: GPUContext, passes: _Fullscreen, textures: _TexturePool, profiler: GpuProfiler) -> None:
        self._context = context
        self._profiler = profiler
        self._passes = passes
        self._textures = textures
        self._params = context.device.create_buffer(
            size=_UNIFORM_STRIDE * 64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def apply(self, encoder: wgpu.GPUCommandEncoder, lit: wgpu.GPUTexture, bloom: Bloom) -> wgpu.GPUTexture:
        width, height = lit.width, lit.height
        # The chain starts at half resolution; stop before mips get tiny.
        levels = max(1, min(bloom.levels, int(np.log2(max(1, min(width, height) // 4)))))
        chain = self._textures.get("bloom chain", (max(1, width // 2), max(1, height // 2)), HDR_FORMAT, mip_levels=levels)
        output = self._textures.get("bloom output", (width, height), HDR_FORMAT)
        cache = self._passes.resources
        mips = [cache.view(chain, base_mip_level=i, mip_level_count=1) for i in range(levels)]
        slot = iter(range(64))

        def step(entry: str, source: wgpu.GPUTexture | wgpu.GPUTextureView, source_size: tuple[int, int], target, **kwargs) -> None:
            params = np.zeros((), BLOOM_PARAMS_DTYPE)
            params["texel"] = (1 / source_size[0], 1 / source_size[1])
            params["threshold"], params["knee"] = bloom.threshold, bloom.knee
            params["intensity"], params["radius"] = bloom.intensity, bloom.radius
            offset = next(slot) * _UNIFORM_STRIDE
            self._context.device.queue.write_buffer(self._params, offset, params)
            source_view = cache.view(source)
            bindings = [
                {"buffer": self._params, "offset": offset, "size": BLOOM_PARAMS_DTYPE.itemsize},
                source_view,
                self._passes.linear_clamp,
            ]
            if entry == "fs_combine":
                bindings.append(mips[0])
            self._passes.run(encoder, "bloom.wgsl", entry, target, bindings, timing=self._profiler.pass_timing("bloom"), **kwargs)

        def mip_size(level: int) -> tuple[int, int]:
            return max(1, chain.width >> level), max(1, chain.height >> level)

        step("fs_prefilter", lit, (width, height), mips[0])
        for level in range(levels - 1):
            step("fs_downsample", mips[level], mip_size(level), mips[level + 1])
        for level in range(levels - 1, 0, -1):
            # Add the upsampled smaller mip onto the next larger one.
            step("fs_upsample", mips[level], mip_size(level), mips[level - 1], blend=_ADDITIVE, clear=False)
        step("fs_combine", lit, (width, height), output)
        return output


_SHADER_EFFECT_TEMPLATE = """
#include "fullscreen.wgsl"
#include "frame.wgsl"
#include "color.wgsl"

struct EffectValues {{
    values: array<vec4<f32>, 4>,
}};

@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;
@group(0) @binding(3) var<uniform> effect_values: EffectValues;

fn value(index: u32) -> f32 {{
    return effect_values.values[index / 4u][index % 4u];
}}

{code}

@fragment
fn fs_effect(in: FullscreenVertex) -> @location(0) vec4<f32> {{
    return effect(in.uv, vec2<i32>(floor(in.position.xy)));
}}
"""


class _ShaderProcessor:
    """GPU side of :class:`ShaderEffect`: wraps user code in a template."""

    def __init__(self, context: GPUContext, frame_uniform: wgpu.GPUBuffer, textures: _TexturePool, resources: ResourceCache) -> None:
        self._context = context
        self._resources = resources
        self._frame_uniform = frame_uniform
        self._textures = textures
        self._sampler = context.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear
        )
        self._values = context.device.create_buffer(
            size=_UNIFORM_STRIDE * 64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        fragment = wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX
        self._layout = context.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": fragment, "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": fragment, "texture": {"sample_type": wgpu.TextureSampleType.float}},
                {"binding": 2, "visibility": fragment, "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": 3, "visibility": fragment, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )
        self._pipelines: dict[str, wgpu.GPURenderPipeline] = {}
        self._uses = 0

    def apply(self, encoder: wgpu.GPUCommandEncoder, source: wgpu.GPUTexture, effect: ShaderEffect, stage: str) -> wgpu.GPUTexture:
        device = self._context.device
        # Alternate between two outputs per stage so effects can be chained.
        self._uses += 1
        output = self._textures.get(f"{stage} effect {self._uses % 2}", (source.width, source.height), HDR_FORMAT)
        values = np.zeros(16, np.float32)
        values[: len(effect.values)] = effect.values
        offset = (self._uses % 64) * _UNIFORM_STRIDE
        device.queue.write_buffer(self._values, offset, values)
        bind_group = self._resources.bind_group(
            self._layout,
            [
                {"buffer": self._frame_uniform, "offset": 0, "size": self._frame_uniform.size},
                self._resources.view(source),
                self._sampler,
                {"buffer": self._values, "offset": offset, "size": 64},
            ],
        )
        render_pass = encoder.begin_render_pass(
            color_attachments=[{"view": self._resources.view(output), "clear_value": (0, 0, 0, 0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}]
        )
        render_pass.set_pipeline(self._pipeline(effect.code))
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(3)
        render_pass.end()
        return output

    def _pipeline(self, code: str) -> wgpu.GPURenderPipeline:
        if code not in self._pipelines:
            device = self._context.device
            module = device.create_shader_module(label="shader effect", code=expand_includes(_SHADER_EFFECT_TEMPLATE.format(code=code)))
            self._pipelines[code] = device.create_render_pipeline(
                label="shader effect",
                layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
                vertex={"module": module, "entry_point": "vs_fullscreen", "buffers": []},
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
                fragment={"module": module, "entry_point": "fs_effect", "targets": [{"format": HDR_FORMAT}]},
            )
        return self._pipelines[code]
