"""Render the pictures of manimx/renderer/README.md: one scene, and the
images a frame of it goes through on the way.

    uv run python scripts/render_internals.py [output folder]

The default output folder is manimx/renderer/images/. The scene is
rendered by the Python renderer (the reference; the core's renderer
draws the same), and its textures are read back after the frame: the
shadow maps, the clip masks, the G-buffer, the lit and vector layers and
the depth of field's distances and tiles. Textures the GPU cannot copy
out (multisampled ones, depth) are drawn into a plain image first by a
fullscreen pass of this script, which also maps their values to colors.
The lighting pass's edge stencil is shown by rendering once more with
the per-sample shading replaced by a flat color.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import wgpu
from manimx.animation import (
    ClippingGroup,
    DirectionalLight,
    PointLight,
    Scene,
    SpotLight,
)
from manimx.renderer import RenderConfig, RenderService
from manimx.renderer.webgpu import gpu as gpu_module
from manimx.userland import Box, Circle, Sphere, Star, Torus
from PIL import Image

OUTPUT = Path(__file__).resolve().parents[1] / "manimx" / "renderer" / "images"
WIDTH, HEIGHT = 800, 450


class Internals(Scene):
    """A floor, lit meshes, a 2D star standing up and a box seen through a
    round mask, under a sun, a spot light and a point light."""

    def construct(self) -> None:
        self.camera.move_to((7.0, -9.0, 6.0)).look_at((0.0, 0.5, 0.8))
        self.camera.set_field_of_view(0.55)
        floor = Box(12.0, 9.0, 0.1, color="#c9c4bb", roughness=0.9).move_to((0.0, 0.5, -0.05))
        ball = Sphere(1.0, color="#d1553f", roughness=0.35).move_to((-2.3, 0.8, 1.0))
        block = Box(1.4, 1.4, 1.4, color="#4b7bb5", roughness=0.6).move_to((1.6, 2.0, 0.7)).rotate(0.5, axis=(0.0, 0.0, 1.0))
        ring = Torus(0.8, 0.25, color="#5f9b6a").move_to((2.2, -1.4, 0.25))
        star = Star(5, outer_radius=1.0, color="#e8b44a", fill_opacity=1.0).rotate(math.pi / 2, axis=(1.0, 0.0, 0.0)).move_to((-0.2, -1.2, 1.2))
        window = ClippingGroup(
            Circle(0.9, fill_opacity=1.0).rotate(math.pi / 2, axis=(1.0, 0.0, 0.0)).move_to((0.0, 3.2, 1.3)),
            Box(2.4, 0.3, 2.4, color="#8a6fb3").move_to((0.0, 3.2, 1.3)),
            "window",
        )
        sun = DirectionalLight(direction=(-0.5, 0.8, -1.2), intensity=2.0, size=0.05)
        spot = SpotLight(position=(-5.0, -3.0, 6.0), target=(-1.5, 0.5, 0.0), intensity=60.0, outer_cone_angle=0.45, color="#ffd9a0", size=0.4)
        lamp = PointLight(position=(1.0, -0.2, 2.2), intensity=8.0, color="#a8c8ff", size=0.1)
        self.add(floor, ball, block, ring, star, window, sun, spot, lamp)
        self.wait(1.0)


# What a texture holds and how its values become colors: `load` reads
# `value` (vec4) at `pixel` from `source`, `color` maps it to RGB.
_SHOW = """
@group(0) @binding(0) var source: {texture};
@group(0) @binding(1) var<uniform> params: vec4<f32>;

struct Out {{ @builtin(position) position: vec4<f32> }};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Out {{
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return Out(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {{
    let size = vec2<f32>(textureDimensions(source));
    let pixel = vec2<i32>(floor(in.position.xy / params.zw * size));
    let value = {load};
    return vec4<f32>({color}, 1.0);
}}
"""

_LOAD = {
    "float": ("texture_2d<f32>", "textureLoad(source, pixel, 0)"),
    "float_msaa": ("texture_multisampled_2d<f32>", "textureLoad(source, pixel, 0)"),
    "depth_msaa": ("texture_depth_multisampled_2d", "vec4<f32>(textureLoad(source, pixel, 0))"),
    "depth_layer": ("texture_depth_2d_array", "vec4<f32>(textureLoad(source, pixel, i32(params.x), 0))"),
    "float_layer": ("texture_2d_array<f32>", "textureLoad(source, pixel, i32(params.x), 0)"),
}

_COLOR = {
    "rgb": "value.rgb",
    "alpha": "vec3<f32>(value.a)",
    "normal": "value.xyz * 0.5 + 0.5",
    "raw": "value.rgb",
    # Linear HDR: a simple tone curve, then sRGB-ish gamma.
    "hdr": "pow(value.rgb / (1.0 + value.rgb), vec3<f32>(1.0 / 2.2)) + vec3<f32>(0.07) * (1.0 - value.a)",
    # Premultiplied sRGB over a dark checkerboard.
    "vector": "value.rgb + checker(pixel) * (1.0 - value.a)",
    "radius": "vec3<f32>(value.r / params.y, value.r / params.y * 0.6, 0.2)",
}

_CHECKER = """
fn checker(pixel: vec2<i32>) -> vec3<f32> {
    let cell = (pixel.x / 16 + pixel.y / 16) % 2;
    return vec3<f32>(select(0.16, 0.22, cell == 0));
}
"""


class Reader:
    """Draws any texture into an rgba8 image with a color mapping and
    reads it back."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipelines: dict[tuple[str, str], wgpu.GPURenderPipeline] = {}
        self.params = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    def image(self, view: wgpu.GPUTextureView, load: str, color: str, size: tuple[int, int], x: float = 0.0, y: float = 1.0) -> np.ndarray:
        """``size`` RGB pixels of the texture behind ``view``, its values
        mapped by ``color`` (see _COLOR); ``x`` and ``y`` are parameters
        of the mapping (``x``: the layer of an array)."""
        return self._draw(view, load, color, size, x, y, wgpu.TextureFormat.rgba8unorm)[..., :3]

    def values(self, view: wgpu.GPUTextureView, load: str, size: tuple[int, int], layer: int = 0) -> np.ndarray:
        """The first channel of the texture as floats."""
        return self._draw(view, load, "raw", size, float(layer), 1.0, wgpu.TextureFormat.rgba32float)[..., 0]

    def _draw(self, view: wgpu.GPUTextureView, load: str, color: str, size: tuple[int, int], x: float, y: float, texture_format: str) -> np.ndarray:
        pipeline = self._pipeline(load, color, texture_format)
        target = self.device.create_texture(
            size=(*size, 1), format=texture_format, usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        )
        self.device.queue.write_buffer(self.params, 0, np.array([x, y, *size], np.float32))
        bind_group = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": view}, {"binding": 1, "resource": {"buffer": self.params}}],
        )
        encoder = self.device.create_command_encoder()
        render = encoder.begin_render_pass(
            color_attachments=[{"view": target.create_view(), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 1)}]
        )
        render.set_pipeline(pipeline)
        render.set_bind_group(0, bind_group)
        render.draw(3)
        render.end()
        self.device.queue.submit([encoder.finish()])
        dtype = np.uint8 if texture_format == wgpu.TextureFormat.rgba8unorm else np.float32
        row = size[0] * 4 * np.dtype(dtype).itemsize
        data = self.device.queue.read_texture({"texture": target}, {"bytes_per_row": row, "rows_per_image": size[1]}, (*size, 1))
        return np.frombuffer(data, dtype).reshape(size[1], size[0], 4)

    def _pipeline(self, load: str, color: str, texture_format: str) -> wgpu.GPURenderPipeline:
        key = (load, color, texture_format)
        if key not in self.pipelines:
            texture, read = _LOAD[load]
            mapping = _COLOR[color]
            code = _CHECKER + _SHOW.format(texture=texture, load=read, color=mapping)
            module = self.device.create_shader_module(code=code)
            self.pipelines[key] = self.device.create_render_pipeline(
                layout="auto",
                vertex={"module": module, "entry_point": "vs"},
                fragment={"module": module, "entry_point": "fs", "targets": [{"format": texture_format}]},
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            )
        return self.pipelines[key]


def gray(values: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """``values`` stretched to black (nearest) to white (farthest) where
    not ``empty``, which is shown dark blue."""
    shown = values[~empty]
    low, high = (shown.min(), shown.max()) if shown.size else (0.0, 1.0)
    level = np.clip((values - low) / ((high - low) or 1.0), 0.0, 1.0)
    image = np.repeat((level * 255).astype(np.uint8)[..., None], 3, axis=2)
    image[empty] = (24, 28, 48)
    return image


def save(image: np.ndarray, name: str) -> None:
    path = OUTPUT / f"{name}.png"
    Image.fromarray(np.ascontiguousarray(image)).save(path, optimize=True)
    print(f"{path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


def main() -> None:
    global OUTPUT
    if len(sys.argv) > 1:
        OUTPUT = Path(sys.argv[1]).resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared = Internals().prepare()
    (view,) = prepared.frame_at(0.5)
    config = RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, background_color="#20232b", shadow_method="pcss", camera=view.camera)

    service = RenderService()
    save(service.render(view, config)[..., :3], "frame")

    # The layers and the textures behind them, from one more render.
    renderer = service.renderer
    reader = Reader(renderer.context.device)
    layers = service.render_layers(view.root, config)
    renderer.context.device.queue.submit([])
    targets = renderer._targets
    size = (WIDTH, HEIGHT)
    save(reader.image(layers.lit.create_view(), "float", "hdr", size), "lit_layer")
    save(reader.image(layers.vector.create_view(), "float", "vector", size), "vector_layer")
    albedo, normal, emissive, scope = (texture.create_view() for texture in targets.gbuffer)
    save(reader.image(albedo, "float_msaa", "rgb", size), "gbuffer_albedo")
    save(reader.image(normal, "float_msaa", "normal", size), "gbuffer_normal")
    save(reader.image(targets.mask_layers_view, "float_layer", "alpha", size, x=0.0), "mask_layer")

    # Shadow maps: every layer, in a row (a point light has six).
    shadows = renderer._shadow_pass
    count = shadows._count
    tile = 256
    maps = []
    for layer in range(count):
        depth = reader.values(shadows.maps_view, "depth_layer", (tile, tile), layer)
        maps.append(gray(depth, depth >= 1.0))
    columns = 4
    rows = math.ceil(count / columns)
    sheet = np.full((rows * (tile + 8) - 8, columns * (tile + 8) - 8, 3), 32, np.uint8)
    for index, image in enumerate(maps):
        top, left = (index // columns) * (tile + 8), (index % columns) * (tile + 8)
        sheet[top : top + tile, left : left + tile] = image
    save(sheet, "shadow_maps")

    # Depth of field: the camera's lens opened on the ball.
    lens_camera = dataclasses.replace(view.camera, aperture=0.04, focus_distance=float(np.linalg.norm(np.subtract((7.0, -9.0, 6.0), (-2.3, 0.8, 1.0)))))
    blurred = dataclasses.replace(config, camera=lens_camera)
    save(service.render(dataclasses.replace(view, camera=lens_camera), blurred)[..., :3], "depth_of_field")
    textures = service.composer()._textures._textures
    distances = reader.values(textures["view distances"].create_view(), "float", size)
    save(gray(np.log(np.minimum(distances, 1e6)), distances > 1e8), "view_distances")
    tiles = textures["blur tiles"]
    largest = 0.05 * HEIGHT
    save(reader.image(tiles.create_view(), "float", "radius", size, y=largest), "blur_tiles")

    # The lighting pass's edges: per-sample shading replaced by magenta.
    original = gpu_module.load_wgsl

    def marked(filename: str, included: set[str] | None = None) -> str:
        code = original(filename, included)
        if filename == "deferred_lighting.wgsl":
            code = code.replace(
                "    return shade(pixel, position.xy, textureLoad(depth_samples, pixel, i32(sample)), i32(sample));",
                "    return vec4<f32>(1.0, 0.0, 1.0, 1.0);",
            )
        return code

    service.renderer.context.device.destroy()  # the adapter refuses a fourth device while two are alive
    gpu_module.load_wgsl = marked
    try:
        edges = RenderService()
        marked_frame = edges.render(view, config)[..., :3]
        # The frame, and a part of it four times enlarged.
        top, left, height, width = 300, 330, 110, 150
        part = np.repeat(np.repeat(marked_frame[top : top + height, left : left + width], 4, axis=0), 4, axis=1)
        save(marked_frame, "lighting_edges")
        save(part, "lighting_edges_detail")
    finally:
        gpu_module.load_wgsl = original


if __name__ == "__main__":
    main()
