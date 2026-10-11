"""Render the pictures of manimx/renderer/README.md: one scene, and the
images a frame of it goes through on the way.

    uv run python scripts/render_internals.py [output folder]

The default output folder is manimx/renderer/images/. The scene is
rendered once, and the renderer's textures are read back after the frame
(``FrameRenderer.inspect``, core/src/render/inspect.rs): the shadow maps,
the clip masks, the G-buffer, the lit and vector layers and the depth of
field's distances and tiles. This script turns their values into colors.
The lighting pass's edge pixels are shown by rendering once more, in a
second process, with shaders whose per-sample shading is a flat color.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from manimx.animation import (
    ClippingGroup,
    DirectionalLight,
    PointLight,
    Scene,
    SpotLight,
)
from manimx.drivers import core_pipeline
from manimx.drivers.core_pipeline import CoreFramePipeline
from manimx.renderer import RenderConfig
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


def gray(values: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """``values`` stretched to black (nearest) to white (farthest) where
    not ``empty``, which is shown dark blue."""
    shown = values[~empty]
    low, high = (shown.min(), shown.max()) if shown.size else (0.0, 1.0)
    level = np.clip((values - low) / ((high - low) or 1.0), 0.0, 1.0)
    image = np.repeat((level * 255).astype(np.uint8)[..., None], 3, axis=2)
    image[empty] = (24, 28, 48)
    return image


def to_bytes(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def hdr(rgba: np.ndarray) -> np.ndarray:
    """Linear HDR radiance with a simple tone curve, over dark gray."""
    rgb = rgba[..., :3]
    return to_bytes((rgb / (1.0 + rgb)) ** (1 / 2.2) + 0.07 * (1.0 - rgba[..., 3:4]))


def over_checker(rgba: np.ndarray) -> np.ndarray:
    """Premultiplied colors over a dark checkerboard."""
    y, x = np.indices(rgba.shape[:2])
    checker = np.where((x // 16 + y // 16) % 2 == 0, 0.22, 0.16)[..., None]
    return to_bytes(rgba[..., :3] + checker * (1.0 - rgba[..., 3:4]))


def save(image: np.ndarray, name: str) -> None:
    path = OUTPUT / f"{name}.png"
    Image.fromarray(np.ascontiguousarray(image)).save(path, optimize=True)
    print(path)


class Frames:
    """Renders the scene's frame and reads its textures."""

    def __init__(self, config: RenderConfig) -> None:
        self.planned = Internals().prepare()
        self.config = config
        self.pipeline = CoreFramePipeline((), config, in_flight=1)

    def render(self, camera=None) -> np.ndarray:
        frame_at = self.planned.frame_at
        if camera is not None:
            frame_at = lambda t: tuple(dataclasses.replace(view, camera=camera) for view in self.planned.frame_at(t))  # noqa: E731
        return self.pipeline.render(frame_at, self.config.time)[..., :3]

    def texture(self, name: str, layer: int = 0) -> np.ndarray:
        width, height, data = self.pipeline.renderer.inspect(name, layer)
        return np.frombuffer(data, np.float32).reshape(height, width, 4)


def main() -> None:
    global OUTPUT
    if len(sys.argv) > 1 and sys.argv[1] == "--edges":
        return edges(Path(sys.argv[2]), Path(sys.argv[3]))
    if len(sys.argv) > 1:
        OUTPUT = Path(sys.argv[1]).resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, background_color="#20232b", shadow_method="pcss", time=0.5)
    frames = Frames(config)
    save(frames.render(), "frame")

    save(hdr(frames.texture("lit")), "lit_layer")
    save(over_checker(frames.texture("vector")), "vector_layer")
    save(to_bytes(frames.texture("albedo")[..., :3]), "gbuffer_albedo")
    save(to_bytes(frames.texture("normal")[..., :3] * 0.5 + 0.5), "gbuffer_normal")
    save(to_bytes(np.repeat(frames.texture("masks", 0)[..., 3:4], 3, axis=2)), "mask_layer")

    # Shadow maps: every layer, in rows of four (a point light has six).
    tile = 256
    maps = []
    for layer in range(frames.pipeline.renderer.shadow_layers()):
        depth = frames.texture("shadow_maps", layer)[..., 0]
        image = Image.fromarray(gray(depth, depth >= 1.0)).resize((tile, tile), Image.Resampling.BOX)
        maps.append(np.asarray(image))
    columns = 4
    rows = math.ceil(len(maps) / columns)
    sheet = np.full((rows * (tile + 8) - 8, columns * (tile + 8) - 8, 3), 32, np.uint8)
    for index, image in enumerate(maps):
        top, left = (index // columns) * (tile + 8), (index % columns) * (tile + 8)
        sheet[top : top + tile, left : left + tile] = image
    save(sheet, "shadow_maps")

    # Depth of field: the camera's lens opened on the ball.
    (view,) = frames.planned.frame_at(config.time)
    eye = view.camera.position
    lens = dataclasses.replace(view.camera, aperture=0.04, focus_distance=float(np.linalg.norm(np.subtract(eye, (-2.3, 0.8, 1.0)))))
    save(frames.render(camera=lens), "depth_of_field")
    distances = frames.texture("view distances")[..., 0]
    save(gray(np.log(np.minimum(distances, 1e6)), distances > 1e8), "view_distances")
    radius = frames.texture("blur tiles")[..., 0] / (0.05 * HEIGHT)
    tiles = to_bytes(np.stack([radius, radius * 0.6, np.full_like(radius, 0.2)], axis=-1))
    save(np.asarray(Image.fromarray(tiles).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)), "blur_tiles")

    # The lighting pass's edges, from shaders that paint them magenta. The
    # core loads its shaders once per process: render them in another one.
    with tempfile.TemporaryDirectory() as folder:
        shaders = Path(folder) / "shaders"
        shutil.copytree(core_pipeline.SHADER_FOLDER, shaders)
        lighting = shaders / "deferred_lighting.wgsl"
        lighting.write_text(
            lighting.read_text().replace(
                "    return shade(pixel, position.xy, textureLoad(depth_samples, pixel, i32(sample)), i32(sample));",
                "    return vec4<f32>(1.0, 0.0, 1.0, 1.0);",
            )
        )
        subprocess.run([sys.executable, __file__, "--edges", str(shaders), str(OUTPUT)], check=True)


def edges(shaders: Path, output: Path) -> None:
    """Render the frame with the shaders in ``shaders`` (see main)."""
    global OUTPUT
    OUTPUT = output
    core_pipeline.SHADER_FOLDER = shaders
    config = RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, background_color="#20232b", shadow_method="pcss", time=0.5)
    marked = Frames(config).render()
    # The frame, and a part of it four times enlarged.
    top, left, height, width = 300, 330, 110, 150
    part = np.repeat(np.repeat(marked[top : top + height, left : left + width], 4, axis=0), 4, axis=1)
    save(marked, "lighting_edges")
    save(part, "lighting_edges_detail")


if __name__ == "__main__":
    main()
