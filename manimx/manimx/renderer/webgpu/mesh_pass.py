"""GPU side of the mesh pipeline: buffers, textures, pipelines, draws.

:meth:`MeshPass.prepare` uploads a :class:`MeshBatch`; :meth:`MeshPass.draw`
issues one ``multi_draw_indexed_indirect`` per material texture set. Meshes
are opaque and drawn into the frame pass before the vector shapes.

Textures are uploaded once per (texture, color space) and kept while any
frame uses them; mip levels are generated on the CPU when uploading.
"""

from __future__ import annotations

import numpy as np
import wgpu
from PIL import Image
from wgpu.backends.wgpu_native.extras import multi_draw_indexed_indirect

from ...objects import Texture
from .gpu import COLOR_FORMAT, DEPTH_FORMAT, GPUContext, GrowableBuffer
from .mesh_batch import DRAW_INDEXED_DTYPE, MeshBatch, TextureSet

_STORAGE = wgpu.BufferUsage.STORAGE

# Color space of each material texture slot, in PbrMaterial.textures order:
# base color and emissive hold sRGB colors, the others hold linear data.
_SLOT_IS_SRGB = (True, False, False, False, True)
# Neutral values for missing textures: white, except a flat normal map.
_SLOT_DEFAULT = ((255, 255, 255, 255),) * 2 + ((128, 128, 255, 255),) + ((255, 255, 255, 255),) * 2


class MeshPass:
    def __init__(self, context: GPUContext, frame_uniform: wgpu.GPUBuffer) -> None:
        self._context = context
        self._frame_uniform = frame_uniform
        self._buffers = {
            "objects": GrowableBuffer(context, "mesh objects", _STORAGE),
            "vertices": GrowableBuffer(context, "mesh vertices", _STORAGE),
            "materials": GrowableBuffer(context, "materials", _STORAGE),
            "lights": GrowableBuffer(context, "lights", _STORAGE),
        }
        self._indices = GrowableBuffer(context, "mesh indices", wgpu.BufferUsage.INDEX)
        self._indirect = GrowableBuffer(context, "mesh draws", wgpu.BufferUsage.INDIRECT)
        self._shader = context.shader("mesh.wgsl")
        self._scene_layout, self._material_layout = self._create_layouts()
        self._sampler = context.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
            address_mode_u=wgpu.AddressMode.repeat,
            address_mode_v=wgpu.AddressMode.repeat,
            max_anisotropy=8,
        )
        self._defaults = [
            self._upload_texture(np.array(value, np.uint8).reshape(1, 1, 4), srgb)
            for value, srgb in zip(_SLOT_DEFAULT, _SLOT_IS_SRGB)
        ]
        self._textures: dict[tuple[int, bool], wgpu.GPUTexture] = {}
        self._material_groups: dict[tuple[int, ...], wgpu.GPUBindGroup] = {}
        self._pipelines: dict[int, wgpu.GPURenderPipeline] = {}
        self._scene_group: wgpu.GPUBindGroup | None = None
        self._mask_layers: wgpu.GPUTextureView | None = None
        self._draw_groups: list[tuple[wgpu.GPUBindGroup, int, int]] = []

    # -- per frame ----------------------------------------------------------

    def prepare(self, batch: MeshBatch, mask_layers: wgpu.GPUTextureView) -> int:
        """Upload the frame's meshes and lights; returns the light count."""
        grown = False
        for name, buffer in self._buffers.items():
            grown |= buffer.upload(getattr(batch, name))
        self._indices.upload(batch.indices)
        self._indirect.upload(batch.draws)
        if grown or self._scene_group is None or mask_layers is not self._mask_layers:
            self._mask_layers = mask_layers
            self._scene_group = self._create_scene_group()

        used_textures: set[tuple[int, bool]] = set()
        used_groups: set[tuple[int, ...]] = set()
        self._draw_groups = []
        first = 0
        for group in batch.groups:
            key = tuple(t.uid if t else 0 for t in group.textures)
            used_groups.add(key)
            used_textures.update(
                (t.uid, srgb) for t, srgb in zip(group.textures, _SLOT_IS_SRGB) if t is not None
            )
            if key not in self._material_groups:
                self._material_groups[key] = self._create_material_group(group.textures)
            self._draw_groups.append((self._material_groups[key], first, len(group.draws)))
            first += len(group.draws)

        # Forget textures and bind groups no longer used by any mesh.
        self._textures = {k: v for k, v in self._textures.items() if k in used_textures}
        self._material_groups = {k: v for k, v in self._material_groups.items() if k in used_groups}
        return len(batch.lights)

    def draw(self, render_pass: wgpu.GPURenderPassEncoder, sample_count: int) -> None:
        if not self._draw_groups:
            return
        render_pass.set_pipeline(self._pipeline(sample_count))
        render_pass.set_bind_group(0, self._scene_group)
        render_pass.set_index_buffer(self._indices.buffer, wgpu.IndexFormat.uint32)
        for material_group, first, count in self._draw_groups:
            render_pass.set_bind_group(1, material_group)
            multi_draw_indexed_indirect(
                render_pass,
                self._indirect.buffer,
                offset=first * DRAW_INDEXED_DTYPE.itemsize,
                count=count,
            )

    # -- textures -----------------------------------------------------------

    def _create_material_group(self, textures: TextureSet) -> wgpu.GPUBindGroup:
        entries = []
        for slot, (texture, srgb) in enumerate(zip(textures, _SLOT_IS_SRGB)):
            gpu_texture = self._defaults[slot] if texture is None else self._texture(texture, srgb)
            entries.append({"binding": slot, "resource": gpu_texture.create_view()})
        entries.append({"binding": len(_SLOT_IS_SRGB), "resource": self._sampler})
        return self._context.device.create_bind_group(layout=self._material_layout, entries=entries)

    def _texture(self, texture: Texture, srgb: bool) -> wgpu.GPUTexture:
        key = (texture.uid, srgb)
        if key not in self._textures:
            self._textures[key] = self._upload_texture(texture.image, srgb)
        return self._textures[key]

    def _upload_texture(self, image: np.ndarray, srgb: bool) -> wgpu.GPUTexture:
        levels = _mip_chain(image)
        height, width = image.shape[:2]
        texture = self._context.device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm_srgb if srgb else wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            mip_level_count=len(levels),
        )
        for level, data in enumerate(levels):
            h, w = data.shape[:2]
            self._context.device.queue.write_texture(
                {"texture": texture, "mip_level": level, "origin": (0, 0, 0)},
                np.ascontiguousarray(data),
                {"offset": 0, "bytes_per_row": 4 * w, "rows_per_image": h},
                (w, h, 1),
            )
        return texture

    # -- setup --------------------------------------------------------------

    def _pipeline(self, sample_count: int) -> wgpu.GPURenderPipeline:
        if sample_count not in self._pipelines:
            device = self._context.device
            self._pipelines[sample_count] = device.create_render_pipeline(
                label=f"mesh x{sample_count}",
                layout=device.create_pipeline_layout(
                    bind_group_layouts=[self._scene_layout, self._material_layout]
                ),
                vertex={"module": self._shader, "entry_point": "vs_main", "buffers": []},
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    # Double sided: the fragment shader flips back-face normals.
                    "cull_mode": wgpu.CullMode.none,
                },
                depth_stencil={
                    "format": DEPTH_FORMAT,
                    "depth_write_enabled": True,
                    "depth_compare": wgpu.CompareFunction.less,
                },
                multisample={"count": sample_count},
                fragment={"module": self._shader, "entry_point": "fs_main", "targets": [{"format": COLOR_FORMAT}]},
            )
        return self._pipelines[sample_count]

    def _create_layouts(self) -> tuple[wgpu.GPUBindGroupLayout, wgpu.GPUBindGroupLayout]:
        device = self._context.device
        vertex, fragment = wgpu.ShaderStage.VERTEX, wgpu.ShaderStage.FRAGMENT
        storage = {"type": wgpu.BufferBindingType.read_only_storage}
        scene = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": vertex | fragment, "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": vertex | fragment, "buffer": storage},  # objects
                {"binding": 2, "visibility": vertex, "buffer": storage},  # vertices
                {"binding": 3, "visibility": fragment, "buffer": storage},  # materials
                {"binding": 4, "visibility": fragment, "buffer": storage},  # lights
                {
                    "binding": 5,
                    "visibility": fragment,
                    "texture": {"sample_type": wgpu.TextureSampleType.float, "view_dimension": wgpu.TextureViewDimension.d2_array},
                },
            ]
        )
        textures = [
            {"binding": slot, "visibility": fragment, "texture": {"sample_type": wgpu.TextureSampleType.float}}
            for slot in range(len(_SLOT_IS_SRGB))
        ]
        sampler = {"binding": len(_SLOT_IS_SRGB), "visibility": fragment, "sampler": {"type": wgpu.SamplerBindingType.filtering}}
        material = device.create_bind_group_layout(entries=[*textures, sampler])
        return scene, material

    def _create_scene_group(self) -> wgpu.GPUBindGroup:
        buffers = [self._frame_uniform, *(b.buffer for b in self._buffers.values())]
        entries = [
            {"binding": i, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}}
            for i, buffer in enumerate(buffers)
        ]
        entries.append({"binding": len(buffers), "resource": self._mask_layers})
        return self._context.device.create_bind_group(layout=self._scene_layout, entries=entries)


def _mip_chain(image: np.ndarray) -> list[np.ndarray]:
    """The image and successive half-size versions down to 1x1."""
    levels = [image]
    current = Image.fromarray(image)
    while current.width > 1 or current.height > 1:
        current = current.resize((max(1, current.width // 2), max(1, current.height // 2)), Image.Resampling.BOX)
        levels.append(np.asarray(current))
    return levels
