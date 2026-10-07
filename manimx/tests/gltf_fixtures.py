"""Write tiny GLB files for importer tests, so tests need no downloads."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pygltflib
from PIL import Image

from manimx.objects.mesh import MeshData

FLOAT, UINT32 = 5126, 5125


class GlbWriter:
    """Accumulates buffer data, accessors and materials, then saves a GLB."""

    def __init__(self) -> None:
        self.gltf = pygltflib.GLTF2(scene=0, scenes=[pygltflib.Scene(nodes=[])])
        self.data = bytearray()

    def _view(self, payload: bytes) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        self.gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(self.data), byteLength=len(payload)))
        self.data += payload
        return len(self.gltf.bufferViews) - 1

    def _accessor(self, array: np.ndarray, kind: str, component: int) -> int:
        view = self._view(array.tobytes())
        accessor = pygltflib.Accessor(bufferView=view, componentType=component, count=len(array), type=kind)
        if kind == "VEC3":
            accessor.min, accessor.max = array.min(axis=0).tolist(), array.max(axis=0).tolist()
        self.gltf.accessors.append(accessor)
        return len(self.gltf.accessors) - 1

    def mesh(self, mesh: MeshData, material: int | None = None) -> int:
        attributes = pygltflib.Attributes(
            POSITION=self._accessor(mesh.positions.astype(np.float32), "VEC3", FLOAT),
            NORMAL=self._accessor(mesh.normals.astype(np.float32), "VEC3", FLOAT),
        )
        if mesh.uvs is not None:
            attributes.TEXCOORD_0 = self._accessor(mesh.uvs.astype(np.float32), "VEC2", FLOAT)
        indices = self._accessor(mesh.indices.astype(np.uint32), "SCALAR", UINT32)
        primitive = pygltflib.Primitive(attributes=attributes, indices=indices, material=material)
        self.gltf.meshes.append(pygltflib.Mesh(primitives=[primitive]))
        return len(self.gltf.meshes) - 1

    def texture(self, image: np.ndarray) -> int:
        png = io.BytesIO()
        Image.fromarray(image).save(png, format="PNG")
        view = self._view(png.getvalue())
        self.gltf.images.append(pygltflib.Image(bufferView=view, mimeType="image/png"))
        self.gltf.textures.append(pygltflib.Texture(source=len(self.gltf.images) - 1))
        return len(self.gltf.textures) - 1

    def material(self, base_color: list[float], metallic: float = 0.0, roughness: float = 1.0, texture: int | None = None) -> int:
        pbr = pygltflib.PbrMetallicRoughness(baseColorFactor=base_color, metallicFactor=metallic, roughnessFactor=roughness)
        if texture is not None:
            pbr.baseColorTexture = pygltflib.TextureInfo(index=texture)
        self.gltf.materials.append(pygltflib.Material(pbrMetallicRoughness=pbr))
        return len(self.gltf.materials) - 1

    def node(self, root: bool = True, **fields: object) -> int:
        self.gltf.nodes.append(pygltflib.Node(**fields))
        index = len(self.gltf.nodes) - 1
        if root:
            self.gltf.scenes[0].nodes.append(index)
        return index

    def save(self, path: Path) -> Path:
        self.gltf.buffers = [pygltflib.Buffer(byteLength=len(self.data))]
        self.gltf.set_binary_blob(bytes(self.data))
        self.gltf.save_binary(str(path))
        return path
