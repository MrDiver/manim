"""Import glTF 2.0 files (.glb or .gltf) as manimx render objects.

    model = load_gltf("helmet.glb")        # a Group, ready to add to a scene

Mapping:

* every glTF node becomes a :class:`Group` with the node's transform
  (glTF stores rotations as quaternions, so they map one to one)
* every triangle primitive of a node's mesh becomes a :class:`Shape3D`
  child with a :class:`PbrMaterial`
* embedded or external PNG/JPEG images become :class:`Texture` s, shared
  between materials that use the same image

glTF is Y-up, manimx's 3D convention is Z-up (as in Blender), so by default
the whole model is wrapped in a group rotated +90 degrees about x.

Not supported yet: animations, skins, morph targets, cameras, lights,
sparse accessors, vertex colors, texture coordinate sets other than 0,
per-texture samplers (everything samples with linear filtering and
repeat), KHR extensions and alpha modes (all meshes are opaque).
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
import pygltflib
import quaternion
from PIL import Image

from ..objects import Group, MeshData, PbrMaterial, RenderObject, Shape3D, Texture, Transform
from ..utils.color import ManimColor

_COMPONENT_TYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_COMPONENT_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_TRIANGLES = 4


def load_gltf(path: str | Path, *, z_up: bool = True) -> Group:
    """Load the default scene of a glTF file as a :class:`Group`."""
    path = Path(path)
    return _Importer(pygltflib.GLTF2().load(str(path)), path.parent).scene(z_up)


class _Importer:
    def __init__(self, gltf: pygltflib.GLTF2, directory: Path) -> None:
        self.gltf = gltf
        self.directory = directory
        self.blob = gltf.binary_blob() or b""
        self._buffers: dict[int, bytes] = {}
        self._textures: dict[int, Texture] = {}
        self._materials: dict[int, PbrMaterial] = {}

    def scene(self, z_up: bool) -> Group:
        gltf = self.gltf
        if gltf.scenes:
            roots = gltf.scenes[gltf.scene or 0].nodes or []
        else:  # no scene: every node that is nobody's child is a root
            children = {c for node in gltf.nodes for c in (node.children or [])}
            roots = [i for i in range(len(gltf.nodes)) if i not in children]
        model = Group(children=[self.node(i) for i in roots])
        if z_up:
            # glTF's +y (up) becomes +z, its +z (front) becomes -y.
            return Group(children=[model], transform=Transform().rotated(math.pi / 2, (1, 0, 0)))
        return model

    # -- nodes and meshes ---------------------------------------------------

    def node(self, index: int) -> Group:
        node = self.gltf.nodes[index]
        children: list[RenderObject] = []
        if node.mesh is not None:
            children += self.mesh(node.mesh)
        children += [self.node(child) for child in node.children or []]
        return Group(children=children, transform=_node_transform(node))

    def mesh(self, index: int) -> list[Shape3D]:
        mesh = self.gltf.meshes[index]
        shapes = []
        for primitive in mesh.primitives:
            mode = primitive.mode if primitive.mode is not None else _TRIANGLES
            if mode != _TRIANGLES:
                continue  # points and lines have no surface to shade
            attributes = primitive.attributes
            positions = self.accessor(attributes.POSITION)
            indices = (
                self.accessor(primitive.indices).reshape(-1)
                if primitive.indices is not None
                else np.arange(len(positions))
            )
            data = MeshData(
                positions=positions,
                indices=indices,
                normals=self.accessor(attributes.NORMAL) if attributes.NORMAL is not None else None,
                uvs=self.accessor(attributes.TEXCOORD_0) if attributes.TEXCOORD_0 is not None else None,
                tangents=self.accessor(attributes.TANGENT) if attributes.TANGENT is not None else None,
            )
            material = self.material(primitive.material) if primitive.material is not None else PbrMaterial()
            shapes.append(Shape3D(mesh=data, material=material))
        return shapes

    # -- materials and textures ---------------------------------------------

    def material(self, index: int) -> PbrMaterial:
        if index not in self._materials:
            source = self.gltf.materials[index]
            pbr = source.pbrMetallicRoughness or pygltflib.PbrMetallicRoughness()
            base = pbr.baseColorFactor or [1.0, 1.0, 1.0, 1.0]
            emissive = source.emissiveFactor or [0.0, 0.0, 0.0]
            self._materials[index] = PbrMaterial(
                # glTF factors are linear; PbrMaterial takes sRGB like all Manim colors.
                base_color=_srgb_color(base[:3]),
                opacity=float(base[3]),
                metallic=float(pbr.metallicFactor if pbr.metallicFactor is not None else 1.0),
                roughness=float(pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0),
                emissive=_srgb_color(emissive),
                normal_scale=float(source.normalTexture.scale or 1.0) if source.normalTexture else 1.0,
                occlusion_strength=float(source.occlusionTexture.strength or 1.0) if source.occlusionTexture else 1.0,
                base_color_texture=self.texture(pbr.baseColorTexture),
                metallic_roughness_texture=self.texture(pbr.metallicRoughnessTexture),
                normal_texture=self.texture(source.normalTexture),
                occlusion_texture=self.texture(source.occlusionTexture),
                emissive_texture=self.texture(source.emissiveTexture),
            )
        return self._materials[index]

    def texture(self, info: object | None) -> Texture | None:
        if info is None:
            return None
        source = self.gltf.textures[info.index].source  # type: ignore[attr-defined]
        if source is None:
            return None
        if source not in self._textures:
            image = Image.open(io.BytesIO(self.image_bytes(source))).convert("RGBA")
            self._textures[source] = Texture(np.asarray(image))
        return self._textures[source]

    def image_bytes(self, index: int) -> bytes:
        image = self.gltf.images[index]
        if image.bufferView is not None:
            view = self.gltf.bufferViews[image.bufferView]
            start = view.byteOffset or 0
            return self.buffer(view.buffer)[start : start + view.byteLength]
        return self.uri(image.uri)

    # -- raw data -------------------------------------------------------------

    def accessor(self, index: int) -> np.ndarray:
        """Accessor contents as a float32 (or integer index) array."""
        accessor = self.gltf.accessors[index]
        if accessor.sparse is not None:
            raise NotImplementedError("sparse glTF accessors are not supported")
        dtype = np.dtype(_COMPONENT_TYPES[accessor.componentType]).newbyteorder("<")
        components = _COMPONENT_COUNTS[accessor.type]
        if accessor.bufferView is None:  # all zeros by definition
            return np.zeros((accessor.count, components), np.float32)

        view = self.gltf.bufferViews[accessor.bufferView]
        data = self.buffer(view.buffer)
        start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
        element = dtype.itemsize * components
        stride = view.byteStride or element
        raw = np.ndarray(
            shape=(accessor.count, components),
            dtype=dtype,
            buffer=data,
            offset=start,
            strides=(stride, dtype.itemsize),
        ).copy()
        if accessor.normalized:
            return (raw.astype(np.float32) / np.iinfo(dtype).max).clip(-1.0, 1.0)
        if dtype.kind == "f":
            return raw.astype(np.float32)
        return raw

    def buffer(self, index: int) -> bytes:
        if index not in self._buffers:
            uri = self.gltf.buffers[index].uri
            self._buffers[index] = self.blob if uri is None else self.uri(uri)
        return self._buffers[index]

    def uri(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        return (self.directory / uri).read_bytes()


def _node_transform(node: pygltflib.Node) -> Transform:
    if node.matrix is not None and node.matrix != list(np.eye(4).reshape(-1)):
        # glTF matrices are column-major; decompose into translation,
        # rotation and scale (glTF requires node matrices to be TRS).
        matrix = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
        scale = np.linalg.norm(matrix[:3, :3], axis=0)
        rotation = quaternion.from_rotation_matrix(matrix[:3, :3] / scale)
        return Transform(position=tuple(matrix[:3, 3]), rotation=rotation, scale=tuple(scale))
    position = node.translation or [0.0, 0.0, 0.0]
    x, y, z, w = node.rotation or [0.0, 0.0, 0.0, 1.0]  # glTF order is xyzw
    scale = node.scale or [1.0, 1.0, 1.0]
    return Transform(position=tuple(position), rotation=quaternion.quaternion(w, x, y, z), scale=tuple(scale))


def _srgb_color(linear: list[float]) -> ManimColor:
    values = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)
    return ManimColor.from_rgb(tuple(srgb))
