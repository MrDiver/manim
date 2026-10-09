"""Pack all RShape3D meshes and lights of a frame into shared GPU arrays.

Pure numpy, like vector_batch.py. Buffers (dtypes mirror mesh.wgsl,
surface.wgsl and lighting.wgsl):

* ``objects``    per mesh: model and normal matrix, material, clip layer
* ``vertices``   all mesh vertices, interleaved (cached per mesh version)
* ``indices``    all triangle indices, relative to each mesh's first vertex
* ``materials``  material factors, deduplicated by material object
* ``lights``     all lights in world space (or the default lights); their
                 ``shadow_view`` is filled in by shadow_views.py

Draws are grouped by the material's texture set, because textures are bound
per group (bind group 1); each group is one multi_draw_indexed_indirect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...objects import RDirectionalLight, PbrMaterial, RPointLight, RShape3D, RSpotLight, Texture
from ...objects.material import linear_rgb
from ...objects.mesh import MeshData
from ..cache import RNodeDataCache
from ..config import RenderConfig
from ..scene_graph import NO_CLIP, FlatScene, LightItem

MESH_OBJECT_DTYPE = np.dtype(
    [
        ("model", "<f4", (4, 4)),
        ("normal_matrix", "<f4", (4, 4)),
        ("material", "<u4"),
        ("clip", "<i4"),
        ("_pad", "<u4", 2),
    ]
)
MESH_VERTEX_DTYPE = np.dtype(
    [("position", "<f4", 3), ("u", "<f4"), ("normal", "<f4", 3), ("v", "<f4"), ("tangent", "<f4", 4)]
)
MATERIAL_DTYPE = np.dtype(
    [
        ("base_color", "<f4", 4),
        ("emissive", "<f4", 4),
        ("metallic", "<f4"),
        ("roughness", "<f4"),
        ("normal_scale", "<f4"),
        ("occlusion_strength", "<f4"),
    ]
)
LIGHT_DTYPE = np.dtype(
    [
        ("position", "<f4", 3),
        ("kind", "<u4"),
        ("direction", "<f4", 3),
        ("range", "<f4"),
        ("color", "<f4", 3),
        ("inner_cone_cos", "<f4"),
        ("outer_cone_cos", "<f4"),
        ("shadow_view", "<i4"),
        ("size", "<f4"),
        ("scope", "<i4"),
    ]
)
# WebGPU's DrawIndexedIndirectArgs.
DRAW_INDEXED_DTYPE = np.dtype(
    [
        ("index_count", "<u4"),
        ("instance_count", "<u4"),
        ("first_index", "<u4"),
        ("base_vertex", "<i4"),
        ("first_instance", "<u4"),
    ]
)

assert MESH_OBJECT_DTYPE.itemsize == 144
assert MESH_VERTEX_DTYPE.itemsize == 48
assert MATERIAL_DTYPE.itemsize == 48
assert LIGHT_DTYPE.itemsize == 64
assert DRAW_INDEXED_DTYPE.itemsize == 20

LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_SPOT = 0, 1, 2

TextureSet = tuple[Texture | None, ...]
"""A material's textures in shader binding order (see PbrMaterial.textures)."""


@dataclass(slots=True)
class DrawGroup:
    """Meshes sharing one texture set: one bind group, one multi-draw."""

    textures: TextureSet
    draws: list[tuple[int, int, int, int, int]] = field(default_factory=list)


@dataclass(slots=True)
class MeshBatch:
    objects: np.ndarray
    vertices: np.ndarray
    indices: np.ndarray
    materials: np.ndarray
    lights: np.ndarray
    light_items: list[LightItem]
    groups: list[DrawGroup]
    shadow_draws: dict[int, np.ndarray]
    """Per clip scope (NO_CLIP: all), draws of the meshes that cast shadows
    for that scope's lights (texture groups do not matter)."""

    @property
    def draws(self) -> np.ndarray:
        """All draw commands, group after group."""
        return _indexed_args([draw for group in self.groups for draw in group.draws])

    @staticmethod
    def build(scene: FlatScene, cache: RNodeDataCache, config: RenderConfig) -> MeshBatch:
        objects, vertices, indices, materials, shadow_draws = [], [], [], [], []
        material_slots: dict[int, int] = {}
        groups: dict[tuple[int, ...], DrawGroup] = {}
        vertex_count = index_count = 0

        light_items = scene.lights or (default_lights(config) if config.default_lights else [])
        for item in scene.items:
            shape = item.obj
            if not isinstance(shape, RShape3D) or shape.mesh.triangle_count == 0:
                continue
            mesh_vertices = cache.get_or_create(
                shape.uid, "mesh-vertices", shape.version("mesh"), lambda: _interleave(shape.mesh)
            )
            material = shape.material
            if id(material) not in material_slots:
                material_slots[id(material)] = len(materials)
                materials.append(_material_entry(material))

            entry = np.zeros((), MESH_OBJECT_DTYPE)
            entry["model"] = item.world.T  # WGSL matrices are column-major
            normal_matrix = np.eye(4)
            normal_matrix[:3, :3] = np.linalg.inv(item.world[:3, :3]).T
            entry["normal_matrix"] = normal_matrix.T
            entry["material"] = material_slots[id(material)]
            entry["clip"] = item.clip

            key = tuple(t.uid if t else 0 for t in material.textures)
            group = groups.setdefault(key, DrawGroup(material.textures))
            draw = (len(shape.mesh.indices), 1, index_count, vertex_count, len(objects))
            group.draws.append(draw)
            if shape.cast_shadows:
                shadow_draws.append((draw, item.clip))

            objects.append(entry)
            vertices.append(mesh_vertices)
            indices.append(shape.mesh.indices)
            vertex_count += len(mesh_vertices)
            index_count += len(shape.mesh.indices)

        return MeshBatch(
            objects=np.array(objects, dtype=MESH_OBJECT_DTYPE) if objects else np.zeros(0, MESH_OBJECT_DTYPE),
            vertices=np.concatenate(vertices) if vertices else np.zeros(0, MESH_VERTEX_DTYPE),
            indices=np.concatenate(indices) if indices else np.zeros(0, np.uint32),
            materials=np.array(materials, dtype=MATERIAL_DTYPE) if materials else np.zeros(0, MATERIAL_DTYPE),
            lights=light_array(light_items, config),
            light_items=light_items,
            groups=list(groups.values()),
            shadow_draws={
                scope: _indexed_args([draw for draw, clip in shadow_draws if scene.in_scope(clip, scope)])
                for scope in (NO_CLIP, *range(len(scene.clips)))
            },
        )


def _interleave(mesh: MeshData) -> np.ndarray:
    mesh = mesh.with_tangents()
    vertices = np.zeros(mesh.vertex_count, MESH_VERTEX_DTYPE)
    vertices["position"] = mesh.positions
    vertices["normal"] = mesh.normals
    if mesh.uvs is not None:
        vertices["u"], vertices["v"] = mesh.uvs[:, 0], mesh.uvs[:, 1]
    if mesh.tangents is not None:
        vertices["tangent"] = mesh.tangents
    return vertices


def _material_entry(material: PbrMaterial) -> np.ndarray:
    entry = np.zeros((), MATERIAL_DTYPE)
    entry["base_color"] = material.base_color_linear
    entry["emissive"][:3] = material.emissive_linear
    entry["metallic"] = material.metallic
    entry["roughness"] = material.roughness
    entry["normal_scale"] = material.normal_scale
    entry["occlusion_strength"] = material.occlusion_strength
    return entry


# --------------------------------------------------------------------------
# Lights
# --------------------------------------------------------------------------


def _indexed_args(rows: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    if not rows:
        return np.zeros(0, DRAW_INDEXED_DTYPE)
    return np.array(rows, dtype=np.int64).astype(np.uint32).view(DRAW_INDEXED_DTYPE).reshape(-1)


def light_array(lights: list[LightItem], config: RenderConfig) -> np.ndarray:
    """World-space light records (shadow views are assigned separately)."""
    array = np.zeros(len(lights), LIGHT_DTYPE)
    for row, item in zip(array, lights):
        light = item.light
        row["position"] = item.world[:3, 3]
        # Lights shine along their local -z axis.
        direction = -item.world[:3, 2]
        row["direction"] = direction / np.linalg.norm(direction)
        row["color"] = np.array(linear_rgb(light.color)) * light.intensity
        row["size"] = light.size
        row["shadow_view"] = -1
        row["scope"] = item.clip
        if isinstance(light, RDirectionalLight):
            row["kind"] = LIGHT_DIRECTIONAL
        elif isinstance(light, RSpotLight):
            row["kind"] = LIGHT_SPOT
            row["range"] = light.range or 0.0
            row["inner_cone_cos"] = math.cos(light.inner_cone_angle)
            row["outer_cone_cos"] = math.cos(light.outer_cone_angle)
        elif isinstance(light, RPointLight):
            row["kind"] = LIGHT_POINT
            row["range"] = light.range or 0.0
        else:  # pragma: no cover - guarded by the class hierarchy
            raise TypeError(f"unsupported light: {type(light).__name__}")
    return array


def default_lights(config: RenderConfig) -> list[LightItem]:
    """A key light shining from above-left behind the camera, so that a scene
    without lights still shows its meshes' shape."""
    camera = config.camera.transform.matrix()
    right, up, backwards = camera[:3, 0], camera[:3, 1], camera[:3, 2]
    direction = -(1.0 * up - 0.6 * right + 0.8 * backwards)
    light = RDirectionalLight.aimed(tuple(direction), intensity=3.0)
    return [LightItem(light, light.transform.matrix())]
