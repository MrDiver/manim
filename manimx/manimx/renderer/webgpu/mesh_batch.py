"""Pack all RShape3D meshes and lights of a frame into shared GPU arrays.

Pure numpy, like vector_batch.py. Buffers (dtypes mirror mesh.wgsl,
surface.wgsl and lighting.wgsl):

* ``objects``    per mesh: model and normal matrix, material, clip layer
* ``vertices``   all mesh vertices, interleaved (cached per mesh version)
* ``indices``    all triangle indices, relative to each mesh's first vertex

Vertices and indices are the large arrays and change only with a mesh's
version, so :class:`RetainedMeshes` keeps them in persistent arrays
mirroring the GPU buffers, with a range per mesh, and reports the ranges
it wrote (like retained.py does for vector records). Objects, materials
and draws are small and rebuilt per frame.
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

from ...robjects import (
    PbrMaterial,
    RDirectionalLight,
    RParticles,
    RPointLight,
    RShape3D,
    RSpotLight,
    SurfaceShader,
    Texture,
)
from ...robjects.material import linear_rgb
from ...robjects.mesh import MeshData
from ...robjects.transform import Transform, rotation_towards
from ..cache import RNodeDataCache
from ..config import RenderConfig
from ..scene_graph import NO_CLIP, FlatScene, LightItem
from .arena import Arena

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
    """Meshes sharing one texture set and surface shader: one bind group,
    one pipeline, one multi-draw."""

    textures: TextureSet
    surface_shader: SurfaceShader | None = None
    draws: list[tuple[int, int, int, int, int]] = field(default_factory=list)


class RetainedMeshes:
    """Vertices and indices of every mesh, kept across builds.

    A mesh keeps its ranges while its uid is drawn with the same mesh
    version; a new version is written over its ranges if the sizes fit,
    else into new ones. Meshes absent from a build free their ranges, and
    when garbage outweighs the live data everything is packed again.
    """

    def __init__(self) -> None:
        self._vertices = Arena(MESH_VERTEX_DTYPE)
        self._indices = Arena(np.dtype(np.uint32))
        self._slots: dict[int, tuple[object, int, int, int, int]] = {}
        """uid -> (mesh version, first vertex, vertex count, first index, index count)."""
        self._seen: set[int] = set()

    def place(self, shape: RShape3D, vertices: np.ndarray) -> tuple[int, int]:
        """``(first index, base vertex)`` of ``shape``'s mesh in the arrays."""
        uid, version = shape.uid, shape.version("mesh")
        self._seen.add(uid)
        slot = self._slots.get(uid)
        if slot is not None and slot[0] == version:
            return slot[3], slot[1]
        indices = shape.mesh.indices
        old = slot or (None, 0, 0, 0, 0)
        first_vertex = self._vertices.place(vertices, old[1], old[2])
        first_index = self._indices.place(indices, old[3], old[4])
        self._slots[uid] = (version, first_vertex, len(vertices), first_index, len(indices))
        return first_index, first_vertex

    def finish(self) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, int]] | None]]:
        """The arrays and their written ranges since the last call (None:
        upload all); frees the ranges of meshes not placed since then.
        """
        for uid in [uid for uid in self._slots if uid not in self._seen]:
            _, _, vertex_count, _, index_count = self._slots.pop(uid)
            self._vertices.garbage += vertex_count
            self._indices.garbage += index_count
        self._seen = set()
        dirty = {"vertices": self._vertices.ranges(), "indices": self._indices.ranges()}
        return self._vertices.view(), self._indices.view(), dirty

    def too_much_garbage(self) -> bool:
        return any(arena.garbage > max(1 << 16, arena.used // 2) for arena in (self._vertices, self._indices))


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
    dirty: dict[str, list[tuple[int, int]] | None] | None = None
    """Ranges of ``vertices`` and ``indices`` written since the previous
    batch of the same RetainedMeshes (None: upload all)."""

    @property
    def draws(self) -> np.ndarray:
        """All draw commands, group after group."""
        return _indexed_args([draw for group in self.groups for draw in group.draws])

    @staticmethod
    def build(scene: FlatScene, cache: RNodeDataCache, config: RenderConfig, geometry: RetainedMeshes | None = None) -> MeshBatch:
        """Pack the meshes and lights of ``scene``; ``geometry`` keeps
        vertices and indices from earlier builds (a new one packs all).
        """
        if geometry is None:
            geometry = RetainedMeshes()
        worlds, object_materials, clips = [], [], []  # per object; particles add one block each
        materials, shadow_draws = [], []
        material_slots: dict[int, int] = {}
        groups: dict[tuple[int, ...], DrawGroup] = {}
        object_count = 0
        material_count = 0  # rows in ``materials`` (a particle block adds one per particle)

        light_items = scene.lights or (default_lights(config) if config.default_lights else [])
        for item in scene.items:
            shape = item.obj
            if not isinstance(shape, RShape3D) or shape.mesh.triangle_count == 0:
                continue
            mesh_vertices = cache.get_or_create(
                shape.uid, "mesh-vertices", shape.version("mesh"), lambda: _interleave(shape.mesh)
            )
            material = shape.material
            # Snapshots build a new PbrMaterial every time; equal versions
            # mean equal values, so the entry (with its color conversions)
            # is cached per shape and version.
            entry = cache.get_or_create(shape.uid, "material-entry", shape.version("material"), lambda: _material_entry(material))
            if item.alpha != 1.0:
                # The item's alpha (Node.alpha, times every one above) fades the material.
                entry = entry.copy()
                entry["base_color"][3] *= item.alpha
            first_object = object_count
            if isinstance(shape, RParticles):
                # One object and one material per drawn particle, drawn as
                # instances of one draw (the shader reads objects[instance]).
                shown = np.all(np.abs(shape.scales) > 1e-9, axis=1)
                count = int(shown.sum())
                if count == 0:
                    continue
                worlds.append(item.world @ _particle_matrices(shape.positions[shown], shape.rotations[shown], shape.scales[shown]))
                object_materials.append(np.arange(material_count, material_count + count))
                materials.append(_tinted(entry, shape.colors[shown]))
                material_count += count
            else:
                count = 1
                slot_key = (id(material), item.alpha)
                if slot_key not in material_slots:
                    material_slots[slot_key] = material_count
                    materials.append(entry[None])
                    material_count += 1
                worlds.append(item.world[None])
                object_materials.append(np.array([material_slots[slot_key]]))
            clips.append(np.full(count, item.clip))
            object_count += count

            shader = material.surface_shader
            key = (*(t.uid if t else 0 for t in material.textures), shader.uid if shader else 0)
            group = groups.setdefault(key, DrawGroup(material.textures, shader))
            first_index, base_vertex = geometry.place(shape, mesh_vertices)
            draw = (len(shape.mesh.indices), count, first_index, base_vertex, first_object)
            group.draws.append(draw)
            if shape.cast_shadows:
                shadow_draws.append((draw, item.clip))

        objects = np.zeros(object_count, MESH_OBJECT_DTYPE)
        if worlds:
            world = np.concatenate(worlds)
            normal = np.zeros_like(world)
            normal[:, :3, :3] = np.linalg.inv(world[:, :3, :3]).transpose(0, 2, 1)
            normal[:, 3, 3] = 1.0
            objects["model"] = world.transpose(0, 2, 1)  # WGSL matrices are column-major
            objects["normal_matrix"] = normal.transpose(0, 2, 1)
            objects["material"] = np.concatenate(object_materials)
            objects["clip"] = np.concatenate(clips)
        vertices, indices, dirty = geometry.finish()
        return MeshBatch(
            objects=objects,
            vertices=vertices,
            indices=indices,
            materials=np.concatenate(materials) if materials else np.zeros(0, MATERIAL_DTYPE),
            lights=light_array(light_items, config),
            light_items=light_items,
            groups=list(groups.values()),
            shadow_draws={
                scope: _indexed_args([draw for draw, clip in shadow_draws if scene.in_scope(clip, scope)])
                for scope in (NO_CLIP, *range(len(scene.clips)))
            },
            dirty=dirty,
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


def _particle_matrices(positions: np.ndarray, rotations: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Local matrices (N, 4, 4) of particles: translate, rotate (unit
    quaternions w, x, y, z, normalized here since they interpolate
    linearly), scale."""
    q = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )
    matrices = np.zeros((len(positions), 4, 4))
    matrices[:, :3, :3] = rotation * scales[:, None, :]
    matrices[:, :3, 3] = positions
    matrices[:, 3, 3] = 1.0
    return matrices


def _tinted(entry: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """``entry`` once per particle: base and emissive color times the
    particle's color, emissive also times its glow."""
    entries = np.repeat(entry[None], len(colors))
    entries["base_color"][:, :3] *= colors[:, :3]
    entries["emissive"][:, :3] *= colors[:, :3] * colors[:, 3:4]
    return entries


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
        row["color"] = np.array(linear_rgb(light.color)) * (light.intensity * item.alpha)
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
    light = RDirectionalLight(transform=Transform(rotation=rotation_towards(tuple(direction))), intensity=3.0)
    return [LightItem(light, light.transform.matrix())]
