"""3D objects: meshes, materials, lights, Curve3D, glTF import, mesh batching."""

import math
from pathlib import Path

import numpy as np
import pytest

from gltf_fixtures import GlbWriter
from manimx import (
    Curve3D,
    DirectionalLight,
    Group,
    MeshData,
    PbrMaterial,
    PointLight,
    Shape2D,
    Shape3D,
    SpotLight,
    Stroke,
    Texture,
    Transform,
    box_mesh,
    sphere_mesh,
    torus_mesh,
)
from manimx.io import load_gltf
from manimx.renderer import MobjectDataCache, RenderConfig
from manimx.renderer.scene_graph import flatten, flatten_scene
from manimx.renderer.webgpu.mesh_batch import LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_SPOT, MeshBatch
from manimx.renderer.webgpu.vector_batch import FLAG_CAMERA_FACING, VectorBatch

# --- meshes ---------------------------------------------------------------------


@pytest.mark.parametrize("mesh", [box_mesh(), sphere_mesh(), torus_mesh()], ids=["box", "sphere", "torus"])
def test_generated_meshes_wind_counter_clockwise_outwards(mesh: MeshData) -> None:
    triangles = mesh.positions[mesh.indices.reshape(-1, 3)]
    faces = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = mesh.normals[mesh.indices.reshape(-1, 3)].mean(axis=1)
    big = np.linalg.norm(faces, axis=1) > 1e-9  # skip degenerate pole triangles
    assert np.all(np.sum(faces * normals, axis=1)[big] > 0)


def test_mesh_computes_normals_and_tangents() -> None:
    quad = MeshData(
        positions=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        indices=[0, 1, 2, 0, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
    )

    np.testing.assert_allclose(quad.normals, np.tile([0, 0, 1], (4, 1)))
    tangents = quad.with_tangents().tangents
    np.testing.assert_allclose(tangents[:, :3], np.tile([1, 0, 0], (4, 1)), atol=1e-6)
    np.testing.assert_allclose(tangents[:, 3], 1.0)
    assert MeshData(quad.positions, quad.indices).with_tangents().tangents is None  # no UVs


def test_mesh_validation() -> None:
    with pytest.raises(ValueError, match="triangles"):
        MeshData([(0, 0, 0)] * 3, [0, 1])
    with pytest.raises(ValueError, match="past the last vertex"):
        MeshData([(0, 0, 0)] * 3, [0, 1, 3])
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        MeshData([(0, 0)], [0, 0, 0])


def test_material_converts_colors_to_linear() -> None:
    material = PbrMaterial("#808080", emissive="#FFFFFF", emissive_strength=2.0)

    assert material.base_color_linear[0] == pytest.approx(0.2158, abs=1e-3)
    assert material.emissive_linear == (2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="roughness"):
        PbrMaterial(roughness=1.5)
    with pytest.raises(ValueError, match=r"\(H, W, 4\)"):
        Texture(np.zeros((2, 2, 3), np.uint8))


# --- lights ---------------------------------------------------------------------


def test_lights_point_along_their_local_minus_z() -> None:
    sun = DirectionalLight.aimed((1.0, 0.0, 0.0))
    spot = SpotLight.aimed((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    down = DirectionalLight.aimed((0.0, 0.0, 1.0))  # opposite of -z

    def direction(light: object) -> np.ndarray:
        return light.transform.matrix()[:3, :3] @ [0.0, 0.0, -1.0]

    np.testing.assert_allclose(direction(sun), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(direction(spot), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(direction(down), [0, 0, 1], atol=1e-12)
    with pytest.raises(ValueError, match="cone"):
        SpotLight(inner_cone_angle=0.5, outer_cone_angle=0.4)
    with pytest.raises(ValueError, match="zero"):
        DirectionalLight.aimed((0.0, 0.0, 0.0))


def test_scene_collects_lights_in_world_space() -> None:
    lamp = PointLight(transform=Transform(position=(1.0, 0.0, 0.0)))
    root = Group(children=[lamp], transform=Transform(position=(0.0, 2.0, 0.0)))

    scene = flatten_scene(root)

    assert scene.items == [] and [item.light for item in scene.lights] == [lamp]
    np.testing.assert_allclose(scene.lights[0].world[:3, 3], [1.0, 2.0, 0.0])


def test_light_records() -> None:
    root = Group(
        children=[
            DirectionalLight.aimed((0, 0, -1), intensity=2.0),
            PointLight(color="#FF0000", intensity=3.0, range=5.0),
            SpotLight.aimed((0, 0, 2), (0, 0, 0), inner_cone_angle=0.1, outer_cone_angle=0.2),
        ],
    )

    lights = MeshBatch.build(flatten_scene(root), MobjectDataCache(), RenderConfig()).lights

    assert list(lights["kind"]) == [LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_SPOT]
    np.testing.assert_allclose(lights["direction"][0], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(lights["color"][1], [3.0, 0.0, 0.0])
    assert lights["range"][1] == 5.0
    assert lights["outer_cone_cos"][2] == pytest.approx(math.cos(0.2))


def test_default_light_only_without_scene_lights() -> None:
    empty = flatten_scene(Group())

    assert len(MeshBatch.build(empty, MobjectDataCache(), RenderConfig()).lights) == 1
    assert len(MeshBatch.build(empty, MobjectDataCache(), RenderConfig(default_lights=False)).lights) == 0


# --- mesh batch -------------------------------------------------------------------


def test_mesh_batch_groups_draws_by_texture_set() -> None:
    texture = Texture(np.zeros((2, 2, 4), np.uint8))
    plain, textured = PbrMaterial(), PbrMaterial(base_color_texture=texture)
    shapes = [
        Shape3D(mesh=box_mesh(), material=plain),
        Shape3D(mesh=sphere_mesh(), material=textured),
        Shape3D(mesh=box_mesh(), material=plain),
    ]

    batch = MeshBatch.build(flatten_scene(Group(children=shapes)), MobjectDataCache(), RenderConfig())

    assert len(batch.materials) == 2
    assert [len(group.draws) for group in batch.groups] == [2, 1]
    draws = batch.draws
    assert list(draws["first_instance"]) == [0, 2, 1]
    assert draws["base_vertex"][1] == box_mesh().vertex_count + sphere_mesh().vertex_count
    assert len(batch.indices) == 2 * len(box_mesh().indices) + len(sphere_mesh().indices)


def test_meshes_cannot_be_masks() -> None:
    from manimx import ClippingGroup

    with pytest.raises(NotImplementedError, match="mask"):
        flatten(ClippingGroup(mask=Shape3D(), content=Shape2D()))


# --- Curve3D ----------------------------------------------------------------------


def test_curve3d_is_a_camera_facing_stroke() -> None:
    curve = Curve3D.from_corners([(0, 0, 0), (1, 0, 1), (2, 0, 0)], stroke=Stroke(width=10))

    batch = VectorBatch.build(flatten_scene(curve), MobjectDataCache(), np.zeros(3))

    assert batch.objects["flags"][0] == FLAG_CAMERA_FACING
    assert len(batch.fill_vertices) == 0  # curves have no fill
    np.testing.assert_allclose(batch.stroke_points["position"][3], [1, 0, 1])


def test_parametric_curve_passes_through_samples() -> None:
    helix = Curve3D.parametric(lambda t: (math.cos(t), math.sin(t), t), (0.0, math.tau), samples=16)

    assert helix.num_curves == 16
    np.testing.assert_allclose(helix.points[0], [1, 0, 0])
    np.testing.assert_allclose(helix.points[-1], [1, 0, math.tau], atol=1e-12)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        Curve3D(points=np.zeros((4, 2)))


# --- glTF import ------------------------------------------------------------------


def test_gltf_import_builds_hierarchy_and_converts_to_z_up(tmp_path: Path) -> None:
    writer = GlbWriter()
    material = writer.material([1.0, 0.0, 0.0, 1.0], metallic=0.5, roughness=0.25)
    mesh = writer.mesh(box_mesh(1.0, 2.0, 3.0), material)
    child = writer.node(root=False, mesh=mesh, translation=[0.0, 5.0, 0.0])
    writer.node(name="parent", children=[child], rotation=[0.0, 0.0, 0.0, 1.0], scale=[2.0, 2.0, 2.0])

    model = load_gltf(writer.save(tmp_path / "box.glb"))

    [item] = flatten(model)
    shape = item.obj
    assert isinstance(shape, Shape3D) and shape.mesh.triangle_count == 12
    assert shape.material.metallic == 0.5 and shape.material.roughness == 0.25
    assert shape.material.base_color_linear == pytest.approx((1.0, 0.0, 0.0, 1.0), abs=1e-6)
    # glTF +y (scaled by 2) becomes +z.
    np.testing.assert_allclose(item.world[:3, 3], [0.0, 0.0, 10.0], atol=1e-9)


def test_gltf_import_keeps_y_up_on_request(tmp_path: Path) -> None:
    writer = GlbWriter()
    writer.node(mesh=writer.mesh(box_mesh()), translation=[0.0, 1.0, 0.0])

    [item] = flatten(load_gltf(writer.save(tmp_path / "box.glb"), z_up=False))

    np.testing.assert_allclose(item.world[:3, 3], [0.0, 1.0, 0.0])


def test_gltf_import_decodes_shared_textures(tmp_path: Path) -> None:
    writer = GlbWriter()
    image = np.zeros((4, 4, 4), np.uint8)
    image[..., 1] = 200
    image[..., 3] = 255
    texture = writer.texture(image)
    material = writer.material([1.0, 1.0, 1.0, 1.0], texture=texture)
    mesh = writer.mesh(box_mesh(), material)
    writer.node(mesh=mesh)
    writer.node(mesh=mesh, matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1])

    first, second = flatten(load_gltf(writer.save(tmp_path / "textured.glb"), z_up=False))

    texture_object = first.obj.material.base_color_texture
    assert texture_object is not None and texture_object is second.obj.material.base_color_texture
    np.testing.assert_array_equal(texture_object.image[0, 0], [0, 200, 0, 255])
    np.testing.assert_allclose(second.world[:3, 3], [3.0, 0.0, 0.0])  # from the node matrix
