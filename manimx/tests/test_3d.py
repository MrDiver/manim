"""3D objects: meshes, materials, lights, RCurve3D, glTF import, mesh batching."""

import math
from pathlib import Path

import numpy as np
import pytest
from gltf_fixtures import GlbWriter
from manimx.animation import Group, Scene
from manimx.robjects import RCurve3D, RDirectionalLight, RShape3D, RSpotLight, Transform
from manimx.robjects.transform import rotation_towards
from manimx.userland import Rotate, load_gltf
from manimx.userland.geometry import (
    box_mesh,
    parametric_points,
    sphere_mesh,
    torus_mesh,
)
from manimx.userland.rate_functions import linear

from manimx import MeshData, PbrMaterial, Texture

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
    sun = RDirectionalLight(transform=Transform(rotation=rotation_towards((1.0, 0.0, 0.0))))
    spot = RSpotLight(transform=Transform(position=(0.0, 0.0, 5.0), rotation=rotation_towards(tuple(np.subtract((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))))))
    down = RDirectionalLight(transform=Transform(rotation=rotation_towards((0.0, 0.0, 1.0))))  # opposite of -z

    def direction(light: object) -> np.ndarray:
        return light.transform.matrix()[:3, :3] @ [0.0, 0.0, -1.0]

    np.testing.assert_allclose(direction(sun), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(direction(spot), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(direction(down), [0, 0, 1], atol=1e-12)
    with pytest.raises(ValueError, match="cone"):
        RSpotLight(inner_cone_angle=0.5, outer_cone_angle=0.4)
    with pytest.raises(ValueError, match="zero"):
        RDirectionalLight(transform=Transform(rotation=rotation_towards((0.0, 0.0, 0.0))))


# --- mesh batch -------------------------------------------------------------------


# --- RCurve3D ----------------------------------------------------------------------


def test_parametric_curve_passes_through_samples() -> None:
    helix = RCurve3D(points=parametric_points(lambda t: (math.cos(t), math.sin(t), t), (0.0, math.tau), samples=16))

    assert helix.num_curves == 16
    np.testing.assert_allclose(helix.points[0], [1, 0, 0])
    np.testing.assert_allclose(helix.points[-1], [1, 0, math.tau], atol=1e-12)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        RCurve3D(points=np.zeros((4, 2)))


# --- glTF import ------------------------------------------------------------------


class _Leaf:
    """A drawn node of a snapshot and its world matrix."""

    def __init__(self, obj, world: np.ndarray) -> None:
        self.obj, self.world = obj, world


def flatten(root, parent: np.ndarray | None = None) -> list[_Leaf]:
    """The leaves of a snapshot tree with their world matrices."""
    world = (np.eye(4) if parent is None else parent) @ root.transform.matrix()
    children = getattr(root, "children", None)
    if children is None:
        return [_Leaf(root, world)]
    return [leaf for child in children for leaf in flatten(child, world)]


def test_gltf_import_builds_hierarchy_and_converts_to_z_up(tmp_path: Path) -> None:
    writer = GlbWriter()
    material = writer.material([1.0, 0.0, 0.0, 1.0], metallic=0.5, roughness=0.25, name="Paint")
    mesh = writer.mesh(box_mesh(1.0, 2.0, 3.0), material, name="Body")
    child = writer.node(root=False, mesh=mesh, translation=[0.0, 5.0, 0.0])
    writer.node(name="parent", children=[child], rotation=[0.0, 0.0, 0.0, 1.0], scale=[2.0, 2.0, 2.0])

    model = load_gltf(writer.save(tmp_path / "box.glb"))

    assert isinstance(model, Group) and model.name == "box"
    assert model.find("parent").scale.value == (2.0, 2.0, 2.0)
    assert model.find("node-0").find("Body-0") is model.find_all(tags="material:Paint")[0]
    [item] = flatten(model.snapshot())
    shape = item.obj
    assert isinstance(shape, RShape3D) and shape.mesh.triangle_count == 12
    assert shape.material.metallic == 0.5 and shape.material.roughness == 0.25
    assert shape.material.base_color_linear == pytest.approx((1.0, 0.0, 0.0, 1.0), abs=1e-6)
    # glTF +y (scaled by 2) becomes +z.
    np.testing.assert_allclose(item.world[:3, 3], [0.0, 0.0, 10.0], atol=1e-9)


def test_gltf_import_keeps_y_up_on_request(tmp_path: Path) -> None:
    writer = GlbWriter()
    writer.node(mesh=writer.mesh(box_mesh()), translation=[0.0, 1.0, 0.0])

    model = load_gltf(writer.save(tmp_path / "box.glb"), z_up=False, name="model")
    [item] = flatten(model.snapshot())
    assert model.name == "model" and isinstance(model.find("node-0"), Group)

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

    model = load_gltf(writer.save(tmp_path / "textured.glb"), z_up=False)
    first, second = flatten(model.snapshot())

    texture_object = first.obj.material.base_color_texture
    assert texture_object is not None and texture_object is second.obj.material.base_color_texture
    np.testing.assert_array_equal(texture_object.image[0, 0], [0, 200, 0, 255])
    np.testing.assert_allclose(second.world[:3, 3], [3.0, 0.0, 0.0])  # from the node matrix
    assert first.obj.mesh is second.obj.mesh  # instanced meshes share their data
    assert first.obj.uid != second.obj.uid  # but are separate nodes


def test_gltf_parts_animate_like_any_node(tmp_path: Path) -> None:
    writer = GlbWriter()
    writer.node(name="arm", mesh=writer.mesh(box_mesh()))
    model = load_gltf(writer.save(tmp_path / "robot.glb"), z_up=False)

    class Wave(Scene):
        def construct(self) -> None:
            self.add(model)
            self.play(Rotate(model.find("arm"), math.pi / 2), model.animate.shift((1.0, 0.0, 0.0)), rate=linear)

    prepared = Wave().prepare()
    [item] = flatten(prepared.state_at(1.0))
    np.testing.assert_allclose(item.world[:3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(item.world[:3, 0], [0.0, 1.0, 0.0], atol=1e-12)  # the arm turned a quarter
