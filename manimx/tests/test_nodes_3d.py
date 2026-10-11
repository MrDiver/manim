"""3D scene nodes: Shape3D, Curve3D and the lights, their snapshots, version
tokens, and the animations that work on them like on Shape2D."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import (
    Curve3D,
    DirectionalLight,
    Group,
    Light,
    PointLight,
    Scene,
    Shape3D,
    Simulation,
    SpotLight,
    derive,
)
from manimx.drivers import render_image
from manimx.renderer import RenderConfig
from manimx.robjects import (
    MeshData,
    PbrMaterial,
    RCurve3D,
    RDirectionalLight,
    RPointLight,
    RShape2D,
    RShape3D,
    RSpotLight,
    Texture,
)
from manimx.userland import (
    OUT,
    RIGHT,
    UP,
    Box,
    Circle,
    FadeIn,
    Indicate,
    ParametricCurve3D,
    ParametricSurface,
    Rotate,
    Sphere,
    Torus,
)
from manimx.userland.geometry import box_mesh, sphere_mesh, surface_mesh
from manimx.userland.rate_functions import linear


def prepare(construct):
    class Built(Scene):
        def construct(self) -> None:
            construct(self)

    return Built().prepare()


# --- Shape3D ----------------------------------------------------------------------


def test_shape3d_snapshots_to_rshape3d_with_its_material() -> None:
    texture = Texture(np.zeros((2, 2, 4), np.uint8))
    template = PbrMaterial("#FF0000", roughness=0.25, normal_scale=0.5, base_color_texture=texture)
    shape = Shape3D(sphere_mesh(), "ball", material=template, metallic=0.75, emissive="#00FF00", cast_shadows=False, tags="ball")

    snapshot = shape.snapshot()
    assert isinstance(snapshot, RShape3D) and snapshot.uid == shape.uid
    assert snapshot.mesh == shape.mesh.value and not snapshot.cast_shadows
    material = snapshot.material
    assert material.base_color_linear == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert (material.metallic, material.roughness, material.normal_scale) == (0.75, 0.25, 0.5)
    assert material.emissive_linear == pytest.approx((0.0, 1.0, 0.0))
    assert material.base_color_texture == texture  # constants come from the template


def test_moving_keeps_mesh_and_material() -> None:
    shape = Box()
    first = shape.snapshot()
    shape.shift(RIGHT).rotate(1.0).scale_by(2.0)
    moved = shape.snapshot()
    assert moved.version("mesh") == first.version("mesh") and moved.material.roughness == first.material.roughness
    assert moved.transform.position == (1.0, 0.0, 0.0)

    shape.set_material(roughness=0.9)
    recolored = shape.snapshot()
    assert recolored.version("mesh") == first.version("mesh") and recolored.material.roughness == 0.9

    shape.set_mesh(sphere_mesh())
    assert shape.snapshot().version("mesh") != first.version("mesh")
    shape.set_mesh(shape.mesh.value)  # the same MeshData is no change
    with pytest.raises(TypeError, match="a mesh must be a MeshData"):
        shape.set_mesh("sphere")


def test_alpha_multiplies_opacity_and_factors_are_clamped() -> None:
    shape = Shape3D(opacity=0.8)
    shape.set_alpha(0.5).set_material(metallic=1.4, roughness=-0.2).set_emissive(strength=-1.0)
    group = Group(shape)
    group.set_alpha(0.5)

    root = group.snapshot()
    material = root.children[0].material
    assert material.opacity == pytest.approx(0.8)  # alphas travel on their own...
    assert root.alpha * root.children[0].alpha == pytest.approx(0.25)  # ...the core multiplies them
    assert (material.metallic, material.roughness, material.emissive_strength) == (1.0, 0.0, 0.0)
    shape.set_emissive("#0000FF")
    assert shape.emissive.value == (0.0, 0.0, 1.0)


def test_solids_and_surfaces() -> None:
    assert Box(1, 2, 3).mesh.value.triangle_count == 12
    assert Sphere(2.0, segments=8, rings=4).mesh.value.vertex_count == 9 * 5
    assert Torus(segments=8, sides=4).mesh.value.vertex_count == 9 * 5
    plane = ParametricSurface(lambda u, v: (u, v, 0.0), resolution=(4, 2), color="#FFFFFF")
    wave = ParametricSurface(lambda u, v: (u, v, 0.2 * math.sin(u)), resolution=(4, 2))
    assert plane.mesh.value.vertex_count == wave.mesh.value.vertex_count == 15
    np.testing.assert_array_equal(plane.mesh.value.indices, wave.mesh.value.indices)  # vertices correspond
    np.testing.assert_allclose(plane.mesh.value.normals, np.tile([0.0, 0.0, 1.0], (15, 1)), atol=1e-6)
    with pytest.raises(ValueError, match="at least one cell"):
        surface_mesh(lambda u, v: (u, v, 0.0), resolution=(0, 2))
    with pytest.raises(ValueError, match="must return a 3D point"):
        surface_mesh(lambda u, v: (u, v))


# --- Curve3D ----------------------------------------------------------------------


def test_curve3d_snapshots_with_stroke() -> None:
    helix = ParametricCurve3D(lambda t: (math.cos(t), math.sin(t), 0.1 * t), (0.0, math.tau), samples=8, color="#FF0000", stroke_width=0.1)
    snapshot = helix.snapshot()
    assert isinstance(snapshot, RCurve3D) and snapshot.points.shape == (32, 3) and snapshot.cast_shadows
    assert snapshot.stroke.width[0] == pytest.approx(0.1)

    helix.shift(UP)
    moved = helix.snapshot()
    np.testing.assert_array_equal(moved.points, snapshot.points)  # points are local: moving changes the transform
    assert moved.transform.position != snapshot.transform.position
    helix.set_stroke(color="#00FF00", width=0.2, opacity=0.5).set_color("#0000FF").set_cast_shadows(False)
    changed = helix.snapshot()
    assert changed.stroke != snapshot.stroke and not changed.cast_shadows
    assert changed.stroke.opacity[0] == pytest.approx(0.5)
    assert Curve3D().snapshot().points.shape == (0, 3)


def test_shape2d_passes_cast_shadows() -> None:
    circle = Circle()
    assert isinstance(circle.snapshot(), RShape2D) and circle.snapshot().cast_shadows
    circle.set_cast_shadows(False)
    assert not circle.snapshot().cast_shadows


# --- lights -----------------------------------------------------------------------


def test_lights_snapshot_to_their_render_classes_with_their_parts() -> None:
    sun = DirectionalLight("sun", direction=(0, 0, -1), color="#FFFF00", size=0.05)
    lamp = PointLight("lamp", position=(1, 2, 3), range=5.0)
    spot = SpotLight("spot", position=(0, 0, 4), target=(0, 0, 0), inner_cone_angle=0.1, outer_cone_angle=0.3)

    sun_state, lamp_state, spot_state = sun.snapshot(), lamp.snapshot(), spot.snapshot()
    assert isinstance(sun_state, RDirectionalLight) and sun_state.intensity == 3.0 and sun_state.size == 0.05
    assert isinstance(lamp_state, RPointLight) and lamp_state.range == 5.0
    assert lamp_state.transform.position == (1.0, 2.0, 3.0)
    assert isinstance(spot_state, RSpotLight) and spot_state.range is None
    assert (spot_state.inner_cone_angle, spot_state.outer_cone_angle) == (0.1, 0.3)
    # Lights shine along local -z: the spot points down at its target.
    np.testing.assert_allclose(-spot_state.transform.matrix()[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(-sun_state.transform.matrix()[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
    assert Light().snapshot() is None  # a light of no kind draws nothing


def test_lights_change_direction_and_keep_valid_cones() -> None:
    sun = DirectionalLight()
    sun.point_along((1, 0, 0))
    np.testing.assert_allclose(-sun.snapshot().transform.matrix()[:3, 2], [1.0, 0.0, 0.0], atol=1e-12)

    spot = SpotLight(position=(0, 0, 0))
    spot.point_at((0, 3, 0))
    np.testing.assert_allclose(-spot.snapshot().transform.matrix()[:3, 2], [0.0, 1.0, 0.0], atol=1e-12)
    spot.point_along((0, -1, 0))
    np.testing.assert_allclose(-spot.snapshot().transform.matrix()[:3, 2], [0.0, -1.0, 0.0], atol=1e-12)
    spot.set_cone(inner=2.0, outer=3.0).set_range(4.0)  # out of range while animating: clamped
    state = spot.snapshot()
    assert state.outer_cone_angle == pytest.approx(math.pi / 2) and state.inner_cone_angle < state.outer_cone_angle
    assert state.range == 4.0
    lamp = PointLight()
    lamp.set_range(2.0).set_intensity(5.0).set_color("#FF0000").set_size(0.5).set_cast_shadows(False)
    state = lamp.snapshot()
    assert (state.range, state.intensity, state.size, state.cast_shadows) == (2.0, 5.0, 0.5, False)


def test_lights_have_only_their_parts() -> None:
    sun, lamp = DirectionalLight("sun"), PointLight("lamp")
    with pytest.raises(TypeError, match="FadeIn\\(shift=...\\) needs a position, but DirectionalLight 'sun' has none"):
        FadeIn(sun, shift=UP)
    with pytest.raises(AttributeError, match="DirectionalLight has no position, so it has no 'shift'"):
        sun.shift(UP)
    with pytest.raises(TypeError, match="Rotate needs a rotation, but PointLight 'lamp' has none"):
        Rotate(lamp, 1.0)
    with pytest.raises(TypeError, match="Indicate needs a scale"):
        Indicate(lamp)
    assert lamp.snapshot().transform.rotation == sun.snapshot().transform.rotation.__class__(1, 0, 0, 0)


# --- animations on 3D nodes ---------------------------------------------------------


def test_fade_animate_rotate_derive_and_follow_on_3d_nodes() -> None:
    ball = Sphere(name="ball", roughness=0.2)
    lamp = PointLight("lamp", intensity=10.0)
    sun = DirectionalLight("sun")

    def construct(scene):
        lamp.follow(ball, OUT * 2)
        sun.intensity.set(1.0)
        sun.intensity = derive(lambda r: 4.0 * r, ball.roughness)  # a reaction to another node
        scene.play(FadeIn(ball), FadeIn(lamp), FadeIn(sun), rate=linear)
        scene.play(ball.animate.shift(RIGHT * 2).set_material(roughness=0.7), Rotate(sun, math.pi / 2, axis=(1, 0, 0)), rate=linear)

    prepared = prepare(construct)
    half = {child.uid: child for child in prepared.state_at(0.5).children}
    assert half[ball.uid].alpha == pytest.approx(0.5)
    assert half[lamp.uid].alpha == pytest.approx(0.5) and half[lamp.uid].intensity == pytest.approx(10.0)
    end = {child.uid: child for child in prepared.state_at(2.0).children}
    assert end[ball.uid].transform.position == (2.0, 0.0, 0.0) and end[ball.uid].material.roughness == pytest.approx(0.7)
    assert end[lamp.uid].transform.position == (2.0, 0.0, 2.0)  # followed the ball
    assert end[sun.uid].intensity == pytest.approx(2.8)  # derived from the roughness
    mid = {child.uid: child for child in prepared.state_at(1.5).children}
    assert mid[ball.uid].version("mesh") == end[ball.uid].version("mesh")  # the mesh never changed


def test_simulation_moves_a_shape3d() -> None:
    ball = Sphere(name="ball")
    velocity = np.array([0.0, 0.0, 0.0])

    def step(dt: float) -> None:
        velocity[2] -= 9.81 * dt
        ball.shift(velocity * dt)

    def construct(scene):
        scene.add(ball)
        scene.play(Simulation(step, dt=1 / 240, duration=1.0))

    prepared = prepare(construct)
    z = prepared.state_at(1.0).children[0].transform.position[2]
    assert z == pytest.approx(-9.81 / 2, abs=0.05)


# --- rendering ----------------------------------------------------------------------


def test_a_faded_light_darkens_the_mesh() -> None:
    ball = Sphere(name="ball", color="#FFFFFF", roughness=1.0)
    sun = DirectionalLight("sun", direction=(0, 0, -1))

    def construct(scene):
        scene.add(ball, sun)
        scene.play(sun.animate.set_alpha(0.0), rate=linear)

    prepared = prepare(construct)
    config = RenderConfig(pixel_width=64, pixel_height=36, ambient_intensity=0.0)
    lit, dark = (render_image(prepared, t, config) for t in (0.0, 1.0))
    assert lit[18, 32, :3].mean() > 100 and dark[18, 32, :3].max() < lit[18, 32, :3].max() / 4


def test_the_mesh_keeps_its_version_while_shapes_move_and_recolor() -> None:
    """Renderers build a mesh's records once per version of it."""
    box = Box(name="box")

    def construct(scene):
        scene.add(box)
        scene.play(box.animate.shift(RIGHT).rotate(1.0).set_color("#FF0000"), run_time=1.0)

    prepared = prepare(construct)
    states = [prepared.state_at(t).children[0] for t in np.linspace(0.0, 1.0, 5)]
    assert len({state.version("mesh") for state in states}) == 1
    assert len({state.version("material") for state in states}) == 5
    assert prepared.state_at(1.0).children[0].version("material") == prepared.state_at(1.0).children[0].version("material")


def test_mesh_data_is_shared_not_copied() -> None:
    mesh = box_mesh()
    assert Shape3D(mesh).snapshot().mesh == mesh and isinstance(mesh, MeshData)


# --- Particles ----------------------------------------------------------------------


def test_particles_snapshot_to_one_rparticles_and_animate_their_arrays() -> None:
    from manimx.animation import Particles
    from manimx.robjects import RParticles

    sparks = Particles(2, sphere_mesh(1.0, 8, 6), "sparks")

    def construct(scene):
        scene.add(sparks)
        scene.play(sparks.animate.set_positions([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]), rate=linear)

    prepared = prepare(construct)
    (snapshot,) = prepared.state_at(0.5).children

    assert isinstance(snapshot, RParticles)
    np.testing.assert_allclose(snapshot.positions, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(snapshot.rotations, [[1.0, 0.0, 0.0, 0.0]] * 2)
    assert not snapshot.positions.flags.writeable
    with pytest.raises(ValueError, match=r"expected shape \(2, 3\)"):
        sparks.set_scales([[1.0, 1.0, 1.0]])


def test_meshes_after_particles_keep_their_own_material() -> None:
    from manimx.animation import Particles

    # A particle block adds one material row per particle: the mesh after
    # it must point past all of them, at its own (red) material.
    sparks = Particles(3, box_mesh(1.0, 1.0, 1.0), color="#ffffff")
    sparks.set_positions([[-3.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    after = Shape3D(box_mesh(1.0, 1.0, 1.0), color="#ff0000").move_to((2.0, 0.0, 0.0))
    config = RenderConfig(pixel_width=160, pixel_height=90, background_color="#000000", shadows=False)
    image = render_image([sparks, after], config=config)
    scale = 90 / 8  # pixels per unit with the default camera
    red = image[45, int(80 + 2.0 * scale), :3]
    assert red[0] > 120 and red[1] < red[0] / 4


def test_particles_draw_as_instances_with_their_color_and_hide_at_scale_zero() -> None:
    from manimx.animation import Particles

    sparks = Particles(3, box_mesh(1.0, 1.0, 1.0), color="#000000", emissive="#ffffff", emissive_strength=1.0)
    sparks.set_positions([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sparks.set_colors([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    sparks.set_scales([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])  # the third is hidden
    config = RenderConfig(pixel_width=160, pixel_height=90, background_color="#000000", shadows=False)
    image = render_image(sparks, config=config)

    scale = 90 / 8  # pixels per unit with the default camera
    red, green, hidden = (image[45, int(80 + x * scale), :3] for x in (-2.0, 0.0, 2.0))
    assert red[0] > 200 and red[0] > 4 * red[1]
    assert green[1] > 200 and green[1] > 4 * green[0]
    assert hidden.max() < 30
