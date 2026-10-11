"""Cube, Prism, Cylinder, Cone, Dot3D, Line3D, Arrow3D and ThreeDAxes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from manimx.animation import Scene
from manimx.userland import (
    Arrow3D,
    Cone,
    Cube,
    Cylinder,
    Dot3D,
    Line3D,
    Prism,
    ThreeDAxes,
)
from manimx.userland.geometry import cone_mesh, cylinder_mesh, prism_mesh, union_points


def outward(mesh) -> float:
    """The share of triangles whose winding agrees with their vertex normals."""
    positions, normals = mesh.positions, mesh.normals
    triangles = np.asarray(mesh.indices).reshape(-1, 3)
    faces = np.cross(positions[triangles[:, 1]] - positions[triangles[:, 0]], positions[triangles[:, 2]] - positions[triangles[:, 0]])
    return float((np.einsum("ij,ij->i", faces, normals[triangles].sum(axis=1)) > 0).mean())


def test_round_and_flat_solids_are_closed_and_face_outwards() -> None:
    for mesh in (cylinder_mesh(1.0, 2.0, 16), cone_mesh(1.0, 2.0, 16), prism_mesh(5, 1.0, 2.0), cylinder_mesh(caps=False), cone_mesh(cap=False)):
        assert outward(mesh) == 1.0
        np.testing.assert_allclose(mesh.positions[:, 2].min(), -1.0)
        np.testing.assert_allclose(mesh.positions[:, 2].max(), 1.0)
    assert len(cylinder_mesh(segments=8, caps=False).indices) == 8 * 6
    assert np.allclose(cone_mesh(segments=8).positions[cone_mesh(segments=8).positions[:, 2] > 0.99][:, :2], 0.0)  # the apex
    with pytest.raises(ValueError, match="3 sides"):
        prism_mesh(2)
    side = Cube(3.0).mesh.value.positions
    assert np.ptp(side, axis=0).tolist() == [3.0, 3.0, 3.0]
    for solid in (Prism(), Cylinder(), Cone()):
        assert len(solid.mesh.value.positions) > 0


def test_lines_arrows_and_dots_mark_points_in_space() -> None:
    line = Line3D((0, 0, 0), (3, 0, 0))
    np.testing.assert_allclose(line.points.value[[0, -1]], [[0, 0, 0], [3, 0, 0]])

    class Built(Scene):
        def construct(self):
            self.add(line)
            self.play(line.animate.set_ends((0, 0, 0), (0, 0, 3)))

    prepared = Built().prepare()
    prepared.evaluate(1.0)
    np.testing.assert_allclose(line.points.value[-1], [0, 0, 3])
    assert line.stroke_width.value == 4.0  # 0.04 world units, like a Shape2D's default

    arrow = Arrow3D((0, 0, 0), (0, 0, 2), tip_length=0.5)
    np.testing.assert_allclose(arrow.line.points.value[-1], [0, 0, 1.5])
    assert arrow.tip.position.z.value == pytest.approx(1.75)
    down = Arrow3D((0, 0, 0), (0, 0, -2))
    assert abs(down.tip.rotation.value.w) < 1e-9  # turned half around
    sideways = Arrow3D((0, 0, 0), (2, 0, 0), color="#ff0000")
    turned = np.array([quaternion_rotate(sideways.tip.rotation.value, (0.0, 0.0, 1.0))])
    np.testing.assert_allclose(turned[0], [1, 0, 0], atol=1e-9)
    with pytest.raises(ValueError, match="no room"):
        Arrow3D((0, 0, 0), (0.1, 0, 0), tip_length=0.3)
    assert Dot3D().mesh.value.positions[:, 0].max() == pytest.approx(0.08)


def quaternion_rotate(rotation, vector):
    import quaternion

    return quaternion.rotate_vectors(rotation, np.asarray(vector, dtype=np.float64))


def test_three_d_axes_place_coordinates_surfaces_and_curves() -> None:
    axes = ThreeDAxes((-2, 2, 1), (-2, 2, 1), (0, 4, 1), 4.0, 4.0, 4.0).move_to((1.0, 0.0, 0.0))
    np.testing.assert_allclose(axes.coords_to_point(1, -1, 3), [2.0, -1.0, 1.0])
    np.testing.assert_allclose(axes.z_axis.position.z.value, 0.0)
    assert axes.x_axis.position.z.value == pytest.approx(-2.0)  # where z is 0, the nearest end
    surface = axes.plot_surface(lambda x, y: x + y, resolution=(4, 4))
    positions = surface.mesh.value.positions
    np.testing.assert_allclose(positions[0], axes.coords_to_point(-2, -2, -4))
    curve = axes.plot_curve(lambda t: (math.cos(t), math.sin(t), 2.0), (0.0, math.tau), 16)
    np.testing.assert_allclose(curve.points.value[0], axes.coords_to_point(1, 0, 2), atol=1e-9)


def test_union_points_merges_overlapping_pieces_into_one_outline() -> None:
    from manimx.userland.geometry import rectangle_points, subpaths

    first, second = rectangle_points(2.0, 1.0), rectangle_points(2.0, 1.0) + [1.0, 0.0]
    merged = union_points([first, second])
    assert len(subpaths(merged)) == 1
    np.testing.assert_allclose(merged.min(axis=0), [-1.0, -0.5])
    np.testing.assert_allclose(merged.max(axis=0), [2.0, 0.5])
    apart = union_points([first, first + [5.0, 0.0]])
    assert len(subpaths(apart)) == 2
    assert len(union_points([])) == 0
