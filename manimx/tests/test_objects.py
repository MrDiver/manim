import math

import numpy as np
import pytest

from manimx import (
    CapStyle,
    RClippingGroup,
    Fill,
    RGroup,
    JointType,
    Quaternion,
    RNode,
    RShape2D,
    RShape3D,
    Stroke,
    Transform,
    quaternion_from_axis_angle,
)
from manimx.objects.geometry import arc_points, circle, rectangle, square, triangle
from manimx.objects.rshape2d import is_closed
from manimx.objects.style import resample, to_rgb_array
from manimx.utils.color import BLUE, RED

# --- RNode: identity and versioning ------------------------------------


def test_render_object_defaults_to_identity_transform() -> None:
    obj = RNode()

    assert obj.transform.rotation == Quaternion(1.0, 0.0, 0.0, 0.0)
    assert obj.visible is True


def test_uids_are_unique() -> None:
    assert len({RNode().uid for _ in range(100)}) == 100


def test_only_versioned_fields_bump_versions() -> None:
    shape = RShape2D.from_corners([(0, 0), (1, 0)])
    before = shape.version("points", "fill", "stroke")

    shape.transform = shape.transform.translated((1.0, 0.0, 0.0))
    assert shape.version("points", "fill", "stroke") == before

    shape.stroke = Stroke(width=1.0)
    assert shape.version("points", "fill", "stroke") == (before[0], before[1], before[2] + 1)


def test_version_rejects_unversioned_field() -> None:
    with pytest.raises(ValueError, match="transform is not versioned"):
        RShape2D().version("transform")


def test_shape_points_are_read_only() -> None:
    shape = RShape2D.from_corners([(0, 0), (1, 0)])

    with pytest.raises(ValueError, match="read-only"):
        shape.points[0, 0] = 5.0

    shape.points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert not shape.points.flags.writeable
    with pytest.raises(ValueError, match="four entries"):
        shape.points = [(0, 0)]


# --- Groups --------------------------------------------------------------------


def test_group_children_are_mutable() -> None:
    group = RGroup()
    child = RShape2D()

    group.children.append(child)

    assert group.children == [child]


def test_clipping_group_children_are_mask_and_content() -> None:
    mask, content = RShape2D(), RShape2D()

    assert RClippingGroup(mask=mask, content=content).children == [mask, content]
    assert RClippingGroup(content=content).children == [content]


# --- Transform -----------------------------------------------------------------


def test_transform_applies_scale_then_rotation_then_translation() -> None:
    transform = Transform(position=(1.0, 2.0, 3.0), scale=(2.0, 2.0, 2.0)).rotated(math.pi / 2)

    np.testing.assert_allclose(transform.apply([(1.0, 0.0, 0.0)]), [[1.0, 4.0, 3.0]], atol=1e-12)


def test_transform_normalizes_rotation() -> None:
    assert abs(Transform(rotation=Quaternion(2.0, 0.0, 0.0, 0.0)).rotation) == pytest.approx(1.0)


def test_transform_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="three components"):
        Transform(position=(1.0, 2.0))
    with pytest.raises(ValueError, match="must not be zero"):
        Transform(rotation=Quaternion(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="non-zero 3D vector"):
        quaternion_from_axis_angle((0.0, 0.0, 0.0), 1.0)


def test_transform_scaled_broadcasts_scalar() -> None:
    assert Transform().scaled(2.0).scale == (2.0, 2.0, 2.0)


# --- Styles --------------------------------------------------------------------


def test_style_lists_resample_over_points() -> None:
    np.testing.assert_allclose(resample(np.array([0.0, 1.0]), 5), [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(resample(np.array([2.0]), 3), [2.0, 2.0, 2.0])
    values = np.array([1.0, 2.0, 3.0])
    assert resample(values, 3) is values


def test_stroke_per_point_values() -> None:
    shape = RShape2D.from_corners(
        [(0, 0), (1, 0), (2, 0)],
        stroke=Stroke([RED, BLUE], width=[0.0, 8.0], opacity=0.5, joint=[JointType.ROUND, JointType.BEVEL]),
    )

    rgba = shape.stroke_rgba()
    assert rgba.shape == (8, 4)
    np.testing.assert_allclose(rgba[0, :3], RED.to_rgb())
    np.testing.assert_allclose(rgba[-1, :3], BLUE.to_rgb())
    np.testing.assert_allclose(rgba[:, 3], 0.5)
    np.testing.assert_allclose(shape.stroke_widths()[[0, -1]], [0.0, 8.0])
    assert shape.stroke_joints() == [JointType.ROUND, JointType.BEVEL]


def test_style_validation() -> None:
    with pytest.raises(ValueError, match="opacity"):
        Fill(opacity=2.0)
    with pytest.raises(ValueError, match="width"):
        Stroke(width=-1.0)
    with pytest.raises(ValueError, match="at least one color"):
        to_rgb_array([])
    with pytest.raises(ValueError, match="joint needs"):
        Stroke(joint=[])


def test_styles_compare_by_value() -> None:
    assert Stroke(RED, width=2.0) == Stroke(RED, width=2.0)
    assert Stroke(RED) != Stroke(BLUE)
    assert Fill(RED, opacity=1.0) == Fill(RED, opacity=1.0)
    assert Fill() != Stroke()
    assert Stroke(cap=CapStyle.ROUND) != Stroke(cap=CapStyle.BUTT)


# --- RShape2D geometry ----------------------------------------------------------


def test_shape2d_validates_points() -> None:
    with pytest.raises(ValueError, match="shape \\(N, 2\\)"):
        RShape2D(points=np.zeros((4, 3)))
    with pytest.raises(ValueError, match="four entries"):
        RShape2D(points=np.zeros((3, 2)))


def test_from_corners_builds_straight_cubics() -> None:
    shape = RShape2D.from_corners([(0, 0), (3, 0)])

    np.testing.assert_allclose(shape.points, [[0, 0], [1, 0], [2, 0], [3, 0]])


def test_subpaths_split_at_gaps() -> None:
    first = RShape2D.from_corners([(0, 0), (1, 0), (1, 1)], closed=True).points
    second = RShape2D.from_corners([(5, 5), (6, 5)]).points
    shape = RShape2D(points=np.vstack([first, second]))

    subpaths = shape.subpaths()

    assert [len(p) for p in subpaths] == [12, 4]
    assert is_closed(subpaths[0]) and not is_closed(subpaths[1])
    assert RShape2D().subpaths() == []


def test_geometry_matches_manim() -> None:
    # Values from manim's Circle() and Triangle().
    np.testing.assert_allclose(arc_points()[:4], [[1, 0], [1, 0.2652], [0.8946, 0.5196], [0.7071, 0.7071]], atol=1e-4)
    np.testing.assert_allclose(triangle().points[::4], [[0, 1], [-0.866, -0.5], [0.866, -0.5]], atol=1e-3)
    assert circle().num_curves == 8
    np.testing.assert_allclose(square().points[0], [1.0, 1.0])
    np.testing.assert_allclose(rectangle(4, 2).points[0], [2.0, 1.0])


def test_shape3d_versions_mesh_and_material() -> None:
    from manimx import PbrMaterial, sphere_mesh

    shape = RShape3D()
    before = shape.version("mesh", "material")
    shape.material = PbrMaterial(roughness=0.1)
    shape.mesh = sphere_mesh()
    assert shape.version("mesh", "material") == (before[0] + 1, before[1] + 1)


# --- reference image helper ---------------------------------------------------


def test_assert_image_matches_creates_compares_and_reports(tmp_path) -> None:
    from manimx.testing import ImageTolerance, assert_image_matches

    reference = tmp_path / "scene.png"
    image = np.zeros((4, 4, 4), np.uint8)
    with pytest.raises(AssertionError, match="did not exist"):
        assert_image_matches(image, reference)
    assert_image_matches(image, reference)  # now identical

    changed = image.copy()
    changed[0, 0] = 255
    with pytest.raises(AssertionError, match="outliers"):
        assert_image_matches(changed, reference)
    assert reference.with_suffix(".diff.png").exists()
    assert_image_matches(changed, reference, tolerance=ImageTolerance(mean_difference=255, outlier_fraction=1.0))
    assert not reference.with_suffix(".diff.png").exists()

    with pytest.raises(AssertionError, match="size"):
        assert_image_matches(np.zeros((2, 2, 4), np.uint8), reference)
    assert_image_matches(changed, reference, update=True)
    assert_image_matches(changed, reference)


def test_style_and_point_fast_paths_match_the_general_parsers() -> None:
    from manimx.objects.rbezier_path import _validated_points
    from manimx.objects.style import to_rgb_array, to_scalar_array

    triple = np.array([0.25, 0.5, 1.5])  # floats pass through unchanged, even out of [0, 1]
    assert np.array_equal(to_rgb_array(triple), [[0.25, 0.5, 1.5]]) and not to_rgb_array(triple).flags.writeable
    assert np.allclose(to_rgb_array(np.array([255, 0, 0])), [[1.0, 0.0, 0.0]])  # integers still mean 0-255
    assert to_scalar_array(0.5, "opacity", 0.0, 1.0).tolist() == [0.5]
    assert to_scalar_array(2, "width", 0.0, np.inf).dtype == np.float64
    with pytest.raises(ValueError, match="opacity must be in"):
        to_scalar_array(1.5, "opacity", 0.0, 1.0)
    assert to_scalar_array(True, "flag", 0.0, 1.0).tolist() == [1.0]  # bools take the general path

    frozen = np.zeros((4, 2))
    frozen.setflags(write=False)
    assert _validated_points(frozen, 2) is frozen  # owned and read-only: shared
    writable = np.zeros((4, 2))
    assert _validated_points(writable, 2) is not writable and not _validated_points(writable, 2).flags.writeable
    with pytest.raises(ValueError, match="four entries"):
        _validated_points(np.zeros((3, 2)), 2)
