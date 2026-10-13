import math

import numpy as np
import pytest
from manimx.robjects import (
    Fill,
    Quaternion,
    RClippingGroup,
    RGroup,
    RNode,
    RShape2D,
    Stroke,
    Transform,
    quaternion_from_axis_angle,
    quaternion_from_rotation_matrix,
)
from manimx.robjects.style import to_rgb_array
from manimx.userland.geometry import (
    arc_points,
    corner_points,
    rectangle_points,
    regular_polygon_points,
    subpaths,
)
from manimx.utils.color import BLUE, RED

from manimx import CapStyle, JointType

# --- RNode: identity ------------------------------------


def test_render_object_defaults_to_identity_transform() -> None:
    obj = RGroup()

    assert isinstance(obj, RNode)
    assert obj.transform.rotation == Quaternion(1.0, 0.0, 0.0, 0.0)
    assert obj.alpha == 1.0
    with pytest.raises(TypeError):
        RNode()  # only its subclasses are render objects


def test_uids_are_unique() -> None:
    assert len({RGroup().uid for _ in range(100)}) == 100


def test_render_objects_and_their_points_are_read_only() -> None:
    shape = RShape2D(points=corner_points([(0, 0), (1, 0)]))

    with pytest.raises(ValueError, match="read-only"):
        shape.points[0, 0] = 5.0
    with pytest.raises(AttributeError):
        shape.points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    with pytest.raises(AttributeError):
        shape.alpha = 0.5


def test_values_keep_their_versions_in_every_node_that_holds_them() -> None:
    fill = Fill("#FF0000", opacity=1.0)
    first, second = RShape2D(fill=fill), RShape2D(fill=fill)

    assert first.version("fill") == second.version("fill")
    assert first.version("points") != second.version("points")  # new points: a new version
    with pytest.raises(ValueError, match="no versioned field"):
        first.version("mesh")


# --- Groups --------------------------------------------------------------------


def test_group_children_are_the_given_nodes() -> None:
    child = RShape2D()

    (held,) = RGroup(children=[child]).children

    assert isinstance(held, RShape2D) and held.uid == child.uid
    with pytest.raises(TypeError, match="render object"):
        RGroup(children=["not a node"])


def test_clipping_group_children_are_mask_and_content() -> None:
    mask, content = RShape2D(), RShape2D()

    assert [child.uid for child in RClippingGroup(mask=mask, content=content).children] == [mask.uid, content.uid]
    assert [child.uid for child in RClippingGroup(content=content).children] == [content.uid]
    assert RClippingGroup(content=content).mask is None


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


def test_rotation_helpers_match_numpy_quaternion() -> None:
    import quaternion

    generator = np.random.default_rng(3)
    for _ in range(200):
        axis, angle = generator.normal(size=3), generator.uniform(-7, 7)
        expected = quaternion.from_rotation_vector(axis / np.linalg.norm(axis) * angle)
        assert np.allclose(quaternion.as_float_array(quaternion_from_axis_angle(axis, angle)), quaternion.as_float_array(expected))
        matrix = quaternion.as_rotation_matrix(expected)
        found = quaternion.as_float_array(quaternion_from_rotation_matrix(matrix))
        wanted = quaternion.as_float_array(expected)
        assert np.allclose(found, wanted) or np.allclose(found, -wanted)  # q and -q are one rotation
    for turn in (np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]), np.eye(3)):
        assert np.allclose(quaternion.as_rotation_matrix(quaternion_from_rotation_matrix(turn)), turn)
    with pytest.raises(ValueError, match="non-zero 3D vector"):
        quaternion_from_axis_angle((1.0, 2.0), 1.0)


def test_transform_scaled_broadcasts_scalar() -> None:
    assert Transform().scaled(2.0).scale == (2.0, 2.0, 2.0)


# --- Styles --------------------------------------------------------------------


def test_stroke_lists_are_kept_as_given() -> None:
    shape = RShape2D(points=corner_points([(0, 0), (1, 0), (2, 0)]), stroke=Stroke([RED, BLUE], width=[0.0, 8.0], opacity=0.5, joint=[JointType.ROUND, JointType.BEVEL]))

    stroke = shape.stroke
    np.testing.assert_allclose(stroke.color, [RED.to_rgb(), BLUE.to_rgb()])
    np.testing.assert_allclose(stroke.width, [0.0, 8.0])
    np.testing.assert_allclose(stroke.opacity, [0.5])
    assert stroke.joint == (JointType.ROUND, JointType.BEVEL) and stroke.cap is CapStyle.AUTO


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
    shape = RShape2D(points=corner_points([(0, 0), (3, 0)]))

    np.testing.assert_allclose(shape.points, [[0, 0], [1, 0], [2, 0], [3, 0]])


def test_subpaths_split_at_gaps() -> None:
    first = corner_points([(0, 0), (1, 0), (1, 1)], closed=True)
    second = corner_points([(5, 5), (6, 5)])

    assert [len(p) for p in subpaths(np.vstack([first, second]))] == [12, 4]


def test_geometry_matches_manim() -> None:
    # Values from manim's Circle() and Triangle().
    np.testing.assert_allclose(arc_points()[:4], [[1, 0], [1, 0.2652], [0.8946, 0.5196], [0.7071, 0.7071]], atol=1e-4)
    np.testing.assert_allclose(RShape2D(points=regular_polygon_points(3, 1.0)).points[::4], [[0, 1], [-0.866, -0.5], [0.866, -0.5]], atol=1e-3)
    assert len(RShape2D(points=arc_points()).points) == 8 * 4
    np.testing.assert_allclose(RShape2D(points=rectangle_points(2.0, 2.0)).points[0], [1.0, 1.0])
    np.testing.assert_allclose(RShape2D(points=rectangle_points(4, 2)).points[0], [2.0, 1.0])


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


def test_style_fast_paths_match_the_general_parsers() -> None:
    from manimx.robjects.style import to_rgb_array, to_scalar_array

    triple = np.array([0.25, 0.5, 1.5])  # floats pass through unchanged, even out of [0, 1]
    assert np.array_equal(to_rgb_array(triple), [[0.25, 0.5, 1.5]]) and not to_rgb_array(triple).flags.writeable
    assert np.allclose(to_rgb_array(np.array([255, 0, 0])), [[1.0, 0.0, 0.0]])  # integers still mean 0-255
    assert to_scalar_array(0.5, "opacity", 0.0, 1.0).tolist() == [0.5]
    assert to_scalar_array(2, "width", 0.0, np.inf).dtype == np.float64
    with pytest.raises(ValueError, match="opacity must be in"):
        to_scalar_array(1.5, "opacity", 0.0, 1.0)
    assert to_scalar_array(True, "flag", 0.0, 1.0).tolist() == [1.0]  # bools take the general path
