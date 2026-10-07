"""CPU-side renderer pieces: scene graph, cache, commands, tessellation, batching."""

import math

import numpy as np
import pytest

from manimx import ClippingGroup, Fill, Group, JointType, RenderObject, Shape2D, Stroke, Transform
from manimx.objects.geometry import arc_points, circle, square
from manimx.renderer import ClearScene, MobjectDataCache, RemoveObjects
from manimx.renderer.config import Camera, RenderConfig
from manimx.renderer.scene_graph import bounding_box, flatten
from manimx.renderer.tessellation import SIDE_CHORD, SIDE_CONTROL, SIDE_SOLID, tessellate_fill
from manimx.renderer.webgpu.vector_batch import NO_NEIGHBOR, STROKE_WIDTH_SCALE, VectorBatch

# --- scene graph ----------------------------------------------------------------


def test_flatten_composes_transforms_and_keeps_painter_order() -> None:
    a = Shape2D("a", transform=Transform(position=(1.0, 0.0, 0.0)))
    b = Shape2D("b")
    hidden = Shape2D("hidden", visible=False)
    root = Group("root", children=[a, Group("inner", children=[hidden, b])], transform=Transform(position=(0.0, 2.0, 0.0)))

    items = flatten(root)

    assert [item.obj for item in items] == [a, b]
    assert [item.order for item in items] == [0, 1]
    np.testing.assert_allclose(items[0].world[:3, 3], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(items[1].world[:3, 3], [0.0, 2.0, 0.0])


def test_flatten_rejects_unsupported_objects() -> None:
    with pytest.raises(NotImplementedError, match="ClippingGroup"):
        flatten(ClippingGroup("clip"))
    with pytest.raises(TypeError, match="RenderObject"):
        flatten(RenderObject("plain"))


def test_bounding_box_uses_world_anchors() -> None:
    root = Group("g", children=[square("s", transform=Transform(position=(1.0, 0.0, 0.0)))])

    low, high = bounding_box(root)

    np.testing.assert_allclose(low, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(high, [2.0, 1.0, 0.0])
    np.testing.assert_allclose(bounding_box(Group("empty"))[0], [0.0, 0.0, 0.0])


# --- cache and commands -----------------------------------------------------------


def test_cache_misses_on_new_version() -> None:
    cache = MobjectDataCache()
    cache.put(1, "mesh", (1,), "old")

    assert cache.get(1, "mesh", (1,)) == "old"
    assert cache.get(1, "mesh", (2,)) is None
    assert cache.get_or_create(1, "mesh", (2,), lambda: "new") == "new"
    assert cache.get_or_create(1, "mesh", (2,), lambda: "unused") == "new"
    assert len(cache) == 1


def test_cache_retain_and_clear() -> None:
    cache = MobjectDataCache()
    for uid in (1, 2, 3):
        cache.put(uid, "mesh", 0, uid)

    cache.retain([2])
    assert len(cache) == 1 and cache.get(2, "mesh", 0) == 2
    cache.clear()
    assert len(cache) == 0


def test_remove_objects_command_drops_whole_subtree() -> None:
    a, b, keep = Shape2D("a"), Shape2D("b"), Shape2D("keep")
    group = Group("g", children=[a, ClippingGroup("clip", content=b)])
    cache = MobjectDataCache()
    for obj in (a, b, keep):
        cache.put(obj.uid, "mesh", 0, obj.id)
        cache.put(obj.uid, "other", 0, obj.id)

    cache.execute(RemoveObjects.subtree(group))

    assert len(cache) == 2 and cache.get(keep.uid, "mesh", 0) == "keep"
    cache.execute(ClearScene())
    assert len(cache) == 0


# --- tessellation ------------------------------------------------------------------


def _filled_area(points: np.ndarray) -> float:
    """Area covered by a fill mesh, using the parabolic segment rule."""
    mesh = tessellate_fill(points, np.ones((len(points), 4)))
    triangles = mesh.positions.reshape(-1, 3, 2)
    sides = mesh.side[::3]
    edges1, edges2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    areas = np.abs(edges1[:, 0] * edges2[:, 1] - edges1[:, 1] * edges2[:, 0]) / 2
    # Chord side of a quadratic covers 2/3 of its triangle, control side 1/3.
    weights = np.select([sides == SIDE_SOLID, sides == SIDE_CHORD, sides == SIDE_CONTROL], [1.0, 2 / 3, 1 / 3])
    return float(np.sum(areas * weights))


def _bezier_circle_area(radius: float) -> float:
    """Exact area of Manim's 8-cubic circle approximation (not quite pi r^2)."""
    return 3.1416093 * radius**2


def test_tessellated_circle_area_is_exact() -> None:
    assert _filled_area(arc_points(1.0)) == pytest.approx(_bezier_circle_area(1.0), rel=1e-5)


def test_tessellation_is_orientation_independent() -> None:
    clockwise = arc_points(1.0, angle=-math.tau)

    assert _filled_area(clockwise) == pytest.approx(_bezier_circle_area(1.0), rel=1e-5)


def test_tessellation_cuts_holes_by_nesting() -> None:
    annulus = np.vstack([arc_points(1.0), arc_points(0.5)])

    assert _filled_area(annulus) == pytest.approx(_bezier_circle_area(1.0) - _bezier_circle_area(0.5), rel=1e-5)


def test_tessellation_closes_open_paths() -> None:
    open_triangle = Shape2D.from_corners("t", [(0, 0), (2, 0), (0, 2)]).points

    assert _filled_area(open_triangle) == pytest.approx(2.0)


def test_tessellation_skips_degenerate_paths() -> None:
    line = Shape2D.from_corners("l", [(0, 0), (1, 0)]).points

    assert len(tessellate_fill(line, np.ones((4, 4)))) == 0
    assert len(tessellate_fill(np.zeros((0, 2)), np.zeros((0, 4)))) == 0


def test_tessellation_interpolates_point_colors() -> None:
    points = square("s").points
    colors = np.zeros((len(points), 4))
    colors[:, 0] = np.linspace(0.0, 1.0, len(points))

    mesh = tessellate_fill(points, colors)

    assert mesh.colors[:, 0].min() >= 0.0 and mesh.colors[:, 0].max() <= 1.0
    assert len(np.unique(mesh.colors[:, 0])) > 1


# --- vector batch -----------------------------------------------------------------


def _batch(*shapes: Shape2D, cache: MobjectDataCache | None = None) -> VectorBatch:
    root = Group("root", children=list(shapes))
    if cache is None:  # not `or`: an empty cache is falsy
        cache = MobjectDataCache()
    return VectorBatch.build(flatten(root), cache, np.array([0.0, 0.0, 11.0]))


def test_batch_links_closed_subpaths_into_a_loop() -> None:
    batch = _batch(square("s", stroke=Stroke(joint=JointType.ROUND)))
    curves = batch.stroke_curves

    assert list(curves["first_point"]) == [0, 4, 8, 12]
    assert list(curves["prev_point"]) == [12, 0, 4, 8]
    assert list(curves["next_point"]) == [4, 8, 12, 0]
    assert set(curves["start_joint"]) == {JointType.ROUND.value}


def test_batch_leaves_open_path_ends_unlinked() -> None:
    curves = _batch(Shape2D.from_corners("l", [(0, 0), (1, 0), (2, 1)])).stroke_curves

    assert list(curves["prev_point"]) == [NO_NEIGHBOR, 0]
    assert list(curves["next_point"]) == [4, NO_NEIGHBOR]


def test_batch_matches_shared_anchor_styles() -> None:
    batch = _batch(Shape2D.from_corners("l", [(0, 0), (1, 0), (2, 0)], stroke=Stroke(width=[0.0, 7.0])))
    widths = batch.stroke_points["width"]

    assert widths[3] == pytest.approx(widths[4])
    assert widths[0] == 0.0 and widths[-1] == pytest.approx(7.0 * STROKE_WIDTH_SCALE)


def test_batch_splits_and_sorts_transparent_draws() -> None:
    opaque = square("opaque", fill=Fill(opacity=1.0))
    back = square("back", fill=Fill(opacity=0.5), stroke=Stroke(width=0.0))
    front = square("front", fill=Fill(opacity=0.5), stroke=Stroke(opacity=0.0))
    batch = _batch(front, opaque, back)

    assert len(batch.opaque_draws) == 2  # opaque fill + opaque stroke
    # Coplanar transparent draws keep painter's order: front (order 0) first.
    first_instances = batch.transparent_draws["first_instance"]
    assert [batch.instances["object"][i] for i in first_instances] == [0, 2]


def test_batch_reuses_cached_fill_mesh_until_points_change() -> None:
    cache = MobjectDataCache()
    shape = circle("c", fill=Fill(opacity=1.0))
    _batch(shape, cache=cache)
    mesh = cache.get(shape.uid, "fill-mesh", shape.version("points", "fill"))
    assert mesh is not None

    shape.transform = shape.transform.translated((1.0, 0.0, 0.0))
    _batch(shape, cache=cache)
    assert cache.get(shape.uid, "fill-mesh", shape.version("points", "fill")) is mesh

    shape.points = arc_points(2.0)
    _batch(shape, cache=cache)
    assert cache.get(shape.uid, "fill-mesh", shape.version("points", "fill")) is not mesh


def test_batch_skips_empty_and_invisible_styles() -> None:
    batch = _batch(Shape2D("empty"), square("nothing", fill=Fill(opacity=0.0), stroke=Stroke(width=0.0)))

    assert len(batch.opaque_draws) == len(batch.transparent_draws) == 0
    assert len(batch.instances) == 0


# --- config ------------------------------------------------------------------------


def test_orthographic_camera_shows_frame_height() -> None:
    config = RenderConfig(pixel_width=1600, pixel_height=900)
    clip = config.camera.projection_matrix(config.aspect_ratio) @ config.camera.view_matrix() @ np.array(
        [config.frame_width / 2, 4.0, 0.0, 1.0]
    )

    np.testing.assert_allclose(clip[:2] / clip[3], [1.0, 1.0])
    assert 0.0 < clip[2] / clip[3] < 1.0


def test_perspective_camera_maps_near_and_far_planes() -> None:
    camera = Camera(field_of_view=math.pi / 2, near=1.0, far=10.0)
    projection = camera.projection_matrix(1.0)

    near, far = projection @ [0, 0, -1.0, 1], projection @ [0, 0, -10.0, 1]
    assert near[2] / near[3] == pytest.approx(0.0)
    assert far[2] / far[3] == pytest.approx(1.0)
