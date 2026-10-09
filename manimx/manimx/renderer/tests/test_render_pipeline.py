"""CPU-side renderer pieces: scene graph, cache, commands, tessellation, batching."""

import math

import numpy as np
import pytest
from manimx.animation.geometry import arc_points, rectangle_points
from manimx.objects import new_uid
from manimx.renderer import ClearScene, RemoveObjects, RNodeDataCache
from manimx.renderer.config import Camera, RenderConfig
from manimx.renderer.scene_graph import NO_CLIP, bounding_box, flatten, flatten_scene
from manimx.renderer.tessellation import (
    SIDE_CHORD,
    SIDE_CONTROL,
    SIDE_SOLID,
    tessellate_fill,
)
from manimx.renderer.webgpu.vector_batch import (
    NO_NEIGHBOR,
    STROKE_WIDTH_SCALE,
    VectorBatch,
)

from manimx import (
    Fill,
    JointType,
    RClippingGroup,
    RGroup,
    RNode,
    RShape2D,
    Stroke,
    Transform,
)

# --- scene graph ----------------------------------------------------------------


def test_flatten_composes_transforms_and_keeps_painter_order() -> None:
    a = RShape2D(transform=Transform(position=(1.0, 0.0, 0.0)))
    b = RShape2D()
    hidden = RShape2D(visible=False)
    root = RGroup(children=[a, RGroup(children=[hidden, b])], transform=Transform(position=(0.0, 2.0, 0.0)))

    items = flatten(root)

    assert [item.obj for item in items] == [a, b]
    assert [item.order for item in items] == [0, 1]
    np.testing.assert_allclose(items[0].world[:3, 3], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(items[1].world[:3, 3], [0.0, 2.0, 0.0])


def test_flatten_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError, match="RNode"):
        flatten(RNode())
    nested_in_mask = RClippingGroup(mask=RClippingGroup(), content=RShape2D())
    with pytest.raises(NotImplementedError, match="inside a mask"):
        flatten(nested_in_mask)


def test_clipping_groups_open_nested_scopes() -> None:
    outer_mask, inner_mask = RShape2D(), RShape2D()
    free, outer_content, inner_content = RShape2D(), RShape2D(), RShape2D()
    inner = RClippingGroup(mask=inner_mask, content=inner_content)
    outer = RClippingGroup(
        mask=outer_mask,
        content=RGroup(children=[outer_content, inner]),
        transform=Transform(position=(1.0, 0.0, 0.0)),
    )

    scene = flatten_scene(RGroup(children=[free, outer]))

    assert [item.obj for item in scene.items] == [free, outer_content, inner_content]
    assert [item.clip for item in scene.items] == [NO_CLIP, 0, 1]
    assert [scope.parent for scope in scene.clips] == [NO_CLIP, 0]
    assert [item.obj for item in scene.clips[0].mask] == [outer_mask]
    # The inner mask is drawn under the outer scope, which clips it.
    assert [(item.obj, item.clip) for item in scene.clips[1].mask] == [(inner_mask, 0)]
    np.testing.assert_allclose(scene.clips[0].mask[0].world[:3, 3], [1.0, 0.0, 0.0])


def test_clipping_group_without_mask_does_not_clip() -> None:
    content = RShape2D()

    scene = flatten_scene(RClippingGroup(content=content))

    assert scene.clips == [] and scene.items[0].clip == NO_CLIP


def test_bounding_box_uses_world_anchors() -> None:
    root = RGroup(children=[RShape2D(points=rectangle_points(2.0, 2.0), transform=Transform(position=(1.0, 0.0, 0.0)))])

    low, high = bounding_box(root)

    np.testing.assert_allclose(low, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(high, [2.0, 1.0, 0.0])
    np.testing.assert_allclose(bounding_box(RGroup())[0], [0.0, 0.0, 0.0])


# --- cache and commands -----------------------------------------------------------


def test_cache_misses_on_new_version() -> None:
    cache, uid = RNodeDataCache(), new_uid()
    cache.put(uid, "mesh", (1,), "old")

    assert cache.get(uid, "mesh", (1,)) == "old"
    assert cache.get(uid, "mesh", (2,)) is None
    assert cache.get_or_create(uid, "mesh", (2,), lambda: "new") == "new"
    assert cache.get_or_create(uid, "mesh", (2,), lambda: "unused") == "new"
    assert len(cache) == 1


def test_cache_retain_and_clear() -> None:
    cache = RNodeDataCache()
    first, second, third = new_uid(), new_uid(), new_uid()
    for uid in (first, second, third):
        cache.put(uid, "mesh", 0, uid)

    cache.retain([second])
    assert len(cache) == 1 and cache.get(second, "mesh", 0) == second
    cache.clear()
    assert len(cache) == 0


def test_remove_objects_command_drops_whole_subtree() -> None:
    a, b, keep = RShape2D(), RShape2D(), RShape2D()
    group = RGroup(children=[a, RClippingGroup(content=b)])
    cache = RNodeDataCache()
    for obj in (a, b, keep):
        cache.put(obj.uid, "mesh", 0, obj.uid)
        cache.put(obj.uid, "other", 0, obj.uid)

    cache.execute(RemoveObjects.subtree(group))

    assert len(cache) == 2 and cache.get(keep.uid, "mesh", 0) == keep.uid
    cache.execute(ClearScene())
    assert len(cache) == 0


# --- tessellation ------------------------------------------------------------------


def _filled_area(points: np.ndarray) -> float:
    """Area covered by a fill mesh, using the parabolic segment rule."""
    mesh = tessellate_fill(points)
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
    open_triangle = RShape2D.from_corners([(0, 0), (2, 0), (0, 2)]).points

    assert _filled_area(open_triangle) == pytest.approx(2.0)


def test_tessellation_skips_degenerate_paths() -> None:
    line = RShape2D.from_corners([(0, 0), (1, 0)]).points

    assert len(tessellate_fill(line)) == 0
    assert len(tessellate_fill(np.zeros((0, 2)))) == 0


def test_tessellation_records_where_every_vertex_comes_from() -> None:
    """Vertices carry (curve, t) instead of a color: the GPU colors them from
    the shape's per-point fill colors, so color changes never re-tessellate."""
    points = np.vstack([arc_points(1.0), arc_points(0.5)])  # two subpaths
    mesh = tessellate_fill(points)
    curves = points.reshape(-1, 4, 2)

    assert set(np.unique(mesh.curve)) == set(range(len(curves)))  # both subpaths, global indices
    assert mesh.t.min() >= 0.0 and mesh.t.max() <= 1.0
    # Curve triangles start and end on their source cubic.
    curved = np.flatnonzero(mesh.side != SIDE_SOLID)[::3]
    for first in curved:
        for corner in (0, 2):
            index = first + corner
            control = curves[mesh.curve[index]]
            t = mesh.t[index]
            weights = np.array([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3])
            np.testing.assert_allclose(mesh.positions[index], weights @ control, atol=1e-9)


# --- vector batch -----------------------------------------------------------------


def _batch(*shapes: RShape2D, cache: RNodeDataCache | None = None) -> VectorBatch:
    root = RGroup(children=list(shapes))
    if cache is None:  # not `or`: an empty cache is falsy
        cache = RNodeDataCache()
    return VectorBatch.build(flatten_scene(root), cache, np.array([0.0, 0.0, 11.0]))


def test_batch_links_closed_subpaths_into_a_loop() -> None:
    batch = _batch(RShape2D(points=rectangle_points(2.0, 2.0), stroke=Stroke(joint=JointType.ROUND)))
    curves = batch.stroke_curves

    assert list(curves["first_point"]) == [0, 4, 8, 12]
    assert list(curves["prev_point"]) == [12, 0, 4, 8]
    assert list(curves["next_point"]) == [4, 8, 12, 0]
    assert set(curves["start_joint"]) == {JointType.ROUND.value}


def test_batch_leaves_open_path_ends_unlinked() -> None:
    curves = _batch(RShape2D.from_corners([(0, 0), (1, 0), (2, 1)])).stroke_curves

    assert list(curves["prev_point"]) == [NO_NEIGHBOR, 0]
    assert list(curves["next_point"]) == [4, NO_NEIGHBOR]


def test_batch_matches_shared_anchor_styles() -> None:
    batch = _batch(RShape2D.from_corners([(0, 0), (1, 0), (2, 0)], stroke=Stroke(width=[0.0, 7.0])))
    widths = batch.stroke_points["width"]

    assert widths[3] == pytest.approx(widths[4])
    assert widths[0] == 0.0 and widths[-1] == pytest.approx(7.0 * STROKE_WIDTH_SCALE)


def test_batch_splits_and_sorts_transparent_draws() -> None:
    opaque = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(opacity=1.0))
    back = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(opacity=0.5), stroke=Stroke(width=0.0))
    front = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(opacity=0.5), stroke=Stroke(opacity=0.0))
    batch = _batch(front, opaque, back)

    assert len(batch.opaque_draws) == 2  # opaque fill + opaque stroke
    # Coplanar transparent draws keep painter's order: front (order 0) first.
    first_instances = batch.transparent_draws["first_instance"]
    assert [batch.instances["object"][i] for i in first_instances] == [0, 2]


def test_batch_reuses_cached_fill_mesh_until_points_change() -> None:
    cache = RNodeDataCache()
    shape = RShape2D(points=arc_points(), fill=Fill(opacity=1.0))
    _batch(shape, cache=cache)
    mesh = cache.get(shape.uid, "fill-mesh", shape.version("points"))
    assert mesh is not None

    shape.transform = shape.transform.translated((1.0, 0.0, 0.0))
    _batch(shape, cache=cache)
    assert cache.get(shape.uid, "fill-mesh", shape.version("points")) is mesh

    shape.points = arc_points(2.0)
    _batch(shape, cache=cache)
    assert cache.get(shape.uid, "fill-mesh", shape.version("points")) is not mesh


def test_batch_draws_masks_separately_and_clipped_content_transparent() -> None:
    mask = RShape2D(points=arc_points(), fill=Fill(opacity=1.0), stroke=Stroke(width=0.0))
    content = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(opacity=1.0), stroke=Stroke(width=0.0))
    clip = RClippingGroup(mask=mask, content=content)

    batch = VectorBatch.build(flatten_scene(clip), RNodeDataCache(), np.zeros(3))

    assert len(batch.opaque_draws) == 0
    assert len(batch.transparent_draws) == 1
    assert len(batch.mask_draws) == 1 and len(batch.mask_draws[0]) == 1
    assert list(batch.objects["clip"]) == [0, NO_CLIP]  # content, then mask


def test_batch_skips_empty_and_invisible_styles() -> None:
    batch = _batch(RShape2D(), RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(opacity=0.0), stroke=Stroke(width=0.0)))

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


def test_look_at_points_the_camera_at_the_target() -> None:
    camera = Camera.look_at((10.0, -10.0, 5.0), (1.0, 1.0, 1.0))
    view = camera.view_matrix()

    target = view @ [1.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(target[:2], [0.0, 0.0], atol=1e-9)
    assert target[2] < 0.0  # in front of the camera (it looks along -z)
    assert (view @ [0.0, 0.0, 1.0, 0.0])[1] > 0.0  # world up is screen up
    with pytest.raises(ValueError, match="parallel"):
        Camera.look_at((0.0, 0.0, 5.0))
