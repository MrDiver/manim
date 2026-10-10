"""The retained vector scene and mesh geometry: patched renders must equal
fresh renders.
"""

from __future__ import annotations

import numpy as np
import pytest
from manimx.renderer import Camera, RenderConfig, RenderService
from manimx.renderer.cache import RNodeDataCache
from manimx.renderer.scene_graph import flatten_scene
from manimx.renderer.webgpu.retained import RetainedVectors
from manimx.robjects import (
    Fill,
    PbrMaterial,
    RClippingGroup,
    RDirectionalLight,
    RGroup,
    RShape2D,
    RShape3D,
    Stroke,
    Transform,
    new_uid,
)
from manimx.robjects.transform import rotation_towards
from manimx.userland.geometry import (
    arc_points,
    box_mesh,
    rectangle_points,
    regular_polygon_points,
    sphere_mesh,
)

CONFIG = RenderConfig(pixel_width=160, pixel_height=90)

# One uid per scene object, allocated like snapshots allocate them: the same
# object in different frames shares its uid.
A, B, C, D = (new_uid() for _ in range(4))


@pytest.fixture(scope="module")
def renderer() -> RenderService:
    try:
        return RenderService()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def circle():
    return arc_points()


def square():
    return rectangle_points(2.0, 2.0)


def triangle():
    return regular_polygon_points(3)


def shape(kind, uid, x=0.0, color="#e07a5f", opacity=1.0, width=4.0, alpha=1.0):
    return RShape2D(
        points=kind(), uid=uid, fill=Fill(color, opacity=opacity), stroke=Stroke("#f2cc8f", width=width),
        transform=Transform(position=(x, 0.0, 0.0)), alpha=alpha, versions={"points": ("p", uid, kind.__name__), "fill": (color, opacity), "stroke": width},
    )  # fmt: skip


def frames():
    """A sequence exercising every kind of patch."""
    yield [shape(circle, A, -2), shape(square, B, 2)]
    yield [shape(circle, A, -1.5), shape(square, B, 2)]  # one moves
    yield [shape(circle, A, -1.5, color="#5ea3d0"), shape(square, B, 2, opacity=0.5)]  # colors, opacity
    yield [shape(triangle, A, -1.5), shape(square, B, 2, width=8)]  # new points (fewer), new stroke
    yield [shape(square, B, 2)]  # one leaves
    yield [shape(square, B, 2), shape(circle, C, 0)]  # one arrives (reuses the slot)
    yield [RClippingGroup(mask=shape(circle, D, 0), content=RGroup(children=[shape(square, B, 0.5)]))]
    yield [shape(circle, C, 1), shape(square, B, -1)]  # order and placement swap
    yield [shape(circle, C, 1, alpha=0.5), shape(square, B, -1)]  # one fades
    yield [RGroup(children=[shape(circle, C, 1, alpha=0.5), shape(square, B, -1)], alpha=0.5)]  # a group fades too
    yield [shape(circle, C, 1, alpha=0.0), shape(square, B, -1)]  # faded out: kept, not drawn
    yield [shape(circle, C, 1), shape(square, B, -1)]  # and back


def test_patched_renders_equal_fresh_renders(renderer: RenderService) -> None:
    for index, children in enumerate(frames()):
        patched = renderer.render(RGroup(children=children), CONFIG)
        fresh = RenderService().render(RGroup(children=children), CONFIG)
        np.testing.assert_array_equal(patched, fresh, err_msg=f"frame {index}")


def test_a_moving_shape_only_rewrites_its_object_entry() -> None:
    retained, cache = RetainedVectors(), RNodeDataCache()
    retained.build(flatten_scene(RGroup(children=[shape(circle, A, 0), shape(square, B, 2)])), cache, np.eye(4))
    batch = retained.build(flatten_scene(RGroup(children=[shape(circle, A, 1), shape(square, B, 2)])), cache, np.eye(4))

    assert batch.dirty["objects"] == [(0, 1)]
    assert all(batch.dirty[name] == [] for name in ("instances", "fill_vertices", "fill_colors", "stroke_points", "stroke_curves"))
    unchanged = retained.build(flatten_scene(RGroup(children=[shape(circle, A, 1), shape(square, B, 2)])), cache, np.eye(4))
    assert all(ranges == [] for ranges in unchanged.dirty.values())


def test_a_fade_only_rewrites_the_object_entry() -> None:
    retained, cache = RetainedVectors(), RNodeDataCache()
    retained.build(flatten_scene(RGroup(children=[shape(circle, A, 0), shape(square, B, 2)])), cache, np.eye(4))
    batch = retained.build(flatten_scene(RGroup(children=[shape(circle, A, 0, alpha=0.5), shape(square, B, 2)])), cache, np.eye(4))

    assert batch.dirty["objects"] == [(0, 1)] and batch.objects[0]["alpha"] == 0.5
    assert all(batch.dirty[name] == [] for name in ("instances", "fill_vertices", "fill_colors", "stroke_points", "stroke_curves"))
    assert len(batch.transparent_draws) == 2 and len(batch.opaque_draws) == 2  # the faded circle blends


def test_alpha_renders_like_the_opacity_it_multiplies(renderer: RenderService) -> None:
    """Fading by alpha on the GPU looks like multiplying the style's
    opacities on the CPU (as snapshots did before), up to rounding."""
    faded = renderer.render(RGroup(children=[shape(circle, A, opacity=0.8, alpha=0.5)]), CONFIG)
    baked = RenderService().render(
        RGroup(children=[RShape2D(points=circle(), fill=Fill("#e07a5f", opacity=0.4), stroke=Stroke("#f2cc8f", opacity=0.5))]), CONFIG
    )
    assert np.abs(faded.astype(int) - baked.astype(int)).max() <= 1


def test_garbage_is_repacked() -> None:
    retained, cache = RetainedVectors(), RNodeDataCache()
    for _ in range(1, 400):  # every frame a new shape replaces the last: its ranges become garbage
        retained.build(flatten_scene(RGroup(children=[shape(circle, new_uid())])), cache, np.eye(4))
    assert retained.arenas["stroke_points"].used < 4096 + 400 and len(retained.slots) == 1


def mesh(make, uid, x=0.0):
    return RShape3D(
        mesh=make(), uid=uid, material=PbrMaterial("#e07a5f"), transform=Transform(position=(x, 0.0, 0.0)),
        versions={"mesh": (uid, make.__name__)},
    )  # fmt: skip


def small_box():
    return box_mesh(1.0, 1.0, 1.0)


def ball():
    return sphere_mesh(0.8)


def test_patched_meshes_render_like_fresh_ones(renderer: RenderService) -> None:
    config = RenderConfig(pixel_width=160, pixel_height=90, camera=Camera.look_at((0, -8, 4), (0, 0, 0)))
    sun = RDirectionalLight(transform=Transform(rotation=rotation_towards((1, 1, -1))))
    sequence = [
        [mesh(small_box, A, -2), mesh(ball, B, 2)],
        [mesh(small_box, A, -1), mesh(ball, B, 2)],  # one moves: geometry kept
        [mesh(ball, A, -1), mesh(ball, B, 2)],  # new mesh with another size
        [mesh(ball, B, 2)],  # one leaves
        [mesh(ball, B, 2), mesh(small_box, C, 0)],  # one arrives
        [mesh(small_box, C, 0), mesh(ball, B, 1)],  # order swaps
    ]
    for index, children in enumerate(sequence):
        root = RGroup(children=[*children, sun])
        np.testing.assert_array_equal(renderer.render(root, config), RenderService().render(root, config), err_msg=f"frame {index}")


def test_unchanged_meshes_write_no_geometry() -> None:
    from manimx.renderer.webgpu.mesh_batch import MeshBatch, RetainedMeshes

    geometry, cache = RetainedMeshes(), RNodeDataCache()
    first = MeshBatch.build(flatten_scene(RGroup(children=[mesh(small_box, A), mesh(ball, B, 2)])), cache, CONFIG, geometry)
    moved = MeshBatch.build(flatten_scene(RGroup(children=[mesh(small_box, A, 1), mesh(ball, B, 2)])), cache, CONFIG, geometry)
    assert first.dirty == {"vertices": None, "indices": None}  # new arrays: upload all
    assert moved.dirty == {"vertices": [], "indices": []}
