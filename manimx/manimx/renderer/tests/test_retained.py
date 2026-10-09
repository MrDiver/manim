"""The retained vector scene: patched renders must equal fresh renders."""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation.geometry import (
    arc_points,
    rectangle_points,
    regular_polygon_points,
)
from manimx.robjects import new_uid
from manimx.renderer import RenderConfig, WGPURenderer
from manimx.renderer.cache import RNodeDataCache
from manimx.renderer.scene_graph import flatten_scene
from manimx.renderer.webgpu.retained import RetainedVectors

from manimx import Fill, RClippingGroup, RGroup, RShape2D, Stroke, Transform

CONFIG = RenderConfig(pixel_width=160, pixel_height=90)

# One uid per scene object, allocated like snapshots allocate them: the same
# object in different frames shares its uid.
A, B, C, D = (new_uid() for _ in range(4))


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def circle():
    return arc_points()


def square():
    return rectangle_points(2.0, 2.0)


def triangle():
    return regular_polygon_points(3)


def shape(kind, uid, x=0.0, color="#e07a5f", opacity=1.0, width=4.0):
    return RShape2D(
        points=kind(), uid=uid, fill=Fill(color, opacity=opacity), stroke=Stroke("#f2cc8f", width=width),
        transform=Transform(position=(x, 0.0, 0.0)), versions={"points": ("p", uid, kind.__name__), "fill": (color, opacity), "stroke": width},
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


def test_patched_renders_equal_fresh_renders(renderer: WGPURenderer) -> None:
    for index, children in enumerate(frames()):
        patched = renderer.render(RGroup(children=children), CONFIG)
        fresh = WGPURenderer().render(RGroup(children=children), CONFIG)
        np.testing.assert_array_equal(patched, fresh, err_msg=f"frame {index}")


def test_a_moving_shape_only_rewrites_its_object_entry() -> None:
    retained, cache = RetainedVectors(), RNodeDataCache()
    retained.build(flatten_scene(RGroup(children=[shape(circle, A, 0), shape(square, B, 2)])), cache, np.zeros(3))
    batch = retained.build(flatten_scene(RGroup(children=[shape(circle, A, 1), shape(square, B, 2)])), cache, np.zeros(3))

    assert batch.dirty["objects"] == [(0, 1)]
    assert all(batch.dirty[name] == [] for name in ("instances", "fill_vertices", "fill_colors", "stroke_points", "stroke_curves"))
    unchanged = retained.build(flatten_scene(RGroup(children=[shape(circle, A, 1), shape(square, B, 2)])), cache, np.zeros(3))
    assert all(ranges == [] for ranges in unchanged.dirty.values())


def test_garbage_is_repacked() -> None:
    retained, cache = RetainedVectors(), RNodeDataCache()
    for _ in range(1, 400):  # every frame a new shape replaces the last: its ranges become garbage
        retained.build(flatten_scene(RGroup(children=[shape(circle, new_uid())])), cache, np.zeros(3))
    assert retained.arenas["stroke_points"].used < 4096 + 400 and len(retained.slots) == 1
