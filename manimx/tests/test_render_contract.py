"""The render-state contract between the animation engine and renderers.

Section 9 of docs/animation_engine_design.md: snapshots of a scene object
share its uid, and their versioned fields carry tokens derived from
signals, so renderer caches survive any number of snapshots.
"""

import numpy as np
import pytest

from manimx import Fill, Group, Shape2D, SpotLight, Stroke
from manimx.animation import PATH, RootSignal, Vector3Signal, derive, version_token
from manimx.objects import new_uid
from manimx.objects.geometry import arc_points
from manimx.renderer import MobjectDataCache
from manimx.renderer.scene_graph import flatten_scene
from manimx.renderer.webgpu.vector_batch import VectorBatch


def test_uid_can_be_given_and_is_unique_otherwise() -> None:
    uid = new_uid()
    first, second = Shape2D("a", uid=uid), Shape2D("a", uid=uid)

    assert first.uid == second.uid == uid
    assert Shape2D("b").uid != Shape2D("b").uid
    assert Shape2D("c").uid > uid  # same counter: no collisions


def test_explicit_versions_replace_assignment_counts() -> None:
    shape = Shape2D("s", points=arc_points(), versions={"points": ("p", 7)})

    assert shape.version("points") == (("p", 7),)
    assert Shape2D("t", points=arc_points(), versions={"points": ("p", 7)}).version("points") == shape.version("points")
    with pytest.raises(ValueError, match=r"no versioned fields \['transform'\]"):
        Shape2D("u", versions={"transform": 1})


def test_assigning_after_explicit_versions_gives_new_tokens() -> None:
    shape = Shape2D("s", versions={"points": ("p", 7)})
    seen = {shape.version("points")}

    for _ in range(3):
        shape.points = arc_points()
        assert shape.version("points") not in seen
        seen.add(shape.version("points"))


def test_subclasses_with_their_own_checks_accept_versions() -> None:
    light = SpotLight("spot", uid=5, inner_cone_angle=0.1, outer_cone_angle=0.2)

    assert light.uid == 5
    with pytest.raises(ValueError, match="cone"):
        SpotLight("bad", inner_cone_angle=0.3, outer_cone_angle=0.2)


def test_version_token_follows_signal_changes() -> None:
    radius = RootSignal(1.0)
    position = Vector3Signal()
    points = derive(lambda r: arc_points(r), radius, kind=PATH)

    token = version_token(points, position)
    assert version_token(points, position) == token
    position.x.set(2.0)
    assert version_token(points, position) != token
    assert version_token(points) == version_token(points)
    assert version_token(radius) != version_token(position.x)  # ids differ


def test_snapshots_from_unchanged_signals_share_cached_triangulations() -> None:
    """Rebuilding the render object every frame must not re-tessellate."""
    uid = new_uid()
    radius = RootSignal(1.0)
    x = RootSignal(0.0)
    points = derive(lambda r: arc_points(r), radius, kind=PATH)
    cache = MobjectDataCache()

    def snapshot() -> Group:
        from manimx import Transform

        shape = Shape2D(
            "circle",
            uid=uid,
            points=points.value,
            fill=Fill(opacity=1.0),
            stroke=Stroke(width=0),
            transform=Transform(position=(x.value, 0.0, 0.0)),
            versions={"points": version_token(points), "fill": "fill-0", "stroke": "stroke-0"},
        )
        VectorBatch.build(flatten_scene(Group("root", children=[shape])), cache, np.zeros(3))
        return shape

    first = snapshot()
    mesh = cache.get(uid, "fill-mesh", first.version("points"))
    assert mesh is not None

    x.set(3.0)  # moving never touches the triangulation
    moved = snapshot()
    assert cache.get(uid, "fill-mesh", moved.version("points")) is mesh

    radius.set(2.0)  # changing the shape does
    grown = snapshot()
    assert cache.get(uid, "fill-mesh", grown.version("points")) is not mesh
    assert len(cache) == 1  # the stale entry was replaced, not kept
