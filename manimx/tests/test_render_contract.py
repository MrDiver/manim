"""The render-state contract between the animation engine and renderers.

Section 9 of docs/animation_engine_design.md: snapshots of a scene object
share its uid, and their versioned fields carry tokens derived from
signals, so renderer caches survive any number of snapshots.
"""

import pytest
from manimx.animation import PATH, RootSignal, Vector3Signal, derive, version_token
from manimx.userland.geometry import arc_points
from manimx.robjects import new_uid

from manimx.robjects import RShape2D, RSpotLight


def test_uid_can_be_given_and_is_unique_otherwise() -> None:
    uid = new_uid()
    first, second = RShape2D(uid=uid), RShape2D(uid=uid)

    assert first.uid == second.uid == uid
    assert RShape2D().uid != RShape2D().uid
    assert RShape2D().uid > uid  # same counter: no collisions


def test_explicit_versions_replace_assignment_counts() -> None:
    shape = RShape2D(points=arc_points(), versions={"points": ("p", 7)})

    assert shape.version("points") == (("p", 7),)
    assert RShape2D(points=arc_points(), versions={"points": ("p", 7)}).version("points") == shape.version("points")
    with pytest.raises(ValueError, match=r"no versioned fields \['transform'\]"):
        RShape2D(versions={"transform": 1})


def test_assigning_after_explicit_versions_gives_new_tokens() -> None:
    shape = RShape2D(versions={"points": ("p", 7)})
    seen = {shape.version("points")}

    for _ in range(3):
        shape.points = arc_points()
        assert shape.version("points") not in seen
        seen.add(shape.version("points"))


def test_subclasses_with_their_own_checks_accept_versions() -> None:
    uid = new_uid()
    light = RSpotLight(uid=uid, inner_cone_angle=0.1, outer_cone_angle=0.2)

    assert light.uid == uid
    with pytest.raises(ValueError, match="cone"):
        RSpotLight(inner_cone_angle=0.3, outer_cone_angle=0.2)


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


