"""Render state identity: every snapshot of a scene node shares its uid."""

import pytest
from manimx.robjects import RShape2D, RSpotLight, new_uid


def test_uid_can_be_given_and_is_unique_otherwise() -> None:
    uid = new_uid()
    first, second = RShape2D(uid=uid), RShape2D(uid=uid)

    assert first.uid == second.uid == uid
    assert RShape2D().uid != RShape2D().uid
    assert RShape2D().uid > uid  # same counter: no collisions


def test_subclasses_with_their_own_checks_take_a_uid() -> None:
    uid = new_uid()
    light = RSpotLight(uid=uid, inner_cone_angle=0.1, outer_cone_angle=0.2)

    assert light.uid == uid
    with pytest.raises(ValueError, match="cone"):
        RSpotLight(inner_cone_angle=0.3, outer_cone_angle=0.2)
