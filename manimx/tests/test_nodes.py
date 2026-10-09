"""Node and its transform mixins: which parts a node has, and the errors
for parts it lacks."""

from __future__ import annotations

import pytest
from manimx.animation import (
    UP,
    Camera,
    Circle,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    Node,
    PositionMixin,
    Rotate,
    RotationMixin,
    ScalableMixin,
    ShrinkToCenter,
    Square,
)
from manimx.objects import RShape2D


class Bare(Node):
    """A node with no transform parts (like a light without position)."""

    def snapshot(self, alphas=()):
        return None


def test_transform_parts_come_from_the_mixins() -> None:
    square = Square()
    assert isinstance(square, PositionMixin) and isinstance(square, RotationMixin) and isinstance(square, ScalableMixin)
    square.shift(UP).scale_by(2.0)

    snapshot = square.snapshot()
    assert isinstance(snapshot, RShape2D) and snapshot.uid == square.uid
    assert snapshot.transform.position == (0.0, 1.0, 0.0) and snapshot.transform.scale == (2.0, 2.0, 2.0)
    assert len(list(square.signals())) == 2 + 3 + 1 + 3 + 6  # node, position, rotation, scale, shape


def test_a_missing_part_is_the_identity_and_has_no_signals() -> None:
    camera = Camera()
    assert not isinstance(camera, ScalableMixin)
    assert camera._transform().scale == (1.0, 1.0, 1.0)
    bare = Bare()
    assert bare._transform().position == (0.0, 0.0, 0.0)
    assert list(bare.signals()) == [bare.alpha, bare.present]


def test_missing_parts_explain_themselves() -> None:
    camera, bare = Camera(name="cam"), Bare(name="bare")
    with pytest.raises(AttributeError, match=r"Camera has no scale, so it has no 'scale_by' \(ScalableMixin provides it\)"):
        camera.scale_by(2.0)
    with pytest.raises(AttributeError, match=r"Bare has no position, so it has no 'shift'"):
        bare.animate.shift(UP)
    with pytest.raises(AttributeError, match=r"Bare has no rotation, so it has no 'rotation'"):
        bare.rotation  # noqa: B018
    with pytest.raises(AttributeError, match=r"'Bare' object has no attribute 'nothing'"):
        bare.nothing  # noqa: B018
    with pytest.raises(AttributeError, match=r"'Circle' object has no attribute 'set_alpha2'"):
        Circle().animate.set_alpha2(0.5)
    with pytest.raises(AttributeError, match=r"Circle.get_center cannot be animated"):
        Circle().animate.get_center()


def test_animations_name_the_part_they_need() -> None:
    camera, bare = Camera(name="cam"), Bare(name="bare")
    with pytest.raises(TypeError, match=r"Indicate needs a scale, but Camera 'cam' has none \(ScalableMixin provides it\)"):
        Indicate(camera)
    with pytest.raises(TypeError, match="GrowFromCenter needs a scale"):
        GrowFromCenter(camera)
    with pytest.raises(TypeError, match="ShrinkToCenter needs a scale"):
        ShrinkToCenter(camera)
    with pytest.raises(TypeError, match=r"FadeIn\(shift=...\) needs a position, but Bare 'bare' has none"):
        FadeIn(bare, shift=UP)
    with pytest.raises(TypeError, match=r"FadeOut\(shift=...\) needs a position"):
        FadeOut(bare, shift=UP)
    with pytest.raises(TypeError, match="Rotate needs a rotation"):
        Rotate(bare, 1.0)
    with pytest.raises(TypeError, match=r"Circle.follow\(\) needs a position, but Bare 'bare' has none"):
        Circle().follow(bare)

    FadeIn(bare), FadeOut(bare)  # without shift, only alpha is needed


def test_a_plain_node_has_no_snapshot() -> None:
    with pytest.raises(NotImplementedError):
        Node().snapshot()
