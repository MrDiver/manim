"""Node and its transform mixins: which parts a node has, and the errors
for parts it lacks."""

from __future__ import annotations

import pytest
from manimx.animation import Camera, ClippingGroup, Group, Node, PositionMixin, RotationMixin, ScalableMixin
from manimx.userland import UP, Circle, FadeIn, FadeOut, GrowFromCenter, Indicate, Rotate, ShrinkToCenter, Square
from manimx.robjects import RShape2D


class Bare(Node):
    """A node with no transform parts (like a light without position)."""

    def snapshot(self):
        return None


def test_transform_parts_come_from_the_mixins() -> None:
    square = Square()
    assert isinstance(square, PositionMixin) and isinstance(square, RotationMixin) and isinstance(square, ScalableMixin)
    square.shift(UP).scale_by(2.0)

    snapshot = square.snapshot()
    assert isinstance(snapshot, RShape2D) and snapshot.uid == square.uid
    assert snapshot.transform.position == (0.0, 1.0, 0.0) and snapshot.transform.scale == (2.0, 2.0, 2.0)
    assert len(list(square.signals())) == 2 + 3 + 1 + 3 + 9  # node, position, rotation, scale, shape


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


# --- names and tags ---------------------------------------------------------------


def model() -> Group:
    wheels = [Circle(name=f"wheel-{i}", tags={"wheel", "rubber"}) for i in range(2)]
    body = Square(name="body", tags="metal")
    window = ClippingGroup(Square(name="frame"), Circle(name="glass", tags=["glass"]), name="window")
    return Group(body, Group(*wheels, name="axle", tags=["wheel"]), window, Circle(name="body"), name="car")


def test_nodes_have_names_and_tags() -> None:
    assert Square(name="s", tags=["a", "b"]).tags == frozenset({"a", "b"})
    assert Square(tags="metal").tags == frozenset({"metal"})  # one tag, not its letters
    assert Camera(name="cam", tags={"main"}).tags == frozenset({"main"})
    assert Square().name.startswith("square-") and Square().tags == frozenset()
    with pytest.raises(TypeError, match="tags must be strings"):
        Square(tags=[1])


def test_find_returns_the_one_node_with_a_name() -> None:
    car = model()
    assert car.find("wheel-1").name == "wheel-1"
    assert car.find("glass") is car.find("window").content  # through clipping groups
    assert car.find("window").find("frame") is car.find("window").mask
    with pytest.raises(LookupError, match=r"Group 'car' has no node named 'wheel-3'; did you mean 'wheel-1', 'wheel-0'\?"):
        car.find("wheel-3")
    with pytest.raises(LookupError, match=r"Group 'car' has no node named 'zzz'$"):
        car.find("zzz")
    with pytest.raises(LookupError, match=r"has 2 nodes named 'body'; use find_all"):
        car.find("body")
    with pytest.raises(LookupError, match="no node named 'car'"):  # the group itself is not searched
        car.find("car")


def test_find_all_matches_names_and_all_tags_in_tree_order() -> None:
    car = model()
    assert [n.name for n in car.find_all(tags="wheel")] == ["axle", "wheel-0", "wheel-1"]
    assert [n.name for n in car.find_all(tags=["wheel", "rubber"])] == ["wheel-0", "wheel-1"]
    assert [n.name for n in car.find_all(name="body")] == ["body", "body"]
    assert car.find_all(name="body", tags="metal") == [car.find_all(name="body")[0]]
    assert car.find_all(tags="nothing") == []
    assert len(car.find_all()) == 8  # every node below the group
    assert [n.name for n in car.find("window").find_all(tags="glass")] == ["glass"]
