"""The tracks under each clip in the preview page (manimx.web.tracks)."""

from __future__ import annotations

import pytest
from manimx.animation import ClippingGroup, Group, Scene, Timeline
from manimx.userland import RIGHT, UP, Circle, FadeIn, Rotate, Square
from manimx.web.tracks import animation_label, clip_nodes


class Moves(Scene):
    def construct(self) -> None:
        circle = Circle(name="circle")
        square = Square(name="square")
        group = Group(circle, square, name="pair")
        mask = Square(name="mask")
        clipped = ClippingGroup(mask, Circle(name="inner"), name="clipped")
        self.play(FadeIn(group), run_time=1.0)
        self.play(circle.animate.shift(RIGHT).rotate(1.0), run_time=2.0)
        self.add(clipped)
        self.play(Rotate(clipped, 1.0), self.camera.animate.shift(UP), run_time=1.0)


@pytest.fixture
def clip():
    timeline = Timeline()
    clip = timeline.add(Moves().prepare(), name="moves")
    clip.start = 0.0
    return clip


def _find(nodes, name):
    for node in nodes:
        if node["name"] == name:
            return node
        if (found := _find(node["children"], name)) is not None:
            return found
    return None


def test_nodes_signals_and_animations(clip) -> None:
    nodes = clip_nodes(clip)
    assert [node["name"] for node in nodes] == ["camera", "pair", "clipped"]
    pair = nodes[1]
    assert pair["type"] == "Group"
    assert [child["name"] for child in pair["children"]] == ["circle", "square"]
    circle = _find(nodes, "circle")
    signals = {signal["name"]: signal["animations"] for signal in circle["signals"]}
    assert [a["name"] for a in signals["position.x"]] == ["animate.shift.rotate"]
    assert signals["position.x"][0]["start"] == pytest.approx(1.0)
    assert signals["position.x"][0]["end"] == pytest.approx(3.0)
    assert "rotation" in signals and "alpha" not in signals  # FadeIn of the group changes the group
    assert [a["name"] for a in {s["name"]: s["animations"] for s in pair["signals"]}["alpha"]] == ["FadeIn"]
    # A clipping group shows its mask and content.
    assert [child["name"] for child in _find(nodes, "clipped")["children"]] == ["mask", "inner"]
    camera = {signal["name"]: signal["animations"] for signal in nodes[0]["signals"]}
    assert camera["position.y"][0]["name"] == "animate.shift"


def test_times_follow_the_clip_and_present_is_add(clip) -> None:
    clip.timeline.clips[0].start = 2.0
    clip._changed()
    circle = _find(clip_nodes(clip), "circle")
    position = next(signal for signal in circle["signals"] if signal["name"] == "position.x")
    assert position["animations"][0]["start"] == pytest.approx(3.0)
    pair = _find(clip_nodes(clip), "pair")
    present = next(signal for signal in pair["signals"] if signal["name"] == "present")
    assert [a["name"] for a in present["animations"]] == ["add"]


def test_skipped_animations_are_left_out(clip) -> None:
    clip.cut(0.0, 0.5)
    circle = _find(clip_nodes(clip), "circle")
    assert all(signal["name"] != "position.x" for signal in circle["signals"])
    assert clip.global_time(10.0) is None


def test_labels() -> None:
    assert animation_label(object()) == "object"
