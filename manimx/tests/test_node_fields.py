"""Every field of a render class has its signal on the scene class that
snapshots to it, and the list-valued style signals (COLORS, SCALARS)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from manimx.animation import COLORS, SCALARS, ClippingGroup, Curve3D, DirectionalLight, PointLight, Scene, Shape2D, Shape3D, SpotLight
from manimx.userland import Circle, Square
from manimx.userland.rate_functions import linear
from manimx.robjects import (
    CapStyle,
    Fill,
    JointType,
    PbrMaterial,
    RClippingGroup,
    RCurve3D,
    RDirectionalLight,
    RPointLight,
    RShape2D,
    RShape3D,
    RSpotLight,
    Stroke,
    Texture,
)

# Render fields that are not signals: identity, structure and placement.
NOT_SIGNALS = {"uid", "transform", "visible", "children", "mask", "content"}


def render_fields(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")} - NOT_SIGNALS


@pytest.mark.parametrize(
    ("scene_class", "render_class", "names"),
    [
        (Shape2D, RShape2D, {"fill": ("fill_color", "fill_opacity"), "stroke": ("stroke_color", "stroke_opacity", "stroke_width", "stroke_joint", "stroke_cap")}),
        (Curve3D, RCurve3D, {"stroke": ("stroke_color", "stroke_opacity", "stroke_width", "stroke_joint", "stroke_cap")}),
        (Shape3D, RShape3D, {"material": tuple(f.name for f in dataclasses.fields(PbrMaterial))}),
        (DirectionalLight, RDirectionalLight, {}),
        (PointLight, RPointLight, {}),
        (SpotLight, RSpotLight, {}),
        (ClippingGroup, RClippingGroup, {}),
    ],
)
def test_every_render_field_has_a_signal(scene_class, render_class, names) -> None:
    """A field of a style value (Fill, Stroke, PbrMaterial) maps to one
    signal per field of that value."""
    node = scene_class(Square(), Circle()) if scene_class is ClippingGroup else scene_class()
    signals = {signal.name.split(".", 1)[1] for signal in node.signals()}
    for field in render_fields(render_class):
        for name in names.get(field, (field,)):
            assert name in signals, f"{scene_class.__name__} has no signal for {render_class.__name__}.{field} ({name})"
    # and the value types are covered field by field
    assert {f.name for f in dataclasses.fields(Fill)} == {"color", "opacity"}
    assert {f.name for f in dataclasses.fields(Stroke)} == {"color", "opacity", "width", "joint", "cap"}


def test_shape2d_carries_gradients_widths_joints_and_caps() -> None:
    shape = Shape2D(
        Square().points.value,
        fill_color=["#FF0000", "#0000FF"],
        fill_opacity=[0.2, 0.8],
        stroke_color="#00FF00",
        stroke_width=[2.0, 8.0, 2.0],
        stroke_joint=[JointType.MITER, JointType.ROUND],
        stroke_cap=CapStyle.ROUND,
    )
    shape.set_alpha(0.5)
    snapshot = shape.snapshot()
    np.testing.assert_allclose(snapshot.fill.color, [[1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(snapshot.fill.opacity, [0.2, 0.8])
    assert snapshot.alpha == 0.5
    np.testing.assert_allclose(snapshot.stroke.color, [[0, 1, 0]])
    np.testing.assert_allclose(snapshot.stroke.width, [2.0, 8.0, 2.0])
    assert snapshot.stroke.joint == (JointType.MITER, JointType.ROUND) and snapshot.stroke.cap is CapStyle.ROUND

    shape.set_stroke(joint=JointType.BEVEL, cap=CapStyle.BUTT, width=-1.0)
    snapshot = shape.snapshot()
    assert snapshot.stroke.joint == (JointType.BEVEL,) and snapshot.stroke.cap is CapStyle.BUTT
    assert snapshot.stroke.width[0] == 0.0  # clamped
    with pytest.raises(TypeError, match="joint must be a JointType"):
        Shape2D(stroke_joint="round")
    with pytest.raises(TypeError, match="cap must be a CapStyle"):
        Shape2D(stroke_cap="round")
    with pytest.raises(ValueError, match="at least one color"):
        Shape2D(color=[])


def test_single_values_match_the_old_single_value_snapshot() -> None:
    """One color and one number take the Fill/Stroke fast path and give the
    same values as plain arguments."""
    shape = Circle(color="#58C4DD", fill_opacity=0.5, stroke_width=3.0)
    snapshot = shape.snapshot()
    assert snapshot.fill == Fill("#58C4DD", opacity=0.5)
    assert snapshot.stroke == Stroke("#58C4DD", opacity=1.0, width=3.0)


def test_curve3d_carries_the_stroke_fields() -> None:
    curve = Curve3D(np.zeros((4, 3)), color=["#FF0000", "#00FF00"], stroke_width=[0.1, 0.3], stroke_cap=CapStyle.SQUARE)
    curve.set_stroke(joint=[JointType.ROUND], opacity=[1.0, 0.5])
    snapshot = curve.snapshot()
    assert snapshot.stroke.color.shape == (2, 3) and snapshot.stroke.cap is CapStyle.SQUARE
    np.testing.assert_allclose(snapshot.stroke.opacity, [1.0, 0.5])
    assert snapshot.stroke.joint == (JointType.ROUND,)


def test_gradients_interpolate_across_lengths() -> None:
    two = COLORS.validate([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    three = COLORS.validate([(0.0, 1.0, 0.0)] * 3)
    middle = COLORS.interpolate(two, three, 0.5)
    np.testing.assert_allclose(middle, [[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])
    assert not middle.flags.writeable
    np.testing.assert_allclose(COLORS.combine(two, COLORS.scale((0.5, 0.5, 0.5), 1.0)), [[1, 0.5, 0.5], [0.5, 0.5, 1]])
    assert COLORS.equal(two, COLORS.validate(two)) and not COLORS.equal(two, three)
    with pytest.raises(ValueError, match="one RGB color or a list"):
        COLORS.validate([(1.0, 0.0)])

    widths = SCALARS.validate([2.0, 4.0])
    np.testing.assert_allclose(SCALARS.interpolate(widths, SCALARS.validate(6.0), 0.5), [4.0, 5.0])
    np.testing.assert_allclose(SCALARS.combine(widths, SCALARS.scale(1.0, 0.5)), [2.5, 4.5])
    assert SCALARS.equal(widths, SCALARS.validate([2.0, 4.0])) and not SCALARS.equal(widths, SCALARS.validate(2.0))
    with pytest.raises(ValueError, match="one number or a list"):
        SCALARS.validate([])


def test_a_gradient_animates_into_another_gradient() -> None:
    square = Square(name="square", fill_opacity=1.0, color="#FF0000")

    class Gradient(Scene):
        def construct(self) -> None:
            self.add(square)
            self.play(square.animate.set_fill(["#FF0000", "#0000FF", "#00FF00"]), rate=linear)

    prepared = Gradient().prepare()
    np.testing.assert_allclose(prepared.state_at(0.5).children[0].fill.color, [[1, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def test_shape3d_carries_every_material_field() -> None:
    texture = Texture(np.zeros((2, 2, 4), np.uint8))
    shape = Shape3D(material=PbrMaterial(normal_scale=0.5, occlusion_strength=0.25, emissive_texture=texture))
    material = shape.snapshot().material
    assert (material.normal_scale, material.occlusion_strength, material.emissive_texture) == (0.5, 0.25, texture)

    other = Texture(np.full((2, 2, 4), 255, np.uint8))
    before = shape.snapshot().version("material")
    shape.set_material(normal_scale=2.0, occlusion_strength=1.5).set_textures(base_color=other, emissive=None)
    snapshot = shape.snapshot()
    assert snapshot.material.normal_scale == 2.0 and snapshot.material.occlusion_strength == 1.0  # clamped
    assert snapshot.material.base_color_texture is other and snapshot.material.emissive_texture is None
    assert snapshot.version("material") != before
    with pytest.raises(TypeError, match="Shape3D has no glow_texture"):
        shape.set_textures(glow=other)
    with pytest.raises(TypeError, match="a texture must be a Texture or None"):
        shape.set_textures(normal="bricks.png")


def test_clipping_group_one_sided_is_a_signal() -> None:
    group = ClippingGroup(Square(), Circle(), one_sided=True)
    assert group.snapshot().one_sided
    group.set_one_sided(False)
    assert not group.snapshot().one_sided

