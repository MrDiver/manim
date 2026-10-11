"""Frames rendered by the core's renderer: every kind of node, effects,
example scenes and projects against reference images, and how frames are
held and checked.

Reference images live in tests/references/; regenerate them with
``pytest --update-references`` and review the changed PNGs before
committing. On a mismatch, ``*.actual.png`` and ``*.diff.png`` are
written next to the reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from manimx.animation import (
    ClippingGroup,
    Curve3D,
    DirectionalLight,
    Group,
    Particles,
    PointLight,
    Scene,
    Shape3D,
    SpotLight,
    Timeline,
)
from manimx.animation.plan import PlannedTimeline, ScenePlan, planned_frames
from manimx.drivers import CoreFramePipeline
from manimx.examples.animated_logo import AnimatedLogo
from manimx.examples.bouncing import Bouncing
from manimx.examples.navigation import Navigation
from manimx.project import Project
from manimx.renderer import (
    Bloom,
    MotionBlur,
    RenderConfig,
    ShaderEffect,
    ToneMap,
    ToneMapOperator,
)
from manimx.testing import assert_image_matches
from manimx.userland import Circle, FadeIn, Rotate, Square
from manimx.userland.geometry import box_mesh, sphere_mesh
from test_plan import EveryAnimation

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = Path(__file__).parent / "references"
CONFIG = RenderConfig(pixel_width=240, pixel_height=136)
VIGNETTE = ShaderEffect(
    """
    fn effect(uv: vec2<f32>, pixel: vec2<i32>) -> vec4<f32> {
        let color = textureLoad(source, pixel, 0);
        return vec4<f32>(color.rgb * (1.0 - value(0u) * dot(uv - 0.5, uv - 0.5)), color.a);
    }
    """,
    values=(1.5,),
)
DARKEN_LIT = ShaderEffect(
    "fn effect(uv: vec2<f32>, pixel: vec2<i32>) -> vec4<f32> { let c = textureLoad(source, pixel, 0); return vec4<f32>(c.rgb * 0.5, c.a); }",
    stage="lit",
)


class EveryNode(Scene):
    """Meshes, particles, a 3D curve, every light and a clipping group."""

    def construct(self) -> None:
        self.camera.move_to((0.0, -9.0, 5.0)).look_at((0.0, 0.0, 0.0))
        self.camera.set_field_of_view(0.8)
        box = Shape3D(box_mesh(2.0, 2.0, 2.0), "box", color="#58C4DD", roughness=0.4)
        ball = Shape3D(sphere_mesh(0.8, 24, 12), "ball", color="#E07A5F", metallic=0.6).move_to((2.5, 0.0, 0.0))
        sparks = Particles(4, box_mesh(0.3, 0.3, 0.3), "sparks", color="#ffffff", emissive="#ffffff")
        sparks.set_positions([[-3.0, 0.0, 1.0], [-3.0, 0.0, -1.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]])
        sparks.set_colors([[1.0, 0.2, 0.2, 1.0], [0.2, 1.0, 0.2, 1.0], [0.2, 0.2, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5]])
        after = Shape3D(box_mesh(0.5, 0.5, 0.5), "after-particles", color="#83C167").move_to((0.0, 0.0, 2.0))
        curve = Curve3D(np.array([[-3, -1, 0], [-1, 1, 1], [1, -1, 1], [3, 1, 0]], dtype=float), "curve", color="#FFFF00")
        window = ClippingGroup(Circle(radius=1.5, fill_opacity=1.0), Group(Square(side_length=4, color="#FC6255", fill_opacity=0.7)), "window")
        window.move_to((0.0, 2.0, 0.0))
        sun = DirectionalLight("sun", direction=(-0.3, 0.5, -1.0))
        lamp = PointLight("lamp", position=(0.0, -2.0, 3.0), color="#ffddaa", intensity=8.0)
        spot = SpotLight("spot", position=(3.0, -3.0, 4.0), target=(2.5, 0.0, 0.0), intensity=20.0)
        self.add(box, ball, sparks, after, curve, window, sun, lamp, spot)
        self.play(Rotate(box, 1.0, axis=(0, 0, 1)), ball.animate.shift((0.0, 0.0, 1.0)), FadeIn(window))
        self.play(sparks.animate.set_positions([[-3.0, 0.0, 2.0], [-3.0, 0.0, -2.0], [-1.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]), lamp.animate.set_intensity(2.0))
        self.wait(0.5)



class ClippedShadow(Scene):
    """Four circles on the sides of a square window, so four half circles
    show, lit at an angle above a floor."""

    def construct(self) -> None:
        floor = Shape3D(box_mesh(14.0, 8.0, 0.2), "floor", color="#ffffff", roughness=1.0).move_to((0.0, 0.0, -0.1))
        circles = Group(*(Circle(radius=0.5, color="#FC6255", fill_opacity=1.0).move_to(side) for side in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]))
        window = ClippingGroup(Square(side_length=2.0, fill_opacity=1.0), circles, "window")
        window.move_to((0.0, 0.0, 2.0))
        sun = DirectionalLight(direction=(0.5, 0.0, -1.0), intensity=3.0)  # shadows land 1 unit right
        self.add(floor, window, sun)
        self.wait()



def render(frame_at, times, effects=(), config: RenderConfig = CONFIG) -> list[np.ndarray]:
    pipeline = CoreFramePipeline(tuple(effects), config, 2)
    futures = [pipeline.submit(index, float(t), frame_at) for index, t in enumerate(times)]
    pipeline.flush()
    return [future.result() for future in futures]


def check(name: str, frame_at, times, update: bool, effects=()) -> None:
    for t, image in zip(times, render(frame_at, times, effects), strict=True):
        assert_image_matches(image, REFERENCES / f"{name}_{t:.2f}.png", update=update)


@pytest.mark.parametrize("scene_class", [EveryAnimation, EveryNode, AnimatedLogo, Bouncing, Navigation])
def test_scenes_render_like_their_references(scene_class: type[Scene], update_references: bool) -> None:
    scene = scene_class().prepare()
    check(scene_class.__name__, planned_frames(scene).frame_at, scene.duration * np.array([0.25, 0.5, 0.75]), update_references)


def test_effects_render_like_their_reference(update_references: bool) -> None:
    effects = (MotionBlur(samples=3), Bloom(intensity=0.5), DARKEN_LIT, VIGNETTE, ToneMap(ToneMapOperator.ACES))
    check("EveryNode_effects", planned_frames(EveryNode().prepare()).frame_at, [1.2], update_references, effects)


@pytest.mark.parametrize("project", ["example-project", "tests/projects/benchmark"])
def test_projects_render_like_their_references(project: str, update_references: bool) -> None:
    timeline = Project.load(ROOT / project).build_timeline()
    planned = PlannedTimeline(timeline)
    assert planned.duration == timeline.duration  # the timeline's own attributes pass through
    check(Path(project).name, planned.frame_at, timeline.duration * np.array([0.25, 0.5, 0.75]), update_references)


def test_several_views_draw_over_each_other_and_no_view_shows_the_background() -> None:
    timeline = Timeline()
    timeline.add(EveryNode().prepare(), start=0.0)
    timeline.add(Navigation().prepare(), start=0.0, layer=1)  # two views at once, the second on top
    planned = PlannedTimeline(timeline)
    assert len(planned.frame_at(0.5)) == 2
    after = timeline.duration + 1.0  # after every clip
    assert planned.frame_at(after) == ()
    both, empty = render(planned.frame_at, [0.5, after])
    background = np.array(CONFIG.background_rgba) * 255
    assert np.allclose(empty.reshape(-1, 4), background, atol=1)
    alone = render(planned_frames(Navigation().prepare()).frame_at, [0.5])[0]
    assert not np.array_equal(both, alone)  # the lower view shows where the upper one is empty


def test_frames_like_the_one_before_are_held_and_shader_effects_never() -> None:
    scene = EveryNode().prepare()
    plan = ScenePlan(scene)
    still = scene.duration - 0.2  # the final wait: nothing moves
    pipeline = CoreFramePipeline((), CONFIG, 2)
    first = pipeline.submit(0, still, lambda t: (plan.frame_at(t),))
    second = pipeline.submit(1, still + 0.05, lambda t: (plan.frame_at(t),))
    pipeline.flush()
    assert second.result() is first.result()  # held: the same image object
    pulsing = CoreFramePipeline((VIGNETTE,), CONFIG, 2)
    a = pulsing.submit(0, still, lambda t: (plan.frame_at(t),))
    b = pulsing.submit(1, still + 0.05, lambda t: (plan.frame_at(t),))
    pulsing.flush()
    assert b.result() is not a.result()


def test_only_frames_of_the_core_are_accepted() -> None:
    scene = EveryNode().prepare()
    pipeline = CoreFramePipeline((), CONFIG, 1)
    with pytest.raises(TypeError, match="PlannedFrame"):
        pipeline.submit(0, 0.5, scene.frame_at)


def test_a_node_class_that_changes_how_it_is_drawn_is_refused() -> None:
    class Custom(Square):
        def _build_snapshot(self):  # a node class with its own render state
            return super()._build_snapshot()

    class WithCustom(Scene):
        def construct(self) -> None:
            self.add(Custom(fill_opacity=1.0))
            self.wait()

    with pytest.raises(TypeError, match="cannot draw"):
        planned_frames(WithCustom().prepare())


def test_clipped_casters_cast_the_shadow_of_what_their_masks_let_through() -> None:
    (image,) = render(planned_frames(ClippedShadow().prepare()).frame_at, [0.5])
    image = image.astype(float)
    width, height = CONFIG.pixel_width, CONFIG.pixel_height

    def floor(x: float, y: float) -> float:
        frame_width = 8.0 * width / height
        column, row = round((x / frame_width + 0.5) * width), round((0.5 - y / 8.0) * height)
        return float(image[row, column, :3].mean())

    lit = floor(5.0, 2.5)
    # The right circle's inner half (x from 0.5 to 1) shades x from 1.5 to 2;
    # its outer half, cut by the window, casts no shadow (x from 2 to 2.5).
    assert floor(1.75, 0.0) < 0.6 * lit
    assert floor(2.25, 0.0) > 0.9 * lit
    # The top circle's outer half (y above 1) is cut too.
    assert floor(1.0, 1.25) > 0.9 * lit
