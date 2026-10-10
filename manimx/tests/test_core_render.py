"""Frames of the core (PlannedFrame, packed in Rust) against frames of the
reference path (RNode trees): every pixel must be the same.
"""

from __future__ import annotations

import dataclasses
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
)
from manimx.animation.plan import PlannedTimeline, ScenePlan
from manimx.examples.animated_logo import AnimatedLogo
from manimx.examples.bouncing import Bouncing
from manimx.examples.navigation import Navigation
from manimx.project import Project
from manimx.renderer import MotionBlur, RenderConfig, RenderService
from manimx.robjects.planned_frame import PlannedFrame
from manimx.userland import Circle, FadeIn, Rotate, Square
from manimx.userland.geometry import box_mesh, sphere_mesh
from test_plan import EveryAnimation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = RenderConfig(pixel_width=240, pixel_height=136)


@pytest.fixture(scope="module")
def service() -> RenderService:
    try:
        return RenderService.shared()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


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


def assert_same_pixels(service: RenderService, reference, planned, times, config: RenderConfig = CONFIG, effects=()) -> None:
    composer = service.composer(effects)
    for t in times:
        frame_config = dataclasses.replace(config, time=float(t))
        expected = composer.compose(reference, frame_config)
        actual = composer.compose(planned, frame_config)
        difference = np.abs(expected.astype(int) - actual.astype(int))
        assert difference.max() == 0, f"t={t}: {np.count_nonzero(difference.max(axis=2))} pixels differ by up to {difference.max()}"


@pytest.mark.parametrize("scene_class", [EveryAnimation, EveryNode, AnimatedLogo, Bouncing, Navigation])
def test_scenes_render_the_same_through_the_core(service: RenderService, scene_class: type[Scene]) -> None:
    scene = scene_class().prepare()
    plan = ScenePlan(scene)
    assert plan.graph is not None
    times = np.linspace(0.0, scene.duration, 9)
    assert_same_pixels(service, scene.frame_at, lambda t: (plan.frame_at(t),), times)


def test_motion_blur_renders_the_same_through_the_core(service: RenderService) -> None:
    scene = EveryNode().prepare()
    plan = ScenePlan(scene)
    assert_same_pixels(service, scene.frame_at, lambda t: (plan.frame_at(t),), [0.4, 1.3], effects=(MotionBlur(samples=4),))


@pytest.mark.parametrize("project", ["example-project", "tests/projects/benchmark"])
def test_projects_render_the_same_through_the_core(service: RenderService, project: str) -> None:
    timeline = Project.load(ROOT / project).build_timeline()
    planned = PlannedTimeline(timeline)
    assert planned.duration == timeline.duration  # the timeline's own attributes pass through
    assert_same_pixels(service, timeline.frame_at, planned.frame_at, np.linspace(0.0, timeline.duration, 7))


def test_a_scene_with_a_node_the_core_cannot_draw_falls_back_to_snapshots(service: RenderService) -> None:
    class Custom(Square):
        def _build_snapshot(self):  # a node class with its own render state
            return super()._build_snapshot()

    class WithCustom(Scene):
        def construct(self) -> None:
            self.add(Custom(fill_opacity=1.0))
            self.wait()

    scene = WithCustom().prepare()
    assert ScenePlan(scene).graph is None

    timeline = Project.load(ROOT / "example-project").build_timeline()
    clip = timeline.clips[0]
    clip.scene = scene
    (view,) = PlannedTimeline(timeline).frame_at(clip.start + 0.5)
    assert not isinstance(view.root, PlannedFrame)


def test_unchanged_planned_frames_are_held(service: RenderService) -> None:
    scene = EveryNode().prepare()
    plan = ScenePlan(scene)
    composer = service.composer()
    still = scene.duration - 0.1  # in the final wait: nothing moves
    composer.compose(lambda t: (plan.frame_at(t),), dataclasses.replace(CONFIG, time=still))
    composer.compose(lambda t: (plan.frame_at(t),), dataclasses.replace(CONFIG, time=still + 0.05))
    assert composer.held
    composer.compose(lambda t: (plan.frame_at(t),), dataclasses.replace(CONFIG, time=0.5))
    assert not composer.held


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


def test_clipped_casters_cast_the_shadow_of_what_their_masks_let_through(service: RenderService) -> None:
    scene = ClippedShadow().prepare()
    plan = ScenePlan(scene)
    assert_same_pixels(service, scene.frame_at, lambda t: (plan.frame_at(t),), [0.5])

    image = service.composer().compose(scene.frame_at, dataclasses.replace(CONFIG, time=0.5)).astype(float)
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
