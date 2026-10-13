"""The core's evaluator against the Python reference: every signal of a
plan must have the value PreparedScene.evaluate() gives it, at any time
and in any order.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import quaternion
from manimx.animation import Group, Scene, Simulation, derive
from manimx.animation.plan import ScenePlan
from manimx.animation.scene import PreparedScene
from manimx.project import Project
from manimx.userland import (
    PI,
    Create,
    Morph,
    Uncreate,
    Write,
    RIGHT,
    UP,
    Circle,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    Rotate,
    ShrinkToCenter,
    Square,
)
from manimx.userland.rate_functions import there_and_back
from manimx.utils.color.manim_colors import BLUE, RED

ROOT = Path(__file__).resolve().parents[1]


def assert_same(expected: object, actual: object, where: str) -> None:
    if isinstance(expected, quaternion.quaternion):
        assert np.allclose(expected.components, actual.components, rtol=1e-12, atol=1e-12), where
    elif isinstance(expected, (float, int, np.floating)) and not isinstance(expected, bool):
        assert math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12), (where, expected, actual)
    elif isinstance(expected, np.ndarray):
        assert expected.shape == np.shape(actual), where
        assert np.allclose(expected, actual, rtol=1e-12, atol=1e-12), where
    elif isinstance(expected, tuple) and all(isinstance(c, float) for c in expected):
        assert np.allclose(expected, actual, rtol=1e-12, atol=1e-12), where
    else:
        assert expected is actual or expected == actual, (where, expected, actual)


def assert_plan_matches(scene: PreparedScene, times: list[float]) -> None:
    plan = ScenePlan(scene)
    for t in times:
        scene.evaluate(t)
        plan.evaluate(t)
        for signal in plan.signals:
            assert_same(signal.value, plan.value(signal), f"{signal.name} at {t}")


def shuffled_times(duration: float, count: int = 60) -> list[float]:
    times = list(np.linspace(-0.5, duration + 0.5, count))
    np.random.default_rng(1).shuffle(times)
    return [float(t) for t in times]


class EveryAnimation(Scene):
    def construct(self) -> None:
        square = Square(color=BLUE, fill_opacity=0.5)
        circle = Circle(color=RED).shift(RIGHT * 2)
        label = Circle(radius=0.2).follow(square, offset=(0, 1, 0))
        spin = Group(Square(side_length=0.5), name="spin")
        spin.rotation = derive(lambda t: quaternion.from_rotation_vector((0, 0, 0.3 * t)), self.time.scene_time, kind=spin.rotation.kind)
        self.add(label, spin)
        self.play(FadeIn(square, shift=UP), GrowFromCenter(circle))
        self.play(square.animate.shift(RIGHT).rotate(PI / 3).set_fill(RED, 0.8), Indicate(circle), run_time=1.5)
        self.play(Rotate(square, 2 * PI, axis=(0, 1, 1)), circle.animate.scale_by(2).set_stroke(width=[1, 8]), rate=there_and_back)
        square.move_to((1, 1, 0))
        self.wait_for("go")
        self.play(square.animate.set_scale(0.5), circle.animate.set_color([RED, BLUE, RED]))

        def fall(dt: float) -> None:
            circle.position.y.set(circle.position.y.value - dt)

        self.play(Simulation(fall, duration=1.0))
        self.play(FadeOut(square, shift=RIGHT), ShrinkToCenter(circle))
        star = Group(Square(side_length=1.0, fill_opacity=0.8), Circle(radius=0.3).shift(RIGHT), name="star")
        self.play(Create(star))
        self.play(Morph(star, Group(Circle(radius=1.2, color=RED), name="ring").shift(UP)))
        words = Group(*(Square(side_length=0.4).shift(RIGHT * i) for i in range(5)), name="words")
        self.play(Write(words))
        self.play(Uncreate(words))


def test_every_built_in_animation() -> None:
    scene = EveryAnimation().prepare()
    assert_plan_matches(scene, shuffled_times(scene.duration, 120))


@pytest.mark.parametrize("project", ["tests/projects/example-project", "tests/projects/benchmark", "tests/projects/light-show"])
def test_projects(project: str) -> None:
    timeline = Project.load(ROOT / project).build_timeline()
    for clip in timeline.clips:
        assert_plan_matches(clip.scene, shuffled_times(clip.scene.duration, 25))


def test_animations_without_data_are_called_once_per_evaluation() -> None:
    from dataclasses import dataclass

    from manimx.animation import Animation, Change, Target
    from manimx.userland import Square

    @dataclass(eq=False)
    class Swap(Animation):
        """Moves x to where y started and y to where x started: needs both
        start values in one call."""

        node: Square
        calls: list = None

        def targets(self):
            x, y, _ = self.node.position.axes
            return [Target(x, Change.ABSOLUTE), Target(y, Change.ABSOLUTE)]

        def contributions(self, alpha, start_values):
            self.calls.append(alpha)
            x, y = start_values
            return [x + (y - x) * alpha, y + (x - y) * alpha]

    calls = []

    class Swapping(Scene):
        def construct(self):
            square = Square().move_to((1.0, 3.0, 0.0))
            self.add(square)
            self.play(Swap(square, calls), run_time=2.0)

    scene = Swapping().prepare()
    plan = ScenePlan(scene)
    calls.clear()
    plan.evaluate(1.0)
    assert calls == [pytest.approx(scene.placements[-1].progress(1.0))]  # one call for both targets
    x, y, _ = scene.roots[0].position.axes
    plan.evaluate(2.0)
    assert (plan.value(x), plan.value(y)) == pytest.approx((3.0, 1.0))
    assert_plan_matches(scene, shuffled_times(scene.duration))


def test_versions_change_with_values() -> None:
    scene = EveryAnimation().prepare()
    plan = ScenePlan(scene)
    square = scene.roots[2]
    plan.evaluate(0.0)
    before = plan.version(square.alpha)
    plan.evaluate(0.0)
    assert plan.version(square.alpha) == before  # same time, same value
    plan.evaluate(0.5)
    assert plan.version(square.alpha) > before
