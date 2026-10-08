"""Simulations: user step functions baked into keyframes (section 7)."""

import numpy as np
import pytest

from manimx.animation import RIGHT, UP, Dot, Polyline, Scene, Simulation, Timeline
from manimx.animation.rate_functions import linear


def prepared(construct):
    class Built(Scene):
        def construct(self):
            construct(self)

    return Built().prepare()


def x_of(prepared_scene, t, name):
    return {c.id: c for c in prepared_scene.state_at(t).children}[name].transform.position


def falling(ball, calls=None):
    velocity = np.array([1.0, 0.0, 0.0])

    def step(dt):
        if calls is not None:
            calls.append(dt)
        velocity[1] -= 10.0 * dt
        ball.shift(velocity * dt)

    return step


def test_a_step_function_is_baked_into_keyframes() -> None:
    ball = Dot(name="ball")
    calls = []
    simulation = Simulation(falling(ball, calls), dt=1 / 1000, duration=1.0, keyframe_rate=50)

    def construct(scene):
        scene.add(ball)
        scene.play(simulation)
        scene.wait(1)

    scene = prepared(construct)
    assert scene.duration == pytest.approx(2.0)  # the cursor moved by the simulated time
    assert len(simulation.times) == 51 and simulation.times[-1] == pytest.approx(1.0)
    assert sum(calls) == pytest.approx(1.0) and len(calls) == 1000
    for t in (0.0, 0.37, 1.0, 1.5):
        s = min(t, 1.0)
        np.testing.assert_allclose(x_of(scene, t, "ball")[:2], (s, -5 * s * s), atol=0.02)

    count = len(calls)
    for t in (0.9, 0.1, 0.5):
        x_of(scene, t, "ball")
    assert len(calls) == count  # evaluation never re-simulates


def test_animated_inputs_drive_the_simulation_and_reactions_follow_it() -> None:
    paddle, ball, label = Dot(name="paddle"), Dot(name="ball"), Dot(name="label")
    label.follow(ball, UP)

    def push(dt):  # the ball is dragged towards the paddle
        gap = paddle.position.x.value - ball.position.x.value
        ball.shift(RIGHT * gap * min(1.0, 20 * dt))

    def construct(scene):
        scene.add(paddle, ball, label)
        scene.play(Simulation(push, dt=1 / 500, duration=2), paddle.animate.shift(RIGHT * 4), rate=linear, run_time=2)

    scene = prepared(construct)
    assert x_of(scene, 1.0, "paddle")[0] == pytest.approx(2.0)
    assert x_of(scene, 1.0, "ball")[0] == pytest.approx(2.0, abs=0.15)  # lags slightly behind
    np.testing.assert_allclose(x_of(scene, 1.0, "label"), x_of(scene, 1.0, "ball") + np.array(UP))


def test_simulations_continue_from_and_into_recorded_animations() -> None:
    ball = Dot(name="ball")

    def construct(scene):
        scene.add(ball)
        scene.play(ball.animate.shift(RIGHT * 3), run_time=1)
        scene.play(Simulation(lambda dt: ball.shift(UP * dt), duration=1), run_time=5)  # run_time does not apply
        assert ball.get_center()[1] == pytest.approx(1.0)
        scene.play(ball.animate.shift(RIGHT), run_time=1)

    scene = prepared(construct)
    assert scene.duration == pytest.approx(3.0)
    np.testing.assert_allclose(x_of(scene, 0.0, "ball"), (0, 0, 0), atol=1e-12)  # the past is untouched
    np.testing.assert_allclose(x_of(scene, 1.5, "ball"), (3, 0.5, 0), atol=1e-9)
    np.testing.assert_allclose(x_of(scene, 3.0, "ball"), (4, 1, 0), atol=1e-9)


def test_until_ends_a_simulation_and_emits_become_events() -> None:
    ball = Dot(name="ball")
    simulation = Simulation(lambda dt: None, until=lambda: False, max_time=0.1)

    def bounce(dt):
        falling_step(dt)
        if ball.position.y.value < -1 and not bounced.events:
            bounced.emit("hit")

    falling_step = falling(ball)
    bounced = Simulation(bounce, dt=1 / 1000, until=lambda: ball.position.y.value < -2)

    def construct(scene):
        scene.add(ball)
        scene.wait(1)
        scene.play(bounced)

    scene = prepared(construct)
    hit = bounced.events[0][1]
    assert hit == pytest.approx(np.sqrt(0.2), abs=2e-3) and scene.duration == pytest.approx(1 + np.sqrt(0.4), abs=2e-3)
    timeline = Timeline()
    timeline.add(scene, start=10.0)
    assert timeline.resolve()["hit"] == pytest.approx(11.0 + hit)

    def endless(scene):
        scene.add(Dot())
        scene.play(simulation)

    with pytest.raises(RuntimeError, match="max_time"):
        prepared(endless)


def test_steps_see_simulated_and_scene_time() -> None:
    seen = []
    holder = Dot(name="holder")

    def construct(scene):
        scene.add(holder)
        scene.wait(2)
        simulation = Simulation(lambda dt: seen.append((simulation.time, scene.time.scene_time.value)), dt=0.25, duration=1)
        scene.play(simulation)

    prepared(construct)
    assert seen == [(0.0, 2.0), (0.25, 2.25), (0.5, 2.5), (0.75, 2.75)]


def test_paths_that_change_point_count_switch_between_keyframes() -> None:
    line = Polyline([(0, 0, 0), (1, 0, 0)], name="line")

    def grow(dt):
        end = line.points.value[-1]
        line.points.set(np.vstack([line.points.value, [end, end, end + 0.1, end + 0.1]]))  # one more curve

    def construct(scene):
        scene.add(line)
        scene.play(Simulation(grow, dt=0.5, duration=1, keyframe_rate=2))

    scene = prepared(construct)
    counts = [len({c.id: c for c in scene.state_at(t).children}["line"].points) for t in (0.0, 0.25, 0.5, 1.0)]
    assert counts[0] < counts[2] < counts[3] and counts[1] == counts[0]


def test_simulation_arguments_and_single_use() -> None:
    with pytest.raises(ValueError, match="positive"):
        Simulation(lambda dt: None, dt=0, duration=1)
    with pytest.raises(ValueError, match="exactly one"):
        Simulation(lambda dt: None)
    with pytest.raises(ValueError, match="exactly one"):
        Simulation(lambda dt: None, duration=1, until=lambda: True)
    simulation = Simulation(lambda dt: None, duration=1)
    with pytest.raises(RuntimeError, match="baked"):
        simulation.targets()

    def twice(scene):
        scene.add(Dot())
        scene.play(simulation)
        scene.play(simulation)

    with pytest.raises(RuntimeError, match="once"):
        prepared(twice)
