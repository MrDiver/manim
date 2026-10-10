"""The core's renderer (CoreFramePipeline) against the Python renderer
(FramePipeline with a FrameComposer): the same frames, pixel for pixel.
"""

from __future__ import annotations

import numpy as np
import pytest
from manimx.animation import Timeline
from manimx.animation.plan import PlannedTimeline, ScenePlan
from manimx.drivers.core_pipeline import CoreFramePipeline
from manimx.drivers.pipeline import FramePipeline
from manimx.examples.navigation import Navigation
from manimx.renderer import (
    Bloom,
    MotionBlur,
    RenderConfig,
    RenderService,
    ShaderEffect,
    ToneMap,
)
from manimx.renderer.effects import ToneMapOperator
from test_core_render import EveryNode
from test_plan import EveryAnimation

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


@pytest.fixture(scope="module")
def service() -> RenderService:
    try:
        return RenderService.shared()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def render_both(service, frames, times, effects=(), config=CONFIG):
    python = FramePipeline(service.composer(effects), config, 2)
    core = CoreFramePipeline(effects, config, 2)
    expected = [python.submit(index, float(t), frames) for index, t in enumerate(times)]
    actual = [core.submit(index, float(t), frames) for index, t in enumerate(times)]
    python.flush()
    core.flush()
    return [future.result() for future in expected], [future.result() for future in actual]


def assert_same(expected, actual) -> None:
    for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
        difference = np.abs(a.astype(int) - b.astype(int))
        assert difference.max() == 0, f"frame {index}: {np.count_nonzero(difference.max(axis=2))} pixels differ by up to {difference.max()}"


@pytest.mark.parametrize("scene_class", [EveryAnimation, EveryNode, Navigation])
def test_scenes_render_the_same_on_the_core(service: RenderService, scene_class) -> None:
    scene = scene_class().prepare()
    plan = ScenePlan(scene)
    times = np.linspace(0.0, scene.duration, 7)
    assert_same(*render_both(service, lambda t: (plan.frame_at(t),), times))


def test_effects_render_the_same_on_the_core(service: RenderService) -> None:
    scene = EveryNode().prepare()
    plan = ScenePlan(scene)
    effects = (MotionBlur(samples=3), Bloom(intensity=0.5), DARKEN_LIT, VIGNETTE, ToneMap(ToneMapOperator.ACES))
    assert_same(*render_both(service, lambda t: (plan.frame_at(t),), [0.3, 1.2, 2.4], effects))


def test_a_timeline_with_several_views_and_empty_frames(service: RenderService) -> None:
    timeline = Timeline()
    timeline.add(EveryNode().prepare(), start=0.0)
    timeline.add(Navigation().prepare(), start=0.0, layer=1)  # two views at once, the second on top
    planned = PlannedTimeline(timeline)
    times = [0.5, 2.0, timeline.duration + 1.0]  # the last is after every clip: background only
    assert len(planned.frame_at(times[0])) == 2
    assert planned.frame_at(times[-1]) == ()
    assert_same(*render_both(service, planned.frame_at, times))


def test_frames_like_the_one_before_are_held_and_shader_effects_never(service: RenderService) -> None:
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


def test_only_frames_of_the_core_are_accepted(service: RenderService) -> None:
    scene = EveryNode().prepare()
    pipeline = CoreFramePipeline((), CONFIG, 1)
    with pytest.raises(TypeError, match="PlannedFrame"):
        pipeline.submit(0, 0.5, scene.frame_at)
