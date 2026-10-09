"""The animation engine through the renderer: scenes evaluated at a time,
rendered, and composed with motion blur."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from manimx.examples.animated_logo import AnimatedLogo, config
from manimx.examples.manim_logo import manim_logo
from manimx.renderer import FrameComposer, MotionBlur, WGPURenderer
from manimx.renderer.webgpu import retained


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def test_the_animation_ends_exactly_as_the_static_logo(renderer: WGPURenderer) -> None:
    prepared = AnimatedLogo().prepare()

    animated = renderer.render(prepared.state_at(prepared.duration), config(480, 270))
    static = renderer.render(manim_logo(), config(480, 270))

    np.testing.assert_array_equal(animated, static)


def test_each_shape_is_triangulated_once_for_the_whole_animation(
    renderer: WGPURenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fades, moves, turns and pulses never re-triangulate: snapshots keep
    uid and version tokens, and fill meshes depend on the points alone
    (colors and opacity are looked up per point on the GPU)."""
    prepared = AnimatedLogo().prepare()
    calls = []
    original = retained.tessellate_fill
    monkeypatch.setattr(retained, "tessellate_fill", lambda *args: calls.append(1) or original(*args))

    for t in np.linspace(0.0, prepared.duration, 40):
        renderer.render(prepared.state_at(t), config(160, 90))

    assert len(calls) == 4  # circle, square, triangle and the M


def test_motion_blur_through_prepared_scene(renderer: WGPURenderer) -> None:
    prepared = AnimatedLogo().prepare()
    composer = FrameComposer(renderer, [MotionBlur(samples=8, shutter=0.2)])

    blurred = composer.compose(prepared.state_at, _at(config(160, 90), 2.0))
    sharp = renderer.render(prepared.state_at(2.0), config(160, 90))

    assert not np.array_equal(blurred, sharp)  # moving shapes smear
    still = composer.compose(prepared.state_at, _at(config(160, 90), 5.9))  # nothing moves at the end
    np.testing.assert_allclose(still.astype(int), renderer.render(prepared.state_at(5.9), config(160, 90)).astype(int), atol=2)


def _at(settings, time: float):
    return dataclasses.replace(settings, time=time)


def test_navigation_badge_properties(renderer: WGPURenderer) -> None:
    from manimx.examples.navigation import Navigation
    from manimx.examples.navigation import config as navigation_config

    prepared = Navigation().prepare()
    settings = navigation_config(320, 180)

    def frame(t: float) -> np.ndarray:
        return renderer.render(prepared.state_at(t), settings).astype(int)

    background = frame(0.0)
    assert len(np.unique(background.reshape(-1, 4), axis=0)) == 1  # empty before it grows
    assert np.array_equal(frame(prepared.duration), background)  # and empty again at the end

    # The streets keep moving on their own (a derived signal of time).
    inside = (slice(70, 110), slice(140, 180))
    assert not np.array_equal(frame(2.0)[inside], frame(2.3)[inside])

    # Outside the circle, at the badge's height, only background: the map
    # (much larger than the circle) is clipped by the mask.
    outside = frame(2.0)[90, 20:60]
    assert np.all(outside == background[90, 20:60])


def test_views_render_through_their_camera_and_blur_camera_moves(renderer: WGPURenderer) -> None:
    from manimx.animation import RIGHT, Camera, Dot, Scene, Timeline, View
    from manimx.animation.rate_functions import linear
    from manimx.renderer import RenderConfig

    camera = Camera(name="camera")

    class Pan(Scene):
        def construct(self):
            self.add(Dot(radius=0.3, name="dot"))
            self.play(camera.animate.shift(RIGHT * 2), run_time=1, rate=linear)

    timeline = Timeline()
    timeline.add(Pan().prepare())
    view = timeline.view(View(timeline.worlds[0], camera))
    settings = RenderConfig(pixel_width=160, pixel_height=90)

    def dot_columns(image: np.ndarray) -> np.ndarray:
        return np.nonzero(image[45, :, 0] > 128)[0]

    start = dot_columns(renderer.render(view(0.0).root, settings))
    composer = FrameComposer(renderer)
    end = dot_columns(composer.compose(view, _at(settings, 1.0)))
    assert start.mean() == pytest.approx(80, abs=1)  # centered
    assert end.mean() == pytest.approx(80 - 2 * 160 / (8 * 16 / 9), abs=1)  # the camera moved right by 2 units

    blurred = FrameComposer(renderer, [MotionBlur(samples=8, shutter=0.5)]).compose(view, _at(settings, 0.5))
    partial = (blurred[45, :, 0] > 10) & (blurred[45, :, 0] < 245)
    assert partial.sum() > 4  # the dot smears along the camera move
