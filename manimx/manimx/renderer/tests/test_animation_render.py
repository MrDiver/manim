"""The animation engine through the renderer: scenes evaluated at a time,
rendered, and composed with motion blur."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from manimx.examples.animated_logo import AnimatedLogo, config
from manimx.examples.manim_logo import manim_logo
from manimx.renderer import FrameComposer, MotionBlur, WGPURenderer
from manimx.renderer.webgpu import vector_batch


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


def test_moving_and_turning_never_retriangulates(renderer: WGPURenderer, monkeypatch: pytest.MonkeyPatch) -> None:
    """While shapes only move and rotate (2.16 s to 3.66 s, after the last
    lagged fade-in ends), their fills come from the renderer's cache:
    snapshots keep uid and version tokens."""
    prepared = AnimatedLogo().prepare()
    calls = []
    original = vector_batch.tessellate_fill
    monkeypatch.setattr(vector_batch, "tessellate_fill", lambda *args: calls.append(1) or original(*args))

    renderer.render(prepared.state_at(2.2), config(160, 90))
    first = len(calls)
    for t in np.linspace(2.25, 3.6, 10):
        renderer.render(prepared.state_at(t), config(160, 90))

    assert first <= 3  # at most once per shape
    assert len(calls) == first


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
