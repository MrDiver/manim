"""Surface shaders: WGSL that changes a Shape3D's surface per pixel
(SurfaceShader, mesh.wgsl's surface_shader.wgsl hook)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from manimx.animation import Scene, Shape3D
from manimx.animation.plan import planned_frames
from manimx.drivers.core_pipeline import CoreFramePipeline
from manimx.renderer import RenderConfig, RenderService
from manimx.userland.geometry import box_mesh

from manimx import SurfaceShader

CONFIG = RenderConfig(pixel_width=96, pixel_height=54, background_color="#000000", default_lights=True)

RED = SurfaceShader("""
fn shade(surface: Surface, in: SurfaceInput) -> Surface {
    var out = surface;
    out.base_color = vec3<f32>(0.0);
    out.emissive = vec3<f32>(1.0, 0.0, 0.0);
    return out;
}
""")

# Emissive green that switches on after half a second of frame time.
BLINK = SurfaceShader("""
fn shade(surface: Surface, in: SurfaceInput) -> Surface {
    var out = surface;
    out.base_color = vec3<f32>(0.0);
    out.emissive = vec3<f32>(0.0, step(0.5, frame.time), 0.0);
    return out;
}
""")


@pytest.fixture(scope="module")
def service() -> RenderService:
    try:
        return RenderService.shared()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def _scene(shader: SurfaceShader | None, switch_to: SurfaceShader | None = None) -> Scene:
    class Shaded(Scene):
        def construct(self) -> None:
            box = Shape3D(box_mesh(3.0, 3.0, 3.0), "box", color="#ffffff", surface_shader=shader)
            self.add(box)
            self.wait(1.0)
            if switch_to is not None:
                self.play(box.animate.set_surface_shader(switch_to), run_time=0.0)
                self.wait(1.0)

    return Shaded()


def _render(service, frames, time: float) -> np.ndarray:
    return service.render(frames.frame_at(time), dataclasses.replace(CONFIG, time=time))


def test_surface_shader_arguments() -> None:
    with pytest.raises(ValueError, match="fn shade"):
        SurfaceShader("fn other() {}")
    with pytest.raises(TypeError, match="SurfaceShader"):
        Shape3D(box_mesh(), surface_shader="red")  # type: ignore[arg-type]
    assert RED.uid != BLINK.uid
    assert "return shade(surface, in);" in RED.wgsl()


def test_the_shader_changes_the_surface(service) -> None:
    plain = _render(service, _scene(None).prepare(), 0.0)
    red = _render(service, _scene(RED).prepare(), 0.0)
    center = red[27, 48]
    assert center[0] > 200 and center[1] < 60 and center[2] < 60  # tone mapping desaturates a little
    assert not np.array_equal(plain[27, 48], center)


def test_frames_with_a_shader_follow_the_time_on_every_path(service) -> None:
    prepared = _scene(BLINK).prepare()
    planned = planned_frames(prepared)
    pipeline = CoreFramePipeline((), CONFIG)
    for time, lit in ((0.2, False), (0.8, True)):
        reference = _render(service, prepared, time)
        assert bool(reference[27, 48, 1] > 150) is lit  # never held from the frame before
        np.testing.assert_array_equal(_render(service, planned, time), reference)
        future = pipeline.submit(0, time, planned.frame_at(time))
        pipeline.flush()
        np.testing.assert_array_equal(future.result(), reference)


def test_the_shader_is_a_signal(service) -> None:
    prepared = _scene(None, switch_to=RED).prepare()
    before, after = _render(service, prepared, 0.5), _render(service, prepared, 1.5)
    assert after[27, 48, 0] > 200 and after[27, 48, 1] < 60
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(_render(service, planned_frames(prepared), 1.5), after)


def test_wgsl_errors_are_reported(service) -> None:
    broken = SurfaceShader("fn shade(surface: Surface, in: SurfaceInput) -> Surface { return nothing; }")
    planned = planned_frames(_scene(broken).prepare())
    pipeline = CoreFramePipeline((), CONFIG)
    with pytest.raises(RuntimeError, match="mesh.wgsl"):
        pipeline.submit(0, 0.0, planned.frame_at(0.0))
