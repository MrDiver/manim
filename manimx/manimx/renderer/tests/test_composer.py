"""Frame composer: temporal and image effects on top of the renderer."""

import numpy as np
import pytest
from manimx.userland.geometry import rectangle_points, sphere_mesh
from manimx.renderer import (
    Bloom,
    FrameComposer,
    MotionBlur,
    RenderConfig,
    ShaderEffect,
    ToneMap,
    ToneMapOperator,
    WGPURenderer,
)
from manimx.robjects import RShape2D
from manimx.utils.color import RED

from manimx import PbrMaterial
from manimx.robjects import Fill, RGroup, RShape3D, Stroke, Transform

SIZE = {"pixel_width": 160, "pixel_height": 90}  # 11.25 pixels per world unit


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def column(image: np.ndarray, x: float) -> np.ndarray:
    """The pixel column at world x through the image's middle row."""
    return image[45, int(80 + x * 11.25), :3].astype(int)


def test_motion_blur_sample_times_are_centered() -> None:
    times = MotionBlur(samples=4, shutter=0.4).sample_times(1.0)

    assert [round(t, 6) for t, _ in times] == [0.85, 0.95, 1.05, 1.15]
    assert sum(w for _, w in times) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        MotionBlur(samples=0)


def test_composer_validates_effects(renderer: WGPURenderer) -> None:
    with pytest.raises(ValueError, match="at most one ToneMap"):
        FrameComposer(renderer, [ToneMap(), ToneMap()])
    with pytest.raises(TypeError, match="not an effect"):
        FrameComposer(renderer, ["bloom"])
    with pytest.raises(ValueError, match="stage"):
        ShaderEffect("", stage="later")


def test_motion_blur_averages_sub_frames(renderer: WGPURenderer) -> None:
    def scene_at(time: float) -> RGroup:
        # A white bar that is only visible during the second half of the shutter.
        bar = RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill("#FFFFFF", opacity=1.0), stroke=Stroke(width=0.0))
        bar.visible = time > 0.0
        return RGroup(children=[bar])

    composer = FrameComposer(renderer, [MotionBlur(samples=8, shutter=1.0)])
    image = composer.compose(scene_at, RenderConfig(**SIZE))

    # Half the exposure white, half black: 0.5 in linear light, 188 in sRGB.
    assert abs(column(image, 0.0)[0] - 188) <= 2
    still = FrameComposer(renderer).compose(scene_at, RenderConfig(**SIZE))
    assert column(still, 0.0)[0] == 0  # without blur, only t = 0 is rendered


def test_motion_blur_skips_static_scenes(renderer: WGPURenderer) -> None:
    calls = []
    original = renderer.render_layers

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    renderer.render_layers = counting
    try:
        FrameComposer(renderer, [MotionBlur(samples=8)]).compose(RGroup(), RenderConfig(**SIZE))
    finally:
        del renderer.render_layers
    assert len(calls) == 1


def _glowing_sphere() -> RGroup:
    sphere = RShape3D(mesh=sphere_mesh(1.0), material=PbrMaterial("#000000", emissive="#FFFFFF", emissive_strength=8.0))
    return RGroup(children=[sphere])


def test_bloom_spreads_bright_light(renderer: WGPURenderer) -> None:
    config = RenderConfig(**SIZE, ambient_intensity=0.0, default_lights=False)
    plain = FrameComposer(renderer).compose(_glowing_sphere(), config)
    glowing = FrameComposer(renderer, [Bloom(intensity=0.5)]).compose(_glowing_sphere(), config)

    assert column(plain, 2.0).sum() == 0  # outside the sphere
    assert column(glowing, 2.0).sum() > 30  # the glow reaches beyond it


def test_tone_map_affects_lit_layer_but_not_2d_colors(renderer: WGPURenderer) -> None:
    config = RenderConfig(**SIZE, ambient_intensity=0.0, default_lights=False)
    flat = RShape2D(points=rectangle_points(1.0, 1.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0.0), transform=Transform(position=(-4.0, 0.0, 0.0)))
    scene = RGroup(children=[flat, *_glowing_sphere().children])

    neutral = FrameComposer(renderer, [ToneMap(ToneMapOperator.PBR_NEUTRAL)]).compose(scene, config)
    aces = FrameComposer(renderer, [ToneMap(ToneMapOperator.ACES)]).compose(scene, config)
    clamped = FrameComposer(renderer, [ToneMap(ToneMapOperator.NONE)]).compose(scene, config)

    assert tuple(column(neutral, -4.0)) == tuple(column(aces, -4.0)) == tuple(RED.to_int_rgb())
    assert tuple(column(clamped, 0.0)) == (255, 255, 255)  # 8x white clips
    for operator in ToneMapOperator:
        image = FrameComposer(renderer, [ToneMap(operator)]).compose(scene, config)
        assert column(image, 0.0).min() > 150  # every curve keeps it bright


def test_shader_effects_on_both_stages(renderer: WGPURenderer) -> None:
    invert = ShaderEffect(
        """
        fn effect(uv: vec2<f32>, pixel: vec2<i32>) -> vec4<f32> {
            let color = textureLoad(source, pixel, 0);
            return vec4<f32>(vec3<f32>(value(0u)) - color.rgb, 1.0);
        }
        """,
        values=(1.0,),
    )
    darken_lit = ShaderEffect(
        "fn effect(uv: vec2<f32>, pixel: vec2<i32>) -> vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, textureLoad(source, pixel, 0).a); }",
        stage="lit",
    )
    config = RenderConfig(**SIZE, background_color="#000000")

    inverted = FrameComposer(renderer, [invert]).compose(RGroup(), config)
    assert tuple(column(inverted, 0.0)) == (255, 255, 255)
    dark = FrameComposer(renderer, [darken_lit]).compose(_glowing_sphere(), RenderConfig(**SIZE))
    assert column(dark, 0.0).sum() == 0
