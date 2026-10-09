"""Frame-time benchmarks for typical and stress-test scenes.

Scenes cover the cost dimensions separately: many 2D objects (batching),
complex glyph outlines (tessellation), long 3D curves (stroke expansion),
lit meshes with shadow-casting lights, and temporal effects (motion blur).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from manimx import (
    Curve3D,
    DirectionalLight,
    Fill,
    Group,
    PbrMaterial,
    PointLight,
    Shape2D,
    Shape3D,
    SpotLight,
    Stroke,
    Transform,
    box_mesh,
    sphere_mesh,
)
from manimx.examples.manim_logo import M_GLYPH_POINTS, manim_logo
from manimx.examples.manim_logo import config as logo_config
from manimx.objects.geometry import circle
from manimx.renderer import Bloom, Camera, FrameComposer, MotionBlur, RenderConfig
from manimx.renderer.tessellation import tessellate_fill
from manimx.utils.color import BLUE, RED, YELLOW

FULL_HD = {"pixel_width": 1920, "pixel_height": 1080}
PERSPECTIVE = Camera.look_at((12, -16, 10), (0, 0, 1), field_of_view=math.radians(45), near=0.5, far=200)


def many_circles(count: int) -> Group:
    rng = np.random.default_rng(1)
    return Group(
        children=[
            circle(
                radius=0.15,
                fill=Fill(RED if i % 2 else BLUE, opacity=0.6 if i % 3 == 0 else 1.0),
                stroke=Stroke("#FFFFFF", width=2),
                transform=Transform(position=(rng.uniform(-7, 7), rng.uniform(-4, 4), 0.0)),
            )
            for i in range(count)
        ],
    )


def glyph_grid(columns: int, rows: int) -> Group:
    points = np.load(M_GLYPH_POINTS)
    return Group(
        children=[
            Shape2D(points=points, fill=Fill("#FFFFFF", opacity=1.0), stroke=Stroke(width=0), transform=Transform(position=(-6.5 + c * 0.7, 3.5 - r * 0.8, 0.0), scale=(1.2, 1.2, 1.0)))
            for c in range(columns)
            for r in range(rows)
        ],
    )


def helices(count: int) -> Group:
    return Group(
        children=[
            Curve3D.parametric(
                lambda t, i=i: (math.cos(t) + (i % 10 - 4.5) * 2.2, math.sin(t) + (i // 10 - 2) * 2.2, 0.1 * t),
                (0.0, 8 * math.pi),
                128,
                stroke=Stroke([RED, YELLOW, BLUE], width=[4, 12, 4]),
            )
            for i in range(count)
        ],
    )


def lit_scene(spheres: int, shadows: bool = True) -> Group:
    rng = np.random.default_rng(2)
    material = PbrMaterial("#b0b0b0", roughness=0.5)
    balls = [
        Shape3D(mesh=sphere_mesh(0.4), material=material, transform=Transform(position=(rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(0.4, 3))))
        for i in range(spheres)
    ]
    floor = Shape3D(mesh=box_mesh(16, 16, 0.2), material=PbrMaterial("#d0d0d0", roughness=0.9), transform=Transform(position=(0, 0, -0.1)))
    lights = [
        DirectionalLight.aimed((0.6, 1.0, -1.2), cast_shadows=shadows),
        SpotLight.aimed((-6, -4, 8), (0, 0, 0), intensity=80, outer_cone_angle=0.6, size=0.3, cast_shadows=shadows),
        PointLight(intensity=8, transform=Transform(position=(2, -2, 2)), cast_shadows=shadows),
    ]
    return Group(children=[floor, *balls, *lights])


# --- 2D ---------------------------------------------------------------------------


def test_manim_logo_1080p(measure, renderer) -> None:
    scene, config = manim_logo(), logo_config()
    measure(lambda: renderer.render(scene, config))


@pytest.mark.parametrize("count", [1000, 10000])
def test_many_circles_1080p(measure, renderer, count: int) -> None:
    scene, config = many_circles(count), RenderConfig(**FULL_HD)
    measure(lambda: renderer.render(scene, config), rounds=10)


def test_glyphs_warm_cache_1080p(measure, renderer) -> None:
    scene, config = glyph_grid(20, 10), RenderConfig(**FULL_HD)
    measure(lambda: renderer.render(scene, config))


def test_glyph_tessellation_cpu(benchmark) -> None:
    """CPU cost of triangulating one complex glyph (what a cache miss costs)."""
    points = np.load(M_GLYPH_POINTS)
    benchmark(tessellate_fill, points)


# --- 3D ---------------------------------------------------------------------------


def test_curves_3d_1080p(measure, renderer) -> None:
    scene = helices(50)
    config = RenderConfig(**FULL_HD, camera=PERSPECTIVE)
    measure(lambda: renderer.render(scene, config))


@pytest.mark.parametrize("shadows", [False, True], ids=["no-shadows", "shadows"])
def test_lit_meshes_1080p(measure, renderer, shadows: bool) -> None:
    scene = lit_scene(200, shadows)
    config = RenderConfig(**FULL_HD, camera=PERSPECTIVE)
    measure(lambda: renderer.render(scene, config))


@pytest.mark.parametrize("resolution", [1024, 4096])
def test_shadow_resolution_1080p(measure, renderer, resolution: int) -> None:
    scene = lit_scene(200)
    config = RenderConfig(**FULL_HD, camera=PERSPECTIVE, shadow_resolution=resolution)
    measure(lambda: renderer.render(scene, config))


# --- composer -------------------------------------------------------------------


def test_motion_blur_16_samples_1080p(measure, renderer) -> None:
    scene = many_circles(1000)

    def scene_at(t: float) -> Group:
        scene.transform = Transform(position=(t * 2.0, 0.0, 0.0))
        return scene

    composer = FrameComposer(renderer, [MotionBlur(samples=16, shutter=1 / 60)])
    config = RenderConfig(**FULL_HD, time=0.5)
    measure(lambda: composer.compose(scene_at, config), rounds=5)


def test_bloom_1080p(measure, renderer) -> None:
    scene = lit_scene(50, shadows=False)
    composer = FrameComposer(renderer, [Bloom(threshold=0.8, intensity=0.2)])
    config = RenderConfig(**FULL_HD, camera=PERSPECTIVE)
    measure(lambda: composer.compose(scene, config))
