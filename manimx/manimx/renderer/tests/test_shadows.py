"""Shadows: view assignment, toggles at every level, softness."""

import math

import numpy as np
import pytest

from manimx import RDirectionalLight, RGroup, PbrMaterial, RPointLight, RShape3D, RSpotLight, Transform, box_mesh
from manimx.renderer import Camera, RNodeDataCache, RenderConfig, WGPURenderer
from manimx.renderer.scene_graph import flatten_scene
from manimx.renderer.webgpu.mesh_batch import MeshBatch
from manimx.renderer.webgpu.shadow_views import NO_SHADOW, build_shadow_views


def _views(root: RGroup, **config: object):
    settings = RenderConfig(**config)
    scene = flatten_scene(root)
    batch = MeshBatch.build(scene, RNodeDataCache(), settings)
    views = build_shadow_views(scene, batch.light_items, batch.lights, RNodeDataCache(), settings)
    return views, batch.lights


def _lights() -> list:
    return [
        RDirectionalLight.aimed((0, 0, -1)),
        RSpotLight.aimed((0, 0, 5), (0, 0, 0)),
        RPointLight(transform=Transform(position=(0, 0, 3))),
    ]


def test_each_light_kind_gets_its_shadow_views() -> None:
    views, lights = _views(RGroup(children=[RShape3D(), *_lights()]))

    assert len(views) == 1 + 1 + 6  # directional, spot, six cube faces
    assert list(lights["shadow_view"]) == [0, 1, 2]
    assert list(views.views["params"][:, 3]) == [0.0] + [1.0] * 7  # orthographic, then perspective


def test_shadow_toggles() -> None:
    lights = _lights()
    lights[1].cast_shadows = False
    views, rows = _views(RGroup(children=[RShape3D(), *lights]))
    assert list(rows["shadow_view"]) == [0, NO_SHADOW, 1]

    views, rows = _views(RGroup(children=[RShape3D(), *_lights()]), shadows=False)
    assert len(views) == 0 and set(rows["shadow_view"]) == {NO_SHADOW}

    views, rows = _views(RGroup(children=_lights()))  # nothing to shadow
    assert len(views) == 0


def test_directional_view_covers_the_scene() -> None:
    root = RGroup(children=[RShape3D(mesh=box_mesh(4, 4, 4)), RDirectionalLight.aimed((1, 1, -1))])
    views, _ = _views(root)

    view_projection = views.views["view_projection"][0].T
    for corner in [(-2, -2, -2), (2, 2, 2), (2, -2, 2)]:
        ndc = view_projection @ (*corner, 1.0)
        assert np.all(np.abs(ndc[:2]) <= 1.0) and 0.0 <= ndc[2] <= 1.0


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def _scene(**options: object) -> RGroup:
    floor = RShape3D(mesh=box_mesh(10, 10, 0.2), material=PbrMaterial("#FFFFFF", roughness=1.0), transform=Transform(position=(0, 0, -0.1)))
    blocker = RShape3D(mesh=box_mesh(2, 2, 0.2), material=PbrMaterial("#FFFFFF"), transform=Transform(position=(0, 0, 2)), **options)
    sun = RDirectionalLight.aimed((0, 0, -1), intensity=3.0)
    return RGroup(children=[floor, blocker, sun])


CAMERA = Camera.look_at((0, -0.01, 12), (0, 0, 0), up=(0, 1, 0), field_of_view=math.radians(60))


def _brightness(renderer: WGPURenderer, root: RGroup, x: float, **config: object) -> float:
    settings = RenderConfig(pixel_width=160, pixel_height=90, camera=CAMERA, ambient_intensity=0.0, **config)
    image = renderer.render(root, settings)
    # The camera looks straight down; world x maps to image columns.
    scale = 90 / (2 * 12 * math.tan(math.radians(30)))
    return float(image[45, int(80 + x * scale), :3].astype(int).mean())


def test_blocker_shadows_the_floor(renderer: WGPURenderer) -> None:
    lit_floor = _brightness(renderer, _scene(), 4.0)

    assert lit_floor > 100
    # Seen from above, the blocker hides its own shadow; look just beside it
    # with a light that comes in at an angle instead.
    tilted = _scene()
    tilted.children[2] = RDirectionalLight.aimed((1, 0, -1), intensity=3.0)
    assert _brightness(renderer, tilted, 2.5) < lit_floor / 3
    assert _brightness(renderer, tilted, 2.5, shadows=False) > lit_floor / 2
    no_caster = _scene(cast_shadows=False)
    no_caster.children[2] = tilted.children[2]
    assert _brightness(renderer, no_caster, 2.5) > lit_floor / 2


def test_softness_widens_the_penumbra(renderer: WGPURenderer) -> None:
    def edge_profile(size: float) -> np.ndarray:
        root = _scene()
        root.children[2] = RDirectionalLight.aimed((1, 0, -1), intensity=3.0, size=size)
        return np.array([_brightness(renderer, root, x) for x in np.linspace(1.0, 4.5, 15)])

    hard, soft = edge_profile(0.0), edge_profile(0.4)
    partial = lambda profile: int(np.sum((profile > 0.15 * profile.max()) & (profile < 0.85 * profile.max())))  # noqa: E731
    assert partial(soft) > partial(hard)
