"""One-sided masks (portals) and lights scoped to clipping groups."""

import math

import numpy as np
import pytest
from manimx.userland.geometry import box_mesh, rectangle_points
from manimx.renderer import RenderConfig, RNodeDataCache, RenderService
from manimx.renderer.scene_graph import NO_CLIP, flatten_scene
from manimx.renderer.webgpu.mesh_batch import MeshBatch
from manimx.renderer.webgpu.shadow_views import build_shadow_views
from manimx.renderer.webgpu.vector_batch import FLAG_FRONT_ONLY, VectorBatch
from manimx.robjects import RShape2D
from manimx.robjects.transform import rotation_towards
from manimx.utils.color import RED

from manimx import PbrMaterial
from manimx.robjects import Fill, RClippingGroup, RDirectionalLight, RGroup, RShape3D, Stroke, Transform


def _portal(world: RGroup, flip: bool = False, **options: object) -> RClippingGroup:
    transform = Transform().rotated(math.pi, (0, 1, 0)) if flip else Transform()
    mask = RShape2D(points=rectangle_points(4.0, 4.0), fill=Fill(opacity=1.0), stroke=Stroke(width=0), transform=transform)
    return RClippingGroup(mask=mask, content=world, **options)


def _world(*children) -> RGroup:
    return RGroup(children=list(children))


def test_scoped_lights_and_scope_membership() -> None:
    inner_light = RDirectionalLight(transform=Transform(rotation=rotation_towards((0, 0, -1))))
    outer_light = RDirectionalLight(transform=Transform(rotation=rotation_towards((0, 0, -1))))
    nested = RClippingGroup(mask=RShape2D(points=rectangle_points(2.0, 2.0)), content=RShape3D())
    scene = flatten_scene(RGroup(children=[outer_light, _portal(_world(inner_light, RShape3D(), nested))]))

    assert [item.clip for item in scene.lights] == [NO_CLIP, 0]
    assert scene.in_scope(1, 0) and scene.in_scope(0, NO_CLIP) and not scene.in_scope(NO_CLIP, 0)


def test_one_sided_masks_are_flagged() -> None:
    scene = flatten_scene(_portal(_world(RShape2D(points=rectangle_points(2.0, 2.0))), one_sided=True))

    batch = VectorBatch.build(scene, RNodeDataCache(), np.eye(4))

    assert batch.objects["flags"][-1] & FLAG_FRONT_ONLY  # the mask, built last
    assert not batch.objects["flags"][0] & FLAG_FRONT_ONLY


def test_scoped_shadows_use_only_their_scope() -> None:
    inside = RShape3D(mesh=box_mesh(1, 1, 1))
    outside = RShape3D(mesh=box_mesh(1, 1, 1), transform=Transform(position=(50, 0, 0)))
    sun = RDirectionalLight(transform=Transform(rotation=rotation_towards((0, 0, -1))))
    scene = flatten_scene(RGroup(children=[outside, _portal(_world(inside, sun))]))
    config = RenderConfig()
    batch = MeshBatch.build(scene, RNodeDataCache(), config)

    views = build_shadow_views(scene, batch.light_items, batch.lights, RNodeDataCache(), config)

    assert views.scopes == [0]
    assert batch.lights["scope"][0] == 0
    assert len(batch.shadow_draws[0]) == 1 and len(batch.shadow_draws[NO_CLIP]) == 2
    assert views.views["params"][0][2] < 5  # fitted to the small world, not to x = 50


@pytest.fixture(scope="module")
def renderer() -> RenderService:
    try:
        return RenderService()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def _center(renderer: RenderService, root: RGroup, **config: object) -> tuple[int, ...]:
    image = renderer.render(root, RenderConfig(pixel_width=160, pixel_height=90, **config))
    return tuple(int(c) for c in image[45, 80, :3])


def test_one_sided_mask_closes_from_behind(renderer: RenderService) -> None:
    def world() -> RGroup:
        return _world(RShape2D(points=rectangle_points(2.0, 2.0), fill=Fill(RED, opacity=1.0), stroke=Stroke(width=0), transform=Transform(position=(0, 0, -1))))

    assert _center(renderer, _portal(world(), one_sided=True)) == tuple(RED.to_int_rgb())
    assert _center(renderer, _portal(world(), flip=True, one_sided=True)) == (0, 0, 0)
    assert _center(renderer, _portal(world(), flip=True)) == tuple(RED.to_int_rgb())  # two-sided


def test_scoped_light_only_lights_its_world(renderer: RenderService) -> None:
    def wall(x: float) -> RShape3D:
        return RShape3D(mesh=box_mesh(2, 2, 0.2), material=PbrMaterial("#FFFFFF", roughness=1.0), transform=Transform(position=(x, 0, 0)))

    sun = RDirectionalLight(transform=Transform(rotation=rotation_towards((0, 0, -1))), intensity=3.0)
    lit = _portal(_world(wall(0.0), sun))
    root = RGroup(children=[lit, wall(3.0)])
    image = renderer.render(root, RenderConfig(pixel_width=160, pixel_height=90, ambient_intensity=0.0))

    assert image[45, 80, :3].astype(int).sum() > 300  # inside the portal: lit
    assert image[45, 80 + int(3 * 11.25), :3].astype(int).sum() == 0  # outside: unlit
