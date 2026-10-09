"""The render service: shared instance, kept GPU objects, bounded caches,
and stages usable on their own."""

from __future__ import annotations

import numpy as np
import pytest
import wgpu
from manimx.userland.geometry import arc_points
from manimx.robjects import RShape2D, new_uid
from manimx.renderer import FramePacker, MotionBlur, RenderConfig, RNodeDataCache, RenderService
from manimx.renderer.webgpu.gpu import ResourceCache

from manimx.robjects import Fill, RGroup


@pytest.fixture(scope="module")
def renderer() -> RenderService:
    try:
        return RenderService.shared()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def scene() -> RGroup:
    return RGroup(children=[RShape2D(points=arc_points(), fill=Fill("#e07a5f", opacity=1.0))])


def test_one_shared_renderer_with_composers_per_effect_chain(renderer: RenderService) -> None:
    assert RenderService.shared() is renderer
    blur = (MotionBlur(samples=4),)
    assert renderer.composer(blur) is renderer.composer(blur)
    assert renderer.composer(blur) is not renderer.composer()


def test_resource_cache_reuses_views_and_bind_groups(renderer: RenderService) -> None:
    device = renderer.renderer.context.device
    resources = ResourceCache(device, capacity=2)
    texture = device.create_texture(size=(4, 4, 1), format=wgpu.TextureFormat.rgba8unorm, usage=wgpu.TextureUsage.TEXTURE_BINDING)
    view = resources.view(texture)
    assert resources.view(texture) is view and resources.view(view) is view
    assert resources.view(texture, base_mip_level=0, mip_level_count=1) is not view

    layout = device.create_bind_group_layout(
        entries=[{"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": wgpu.TextureSampleType.float}}]
    )
    group = resources.bind_group(layout, [view])
    assert resources.bind_group(layout, [view]) is group
    other = resources.view(device.create_texture(size=(4, 4, 1), format=wgpu.TextureFormat.rgba8unorm, usage=wgpu.TextureUsage.TEXTURE_BINDING))
    third = resources.view(device.create_texture(size=(4, 4, 1), format=wgpu.TextureFormat.rgba8unorm, usage=wgpu.TextureUsage.TEXTURE_BINDING))
    resources.bind_group(layout, [other])
    resources.bind_group(layout, [third])  # capacity 2: the first entry goes
    assert resources.bind_group(layout, [view]) is not group


def test_renders_after_size_changes_match_a_fresh_renderer(renderer: RenderService) -> None:
    small, large = RenderConfig(pixel_width=160, pixel_height=90), RenderConfig(pixel_width=320, pixel_height=180)
    first = renderer.render(scene(), small)
    renderer.render(scene(), large)  # new targets, attachments and views
    again = renderer.render(scene(), small)

    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(again, RenderService().render(scene(), small))


def test_the_renderer_takes_packets_from_any_packer() -> None:
    """Each packer's dirty ranges are relative to its own previous packet;
    a renderer that was fed by another packer in between uploads all."""
    service = RenderService()
    config = RenderConfig(pixel_width=160, pixel_height=90)
    root = scene()  # the same uids each time: the packer's slots stay
    own = service.render(root, config)
    other = RGroup(children=[RShape2D(points=arc_points()[:8] * 0.5, fill=Fill("#3d405b", opacity=1.0))])
    shared_packer = service.packer
    service.packer = FramePacker()
    try:
        service.render(other, config)
    finally:
        service.packer = shared_packer
    np.testing.assert_array_equal(service.render(root, config), own)


def test_cache_entries_unused_for_a_while_are_swept() -> None:
    cache = RNodeDataCache()
    old, kept, other = new_uid(), new_uid(), new_uid()
    cache.put(old, "mesh", "v1", "old")
    cache.next_generation()
    cache.get_or_create(kept, "mesh", "v1", lambda: "kept")
    cache.next_generation()

    assert cache.sweep(max_age=1) == 1  # old was last used two generations ago
    assert cache.get(old, "mesh", "v1") is None and cache.get(kept, "mesh", "v1") == "kept"
    cache.invalidate(kept)
    cache.put(other, "mesh", "v1", "x")
    cache.retain([])
    assert len(cache) == 0 and cache.sweep(0) == 0
