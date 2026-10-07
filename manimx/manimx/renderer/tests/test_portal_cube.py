"""The portal cube example: structure and the properties of the illusion.

Reference images (test_visual.py) catch any change; these tests check *why*
the image is right, by rendering the cube twice with one thing changed and
comparing:

* each face shows exactly one world, and nothing of the other three
* at a corner, each world only touches the pixels of its own face
* a world's lights do not reach the other worlds
* the base plane under the floors never shows through (regression test for
  the painter's-order bias pulling 2D shapes in front of meshes)

The changes never hide objects: hiding removes them from the tree, which
shifts the painter's-order layers of everything after them. Instead,
worlds are moved far away (their portal then shows nothing of them) and
lights are dimmed to zero, which keeps the draw order identical.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from manimx import ClippingGroup, Fill, Group, Light, Shape3D, Stroke, Transform
from manimx.examples.portal_cube import portal_cube, views
from manimx.renderer import WGPURenderer
from manimx.renderer.scene_graph import flatten_scene

WORLDS = {"front": "logo-world", "right": "mesh-world", "back": "curve-world", "left": "grid-world"}
WIDTH, HEIGHT = 320, 180


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def _find(root: Group, id: str):
    stack = [root]
    while stack:
        node = stack.pop()
        if node.id == id:
            return node
        stack.extend(getattr(node, "children", []))
    raise KeyError(id)


def _send_away(node) -> None:
    """Move ``node`` far out of the cube (and out of view)."""
    node.transform = Transform(position=(0.0, 0.0, -1000.0))


def _render(renderer: WGPURenderer, scene: Group, view: str) -> np.ndarray:
    return renderer.render(scene, views(WIDTH, HEIGHT)[view]).astype(int)


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mask of pixels that differ visibly."""
    return np.abs(a - b).max(axis=2) > 2


# --- structure ------------------------------------------------------------------


def test_cube_has_four_one_sided_portals_with_their_own_lights() -> None:
    cube = portal_cube()
    portals = [child for child in cube.children if isinstance(child, ClippingGroup)]

    assert [p.content.id for p in portals] == list(WORLDS.values())
    assert all(p.one_sided for p in portals)
    scene = flatten_scene(cube)
    assert sorted({light.clip for light in scene.lights}) == [0, 1, 2, 3]  # every world lit, only by its own lights
    assert all(isinstance(item.light, Light) for item in scene.lights)


def test_rooms_do_not_cast_shadows() -> None:
    rooms = [_find(portal_cube(), f"{name}-room") for name in ("logo", "mesh", "curve", "grid")]

    for room in rooms:
        assert room.children and all(isinstance(wall, Shape3D) and not wall.cast_shadows for wall in room.children)


# --- rendering ------------------------------------------------------------------


@pytest.mark.parametrize("side", list(WORLDS))
def test_each_face_shows_only_its_own_world(renderer: WGPURenderer, side: str) -> None:
    full = _render(renderer, portal_cube(), side)

    others_hidden = portal_cube()
    for other, world in WORLDS.items():
        if other != side:
            _send_away(_find(others_hidden, world))
    assert not _changed(full, _render(renderer, others_hidden, side)).any()

    own_hidden = portal_cube()
    _send_away(_find(own_hidden, WORLDS[side]))
    assert _changed(full, _render(renderer, own_hidden, side)).mean() > 0.05  # the face empties


def test_two_worlds_meet_at_a_corner_without_overlap(renderer: WGPURenderer) -> None:
    full = _render(renderer, portal_cube(), "front-right")
    regions = {}
    for side in ("front", "right"):
        hidden = portal_cube()
        _send_away(_find(hidden, WORLDS[side]))
        regions[side] = _changed(full, _render(renderer, hidden, "front-right"))
    front, right = regions["front"], regions["right"]

    assert front.mean() > 0.02 and right.mean() > 0.02
    # Every changed pixel belongs to exactly one world (antialiased edge
    # pixels may see both).
    assert (front & right).sum() < 0.01 * (front | right).sum()
    # From front-right, the front face is on the left of the image.
    columns = np.arange(WIDTH)
    assert columns[front.any(axis=0)].mean() < columns[right.any(axis=0)].mean()


@pytest.mark.parametrize("world", ["logo-world", "mesh-world"])
def test_lights_stay_in_their_world(renderer: WGPURenderer, world: str) -> None:
    full = _render(renderer, portal_cube(), "front-right")
    dark = portal_cube()
    for light in [node for node in _find(dark, world).children if isinstance(node, Light)]:
        light.intensity = 0.0
    changed = _changed(full, _render(renderer, dark, "front-right"))

    own_side = "front" if world == "logo-world" else "right"
    other_side = "right" if own_side == "front" else "front"
    hidden = portal_cube()
    _send_away(_find(hidden, WORLDS[other_side]))
    other_world_pixels = _changed(full, _render(renderer, hidden, "front-right"))
    assert changed.mean() > 0.01  # the world got darker
    assert not (changed & other_world_pixels).any()  # the neighbour did not


@pytest.mark.parametrize("view", ["front", "right", "back", "left", "front-right", "back-left"])
def test_base_never_shows_through_the_floors(renderer: WGPURenderer, view: str) -> None:
    """The base plane lies 4 cm under the room floors. It may only be seen
    along the cube's bottom edge (the floors are inset by 2 cm), never
    through a floor."""
    config = views(WIDTH, HEIGHT)[view]

    def scene(base_fill: float) -> Group:
        cube = portal_cube()
        base = _find(cube, "base")
        base.stroke = Stroke(width=0)  # compare the fill alone
        base.fill = Fill("#2a2a35", opacity=base_fill)
        return cube

    changed = _changed(_render(renderer, scene(1.0), view), _render(renderer, scene(0.0), view))

    # Where the base may legitimately show: a band along its outline.
    outline = _find(portal_cube(), "base")
    outline.fill = Fill(opacity=0.0)
    outline.stroke = Stroke("#FFFFFF", width=40)
    band = renderer.render(Group("band", children=[outline]), dataclasses.replace(config, background_color="#000000"))[..., 0] > 0
    assert not (changed & ~band).any()
