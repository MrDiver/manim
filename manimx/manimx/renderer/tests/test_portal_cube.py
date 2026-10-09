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
from manimx.animation import ClippingGroup, Group, Light, Shape3D
from manimx.examples.portal_cube import portal_cube, views
from manimx.renderer import RenderService
from manimx.renderer.scene_graph import flatten_scene

from manimx.robjects import RGroup, RLight

SIDES = ["front", "right", "back", "left"]
WIDTH, HEIGHT = 320, 180


@pytest.fixture(scope="module")
def renderer() -> RenderService:
    try:
        return RenderService()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def _send_away(node) -> None:
    """Move ``node`` far out of the cube (and out of view)."""
    node.move_to((0.0, 0.0, -1000.0))


def _render(renderer: RenderService, cube: Group, view: str) -> np.ndarray:
    return renderer.render(cube.snapshot(), views(WIDTH, HEIGHT)[view]).astype(int)


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mask of pixels that differ visibly."""
    return np.abs(a - b).max(axis=2) > 2


# --- structure ------------------------------------------------------------------


def test_cube_has_four_one_sided_portals_with_their_own_lights() -> None:
    cube = portal_cube()
    portals = [cube.find(f"{side}-portal") for side in SIDES]

    assert all(isinstance(p, ClippingGroup) and p.content is cube.find(f"{side}-world") for p, side in zip(portals, SIDES))
    assert all(p.one_sided.value for p in portals)
    scene = flatten_scene(cube.snapshot())
    assert sorted({light.clip for light in scene.lights}) == [0, 1, 2, 3]  # every world lit, only by its own lights
    assert all(isinstance(item.light, RLight) for item in scene.lights)
    assert len(scene.lights) == len(cube.find_all(tags="light"))


def test_rooms_do_not_cast_shadows() -> None:
    cube = portal_cube()
    for side in SIDES:
        room = cube.find(f"{side}-room")
        assert room.children and all(isinstance(wall, Shape3D) and not wall.cast_shadows.value for wall in room.children)


# --- rendering ------------------------------------------------------------------


@pytest.mark.parametrize("side", SIDES)
def test_each_face_shows_only_its_own_world(renderer: RenderService, side: str) -> None:
    full = _render(renderer, portal_cube(), side)

    others_hidden = portal_cube()
    for other in SIDES:
        if other != side:
            _send_away(others_hidden.find(f"{other}-world"))
    assert not _changed(full, _render(renderer, others_hidden, side)).any()

    own_hidden = portal_cube()
    _send_away(own_hidden.find(f"{side}-world"))
    assert _changed(full, _render(renderer, own_hidden, side)).mean() > 0.05  # the face empties


def test_two_worlds_meet_at_a_corner_without_overlap(renderer: RenderService) -> None:
    full = _render(renderer, portal_cube(), "front-right")
    regions = {}
    for side in ("front", "right"):
        hidden = portal_cube()
        _send_away(hidden.find(f"{side}-world"))
        regions[side] = _changed(full, _render(renderer, hidden, "front-right"))
    front, right = regions["front"], regions["right"]

    assert front.mean() > 0.02 and right.mean() > 0.02
    # Every changed pixel belongs to exactly one world (antialiased edge
    # pixels may see both).
    assert (front & right).sum() < 0.01 * (front | right).sum()
    # From front-right, the front face is on the left of the image.
    columns = np.arange(WIDTH)
    assert columns[front.any(axis=0)].mean() < columns[right.any(axis=0)].mean()


@pytest.mark.parametrize("own_side", ["front", "right"])  # the logo and the mesh world
def test_lights_stay_in_their_world(renderer: RenderService, own_side: str) -> None:
    full = _render(renderer, portal_cube(), "front-right")
    dark = portal_cube()
    for light in dark.find(f"{own_side}-world").find_all(tags="light"):
        assert isinstance(light, Light)
        light.intensity.set(0.0)
    changed = _changed(full, _render(renderer, dark, "front-right"))

    other_side = "right" if own_side == "front" else "front"
    hidden = portal_cube()
    _send_away(hidden.find(f"{other_side}-world"))
    other_world_pixels = _changed(full, _render(renderer, hidden, "front-right"))
    assert changed.mean() > 0.01  # the world got darker
    assert not (changed & other_world_pixels).any()  # the neighbour did not


@pytest.mark.parametrize("view", ["front", "right", "back", "left", "front-right", "back-left"])
def test_base_never_shows_through_the_floors(renderer: RenderService, view: str) -> None:
    """The base plane lies 4 cm under the room floors. It may only be seen
    along the cube's bottom edge (the floors are inset by 2 cm), never
    through a floor."""
    config = views(WIDTH, HEIGHT)[view]

    def scene(base_fill: float) -> Group:
        cube = portal_cube()
        base = cube.find("base")
        base.stroke_width.set(0.0)  # compare the fill alone
        base.fill_opacity.set(base_fill)
        return cube

    changed = _changed(_render(renderer, scene(1.0), view), _render(renderer, scene(0.0), view))

    # Where the base may legitimately show: a band along its outline.
    outline = portal_cube().find("base")
    outline.fill_opacity.set(0.0)
    outline.set_stroke(color="#FFFFFF", width=40)
    band = renderer.render(RGroup(children=[outline.snapshot()]), dataclasses.replace(config, background_color="#000000"))[..., 0] > 0
    assert not (changed & ~band).any()
