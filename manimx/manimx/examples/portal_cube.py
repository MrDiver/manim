"""The "four worlds in one cube" illusion.

A cube made of six stroked planes. Its four side faces are one-sided masks
(portals): each one shows a different world, and because the faces of a
convex cube seen from outside never overlap on screen, every view shows
each world only through its own face. Orbiting the camera reveals them one
after another, and at the corners two worlds sit side by side inside the
same cube.

Each world is a diorama filling the cube: a room (floor, ceiling and three
walls, open towards its portal) with its own content and its own lights.
Lights inside an RClippingGroup only light, and only cast shadows from, that
group's objects, so the worlds do not leak light or shadows into each other.

    front (-y)  the Manim logo in a Manim-beige room, warm studio light
    right (+x)  lit meshes with soft shadows in a sunset
    back  (+y)  rainbow 3D curves in a dark room
    left  (-x)  Blender's grid world with stress-test shapes

World +z is up. The cube spans [-2, 2] x [-2, 2] x [0, 4].
"""

from __future__ import annotations

import math

import numpy as np

from ..objects import (
    CapStyle,
    RClippingGroup,
    RCurve3D,
    RDirectionalLight,
    Fill,
    RGroup,
    JointType,
    PbrMaterial,
    RPointLight,
    RNode,
    RShape2D,
    RShape3D,
    RSpotLight,
    Stroke,
    Transform,
    box_mesh,
    sphere_mesh,
    torus_mesh,
)
from ..objects.geometry import circle, regular_polygon, square
from ..renderer.config import Camera, RenderConfig
from ..utils.color import BLUE, GREEN, ORANGE, PURPLE, RED, WHITE, YELLOW
from .manim_logo import manim_logo

SIZE = 4.0
HALF = SIZE / 2
CENTER = np.array([0.0, 0.0, HALF])
BACKGROUND = "#15151c"
FRAME_COLOR = "#f0f0f0"

# Outward normal of each side face, and the rotation that turns a shape's
# local +z (its front) into that normal.
_SIDES = {
    "front": ((0.0, -1.0, 0.0), (math.pi / 2, (1, 0, 0))),
    "right": ((1.0, 0.0, 0.0), (math.pi / 2, (0, 1, 0))),
    "back": ((0.0, 1.0, 0.0), (-math.pi / 2, (1, 0, 0))),
    "left": ((-1.0, 0.0, 0.0), (-math.pi / 2, (0, 1, 0))),
}


def portal_cube() -> RGroup:
    worlds = {"front": logo_world(), "right": mesh_world(), "back": curve_world(), "left": grid_world()}
    portals = [
        RClippingGroup(mask=_face(side, Fill(opacity=1.0), Stroke(width=0)), content=world, one_sided=True)
        for side, world in worlds.items()
    ]
    return RGroup(children=[*portals, _frame()])


def views(pixel_width: int = 1280, pixel_height: int = 720) -> dict[str, RenderConfig]:
    """Each face head-on, plus two corners where two worlds meet."""
    cameras = {
        "front": orbit_camera(-90.0),
        "right": orbit_camera(0.0),
        "back": orbit_camera(90.0),
        "left": orbit_camera(180.0),
        "front-right": orbit_camera(-45.0),
        "back-left": orbit_camera(135.0, elevation=35.0),
    }
    return {name: RenderConfig(pixel_width=pixel_width, pixel_height=pixel_height, background_color=BACKGROUND, camera=camera) for name, camera in cameras.items()}


def orbit_camera(azimuth: float, elevation: float = 22.0, distance: float = 11.0) -> Camera:
    """A perspective camera on a circle around the cube (degrees)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    eye = CENTER + distance * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return Camera.look_at(tuple(eye), tuple(CENTER), field_of_view=math.radians(40), near=0.3, far=100)


# --------------------------------------------------------------------------
# The cube
# --------------------------------------------------------------------------


def _face(side: str, fill: Fill, stroke: Stroke) -> RShape2D:
    normal, (angle, axis) = _SIDES[side]
    return square(side_length=SIZE, fill=fill, stroke=stroke, transform=Transform(position=tuple(CENTER + HALF * np.array(normal))).rotated(angle, axis))


def _frame() -> RGroup:
    """Six stroked planes: four open side outlines, a solid lid and base."""
    outline = Stroke(FRAME_COLOR, width=6, joint=JointType.MITER)
    sides = [_face(side, Fill(opacity=0.0), outline) for side in _SIDES]
    lid = square(side_length=SIZE, fill=Fill("#2a2a35", opacity=1.0), stroke=outline, transform=Transform(position=(0.0, 0.0, SIZE)))
    base = square(side_length=SIZE, fill=Fill("#2a2a35", opacity=1.0), stroke=outline, transform=Transform().rotated(math.pi, (1, 0, 0)))
    return RGroup(children=[*sides, lid, base])


def _room(open_side: str, wall: str, floor: str) -> RGroup:
    """Floor, ceiling and three walls just inside the cube; the wall behind
    the portal is left out so the camera can look in.

    The room does not cast shadows: a world's sun shines in through the
    ceiling, otherwise the closed room would keep every world in the dark.
    """
    inset = SIZE - 0.04
    thickness = 0.02
    walls = [
        RShape3D(mesh=box_mesh(inset, inset, thickness), material=PbrMaterial(floor, roughness=0.8), transform=Transform(position=(0.0, 0.0, 0.03)), cast_shadows=False),
        RShape3D(mesh=box_mesh(inset, inset, thickness), material=PbrMaterial(wall, roughness=0.9), transform=Transform(position=(0.0, 0.0, SIZE - 0.03)), cast_shadows=False),
    ]
    for side, (normal, _) in _SIDES.items():
        if side == open_side:
            continue
        position = CENTER + (HALF - 0.03) * np.array(normal)
        size = (thickness, inset, inset) if normal[0] else (inset, thickness, inset)
        walls.append(RShape3D(mesh=box_mesh(*size), material=PbrMaterial(wall, roughness=0.9), transform=Transform(position=tuple(position)), cast_shadows=False))
    return RGroup(children=walls)


def _facing(side: str, obj: RNode, position: tuple[float, float, float]) -> RNode:
    """Stand a flat object up so its front faces out of portal ``side``."""
    _, (angle, axis) = _SIDES[side]
    obj.transform = Transform(position=position, rotation=obj.transform.rotation, scale=obj.transform.scale).rotated(angle, axis)
    return obj


# --------------------------------------------------------------------------
# The four worlds
# --------------------------------------------------------------------------


def logo_world() -> RGroup:
    logo = manim_logo()
    logo.transform = Transform(scale=(0.55, 0.55, 0.55))
    return RGroup(
        children=[
            _room("front", wall="#ece6e2", floor="#d8d0ca"),
            _facing("front", logo, (0.0, 0.6, 2.0)),
            RSpotLight.aimed((-1.0, -1.6, 3.6), (0.0, 0.6, 1.5), intensity=25, color="#fff1dc", outer_cone_angle=0.9, inner_cone_angle=0.5, size=0.4),
            RPointLight(intensity=3, color="#ffe0c0", transform=Transform(position=(1.2, -1.2, 1.0))),
        ],
    )


def mesh_world() -> RGroup:
    rng = np.random.default_rng(7)
    palette = ["#58C4DD", "#FC6255", "#83C167", "#FFFF00", "#9A72AC"]
    balls = [
        RShape3D(mesh=sphere_mesh(0.25 + 0.15 * rng.random()), material=PbrMaterial(palette[i % 5], roughness=0.3 + 0.4 * rng.random(), metallic=float(i % 3 == 0)), transform=Transform(position=(rng.uniform(-1.4, 1.4), rng.uniform(-1.4, 1.4), 0.45)))
        for i in range(9)
    ]
    torus = RShape3D(mesh=torus_mesh(0.6, 0.2), material=PbrMaterial("#d4af37", metallic=1.0, roughness=0.25), transform=Transform(position=(0.3, 0.2, 1.6)).rotated(1.1, (0, 1, 0)))
    column = RShape3D(mesh=box_mesh(0.5, 0.5, 2.4), material=PbrMaterial("#e8e8e8", roughness=0.6), transform=Transform(position=(-0.9, 1.1, 1.25)))
    return RGroup(
        children=[
            _room("right", wall="#3a2e3e", floor="#c9b8a8"),
            *balls,
            torus,
            column,
            RDirectionalLight.aimed((-1.0, 0.4, -0.45), intensity=4.0, color="#ffb070", size=0.05),
            RPointLight(intensity=4, color="#7090ff", transform=Transform(position=(1.2, -1.2, 3.4)), size=0.3),
        ],
    )


def curve_world() -> RGroup:
    rainbow = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED]
    knot = RCurve3D.parametric(
        lambda t: (0.45 * (math.sin(t) + 2 * math.sin(2 * t)), 0.45 * (math.cos(t) - 2 * math.cos(2 * t)), 2.0 - 0.45 * math.sin(3 * t)),
        (0.0, 2 * math.pi),
        120,
        stroke=Stroke(rainbow, width=[6, 18, 6, 18, 6]),
    )
    helices = [
        RCurve3D.parametric(
            lambda t, i=i: (1.45 * math.cos(i * math.pi / 2) + 0.25 * math.cos(t), 1.45 * math.sin(i * math.pi / 2) + 0.25 * math.sin(t), 0.2 + 0.11 * t),
            (0.0, 10 * math.pi),
            120,
            stroke=Stroke([WHITE, BLUE], width=4),
        )
        for i in range(4)
    ]
    return RGroup(
        children=[_room("back", wall="#101018", floor="#181828"), knot, *helices, RPointLight(intensity=2, color="#6060ff", transform=Transform(position=(0, 0, 3.5)))],
    )


def grid_world() -> RGroup:
    lines = [((c, -1.9), (c, 1.9)) for c in np.arange(-1.5, 1.6, 0.5)] + [((-1.9, c), (1.9, c)) for c in np.arange(-1.5, 1.6, 0.5)]
    grid = RShape2D(points=np.vstack([RShape2D.from_corners(line).points for line in lines]), stroke=Stroke("#7a7a7a", width=2), transform=Transform(position=(0, 0, 0.05)))
    axes = RGroup(
        children=[
            RShape2D.from_corners([(-1.9, 0), (1.9, 0)], stroke=Stroke("#ff3352", width=3), transform=Transform(position=(0, 0, 0.06))),
            RShape2D.from_corners([(0, -1.9), (0, 1.9)], stroke=Stroke("#8bdc00", width=3), transform=Transform(position=(0, 0, 0.06))),
        ],
    )
    zigzag = RShape2D.from_corners([(-0.6, -0.3), (-0.3, 0.3), (0.0, -0.3), (0.3, 0.3), (0.6, -0.3)], stroke=Stroke(WHITE, width=14, joint=JointType.ROUND, cap=CapStyle.ROUND))
    hexagon = RClippingGroup(
        mask=regular_polygon(6, radius=0.6, fill=Fill(opacity=1.0), stroke=Stroke(width=0)),
        content=RGroup(children=[square(side_length=0.2, fill=Fill(c, opacity=1.0), stroke=Stroke(width=0), transform=Transform(position=(-0.7 + 0.2 * i, 0, 0), scale=(1, 7, 1))) for i, c in enumerate([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED, ORANGE])]),
    )
    ring = circle(radius=0.5, stroke=Stroke([RED, YELLOW, GREEN, BLUE, PURPLE, RED], width=[4, 16, 4, 16, 4]))
    pentagon = regular_polygon(5, radius=0.5, fill=Fill([RED, BLUE], opacity=0.7), stroke=Stroke(YELLOW, width=6, opacity=0.6))
    return RGroup(
        children=[
            _room("left", wall="#3d3d3d", floor="#353535"),
            grid,
            axes,
            _facing("left", zigzag, (0.0, -1.0, 2.6)),
            _facing("left", hexagon, (0.5, 0.6, 1.4)),
            _facing("left", ring, (-0.6, 0.9, 2.8)),
            RGroup(children=[pentagon], transform=Transform(position=(-0.4, -0.7, 0.4))),
            RDirectionalLight.aimed((0.3, 0.5, -1.0), intensity=2.0),
        ],
    )
