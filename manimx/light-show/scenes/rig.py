"""The stage both scenes play on, and the vocabulary of the show.

Not a scene itself (it is not registered in manimx.json): ``light_show``
and ``finale`` import it and build their own copy of the stage.

The vocabulary, all ordinary manimx animations and derived signals:

* :class:`MovingHead`: a glTF fixture with a SpotLight in its head. Its
  lens glows with the light's color and intensity (derived signals), and
  :meth:`MovingHead.pan_tilt` turns a world point into the yoke's pan and
  the head's tilt.
* :class:`Aim`: point a moving head at a moving world point, ``target(t)``.
  Panning the yoke and tilting the head turn the spot with them, because
  the light is a child of the head.
* :class:`Drive`: set any signal to ``value(t)`` for a while (intensity
  chases, strobes, color cycles).
* :func:`fly`: the camera along a smooth path through timed waypoints, as
  derived signals of ``scene_time``, so it keeps flying while the scene
  plays one section after another.

t is always the time in seconds since the animation started.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import quaternion
from manimx.animation import (
    COLOR,
    ROTATION,
    Animation,
    Camera,
    Change,
    Group,
    RootSignal,
    Shape2D,
    Shape3D,
    SpotLight,
    Target,
    TimeSignals,
    derive,
    linear,
    smooth,
)
from manimx.examples.manim_logo import LOGO_BLUE, LOGO_GREEN, LOGO_RED, M_GLYPH_POINTS
from manimx.renderer import Camera as RenderCamera
from manimx.robjects import PbrMaterial
from manimx.robjects.transform import quaternion_from_axis_angle
from manimx.userland import Box, Circle, Square, Torus, Triangle, load_gltf
from manimx.userland.geometry import sphere_mesh, torus_mesh
from manimx.utils.color import ManimColor

from .music import kick_envelope

ASSETS = Path(__file__).resolve().parent.parent / "assets"

FRONT_TRUSS = (3.0, 8.5)
"""y and height of the truss above the floor."""
BACK_TRUSS = (9.5, 8.5)
"""y and height of the truss above the stage."""
STAGE = (6.0, 12.0, 0.8)
"""The stage's front and back edge (y) and its height."""
SCULPTURE = np.array([0.0, 0.0, 2.2])
"""The center of the moving shapes on the floor."""
LOGO_CENTER = np.array([0.0, 8.6, 4.0])

DEEP_BLUE = "#1d3bff"
CYAN = "#19e6ff"
MAGENTA = "#ff2bd6"
AMBER = "#ffae2b"
RED = "#ff2b3d"
VIOLET = "#8a2bff"
WHITE = "#ffffff"
LOGO_LIGHT = "#ece6e2"
"""The M of the logo on dark backgrounds."""
X_GLOW = "#41e8ff"


def rgb(color: str) -> tuple[float, float, float]:
    r, g, b = ManimColor.parse(color).to_rgb()
    return (float(r), float(g), float(b))


def hue(h: float, saturation: float = 1.0) -> tuple[float, float, float]:
    r, g, b = ManimColor.from_hsv((h % 1.0, saturation, 1.0)).to_rgb()
    return (float(r), float(g), float(b))


# -- moving heads ------------------------------------------------------------------

LIGHT_SPOT = 0.26
"""How far below the head's center its spot sits (just outside the lens)."""
HEAD_PIVOT = np.array([0.0, -0.62, 0.0])
"""The head's center in the model (glTF space: base 0, yoke -0.18, head -0.44 lower)."""


@dataclass(eq=False)
class MovingHead:
    """``moving_head.glb`` hanging at ``mount``, with a SpotLight in its head."""

    model: Group
    light: SpotLight
    mount: np.ndarray
    yoke: Group = field(init=False)
    head: Group = field(init=False)

    def __post_init__(self) -> None:
        self.yoke = self.model.find("yoke")
        self.head = self.model.find("head")
        self._turn = quaternion.as_rotation_matrix(self.model.rotation.value)  # glTF space -> world
        self.pivot = self.mount + self._turn @ HEAD_PIVOT

    @classmethod
    def hang(cls, name: str, mount: Sequence[float], color: str = DEEP_BLUE) -> MovingHead:
        model = load_gltf(ASSETS / "moving_head.glb", name=name).move_to(mount)
        # In the head's own (glTF) space the lens looks along -y.
        light = SpotLight(
            f"{name}.spot",
            position=(0.0, -LIGHT_SPOT, 0.0),
            target=(0.0, -10.0, 0.0),
            color=color,
            intensity=0.0,
            inner_cone_angle=0.1,
            outer_cone_angle=0.24,
            size=0.05,
        )
        model.find("head").add(light)
        for part in model.find_all():
            if isinstance(part, Shape3D):
                part.set_cast_shadows(False)  # the spot sits in the head: its housing must not block it
        # The lens shows what the light does: its color, glowing with its intensity.
        lens = model.find_all(tags="material:Lens")[0]
        lens.emissive = derive(lambda c: c, light.color, kind=COLOR, name=f"{name}.lens.emissive")
        lens.emissive_strength = derive(lambda i: 0.02 + i / 150.0, light.intensity, name=f"{name}.lens.strength")
        return cls(model, light, np.asarray(mount, dtype=np.float64))

    def pan_tilt(self, target: Sequence[float]) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """The yoke's and the head's rotation that point the lens at ``target``.

        In glTF space the head first tilts about x, then the yoke pans
        about y: the lens direction ``(0, -1, 0)`` becomes
        ``(-sin(tilt) sin(pan), -cos(tilt), -sin(tilt) cos(pan))``.
        """
        direction = self._turn.T @ (np.asarray(target, dtype=np.float64) - self.pivot)
        dx, dy, dz = direction / np.linalg.norm(direction)
        tilt = math.acos(max(-1.0, min(1.0, -dy)))
        pan = math.atan2(-dx, -dz)
        return quaternion_from_axis_angle((0, 1, 0), pan), quaternion_from_axis_angle((1, 0, 0), tilt)

    def point_at(self, target: Sequence[float]) -> MovingHead:
        """Aim right away (while composing a scene)."""
        pan, tilt = self.pan_tilt(target)
        self.yoke.rotation.set(pan)
        self.head.rotation.set(tilt)
        return self


@dataclass(eq=False)
class Aim(Animation):
    """Point ``fixture`` at ``target(t)``, a world point. During the first
    ``blend`` seconds it turns smoothly from wherever it pointed before."""

    fixture: MovingHead
    target: Callable[[float], Sequence[float]]
    run_time: float = 1.0
    blend: float = 0.5
    rate: Any = staticmethod(linear)

    def targets(self) -> Sequence[Target]:
        return [
            Target(self.fixture.yoke.rotation, Change.ABSOLUTE),
            Target(self.fixture.head.rotation, Change.ABSOLUTE),
        ]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        t = alpha * self.run_time
        pan, tilt = self.fixture.pan_tilt(self.target(t))
        value = pan if signal is self.fixture.yoke.rotation else tilt
        if t < self.blend:
            return ROTATION.interpolate(start_value, value, smooth(t / self.blend))
        return value


@dataclass(eq=False)
class Drive(Animation):
    """Set ``signal`` to ``value(t)``."""

    signal: RootSignal
    value: Callable[[float], Any]
    run_time: float = 1.0
    rate: Any = staticmethod(linear)

    def targets(self) -> Sequence[Target]:
        return [Target(self.signal, Change.ABSOLUTE)]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        return self.value(alpha * self.run_time)


def on_floor(x: float, y: float, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z])


# -- the camera ------------------------------------------------------------------------


def fly(
    camera: Camera,
    time: TimeSignals,
    waypoints: Sequence[tuple[float, Sequence[float], Sequence[float]]],
) -> None:
    """Fly ``camera`` through ``waypoints`` ``(time, eye, look_at)``: a cubic
    Hermite spline through them (Catmull-Rom tangents, at rest at the
    first and the last), looking at a point on a second spline. Before
    the first and after the last waypoint it holds."""
    times = np.array([w[0] for w in waypoints], dtype=np.float64)
    eyes = np.array([w[1] for w in waypoints], dtype=np.float64)
    looks = np.array([w[2] for w in waypoints], dtype=np.float64)

    def tangents(points: np.ndarray) -> np.ndarray:
        result = np.zeros_like(points)
        result[1:-1] = (points[2:] - points[:-2]) / (times[2:] - times[:-2])[:, None]
        return result

    eye_tangents, look_tangents = tangents(eyes), tangents(looks)

    def at(t: float, points: np.ndarray, slopes: np.ndarray) -> np.ndarray:
        t = min(max(t, times[0]), times[-1])
        i = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
        span = times[i + 1] - times[i]
        s = (t - times[i]) / span
        h00, h10, h01, h11 = (
            2 * s**3 - 3 * s**2 + 1,
            s**3 - 2 * s**2 + s,
            -2 * s**3 + 3 * s**2,
            s**3 - s**2,
        )
        return h00 * points[i] + h10 * span * slopes[i] + h01 * points[i + 1] + h11 * span * slopes[i + 1]

    for index, name in enumerate("xyz"):
        setattr(
            camera.position,
            name,
            derive(
                lambda t, i=index: float(at(t, eyes, eye_tangents)[i]),
                time.scene_time,
                name=f"camera.{name}",
            ),
        )
    camera.rotation = derive(
        lambda t: RenderCamera.look_at(tuple(at(t, eyes, eye_tangents)), tuple(at(t, looks, look_tangents))).transform.rotation,
        time.scene_time,
        kind=ROTATION,
        name="camera.rotation",
    )


# -- the stage ---------------------------------------------------------------------------


@dataclass(eq=False)
class Stage:
    root: Group
    front: list[MovingHead]
    back: list[MovingHead]
    sculpture: Group
    ball: Group
    arches: list[Torus]
    woofers: list[Group]
    floor_lines: list[Box]

    @property
    def heads(self) -> list[MovingHead]:
        return [*self.front, *self.back]


def build_stage(
    time: TimeSignals,
    beat: Callable[[float], float] = kick_envelope,
    centerpiece: bool = True,
) -> Stage:
    """The floor, the stage, two trusses with moving heads, speaker stacks,
    the mirror ball, the sculpture of moving shapes and the arches of
    light over the aisle (without the sculpture and the ball when
    ``centerpiece`` is False: the finale clears the floor). Everything that moves on its own (the ball,
    the sculpture, the arches, the floor grid, the speaker cones) is a
    derived signal of ``scene_time``; ``beat(t)`` is the pulse they follow."""
    scene_time = time.scene_time
    dark = PbrMaterial("#3a3d48", roughness=0.3, metallic=0.1)
    steel = PbrMaterial("#9aa0aa", roughness=0.35, metallic=1.0)

    floor = Box(70, 70, 0.2, name="floor", material=dark).move_to((0, 0, -0.1))
    stage_front, stage_back, stage_height = STAGE
    platform = Box(
        22,
        stage_back - stage_front,
        stage_height,
        name="platform",
        material=PbrMaterial("#2c2f3a", roughness=0.25),
    ).move_to((0, (stage_front + stage_back) / 2, stage_height / 2))
    edge = Box(
        22,
        0.06,
        0.06,
        name="stage-edge",
        color="#000000",
        emissive=CYAN,
        emissive_strength=2.5,
        cast_shadows=False,
    )
    edge.move_to((0, stage_front - 0.03, stage_height - 0.03))

    # A grid of faint lines in the floor that flashes with the kick.
    floor_lines = []
    for i, x in enumerate(np.arange(-20, 21, 4.0)):
        line = Box(
            0.05,
            44,
            0.02,
            name=f"floor-line-{i}",
            color="#000000",
            emissive=DEEP_BLUE,
            cast_shadows=False,
        ).move_to((x, -16, 0.0))
        floor_lines.append(line)
    for line in floor_lines:
        line.emissive_strength = derive(lambda t: 0.25 + 1.6 * beat(t), scene_time, name=f"{line.name}.strength")

    trusses = []
    for name, (y, height) in (("front-truss", FRONT_TRUSS), ("back-truss", BACK_TRUSS)):
        trusses.append(Box(22, 0.35, 0.35, name=name, material=steel).move_to((0, y, height + 0.2)))
        for side in (-1, 1):
            trusses.append(
                Box(0.35, 0.35, height + 0.4, name=f"{name}-leg-{side}", material=steel).move_to((side * 11, y, (height + 0.4) / 2))
            )

    front = [MovingHead.hang(f"front-{i}", (x, FRONT_TRUSS[0], FRONT_TRUSS[1])) for i, x in enumerate((-8.0, -4.0, 0.0, 4.0, 8.0))]
    back = [MovingHead.hang(f"back-{i}", (x, BACK_TRUSS[0], BACK_TRUSS[1])) for i, x in enumerate((-6.0, -2.0, 2.0, 6.0))]
    for head in front:
        head.point_at(SCULPTURE * (1, 1, 0))
    for head in back:
        head.point_at((head.mount[0] * 0.5, 8.0, stage_height))

    speakers, woofers = [], []
    for side in (-1, 1):
        for level in range(2):
            speaker = load_gltf(ASSETS / "speaker.glb", name=f"speaker-{side}-{level}")
            speaker.move_to((side * 9.0, stage_front - 0.8, 0.8 + 1.6 * level)).rotate(-side * 0.25, (0, 0, 1))
            woofer = speaker.find("woofer")
            # The cone pumps with the kick (along its own axis, glTF z).
            woofer.scale.z = derive(
                lambda t: 1.0 + 0.8 * beat(t),
                scene_time,
                name=f"{speaker.name}.woofer.z",
            )
            speakers.append(speaker)
            woofers.append(woofer)

    ball = load_gltf(ASSETS / "mirror_ball.glb", name="mirror-ball").move_to((0.0, 4.5, 7.2))
    upright = ball.rotation.value
    ball.rotation = derive(
        lambda t: quaternion_from_axis_angle((0, 0, 1), 0.6 * t) * upright,
        scene_time,
        kind=ROTATION,
        name="mirror-ball.rotation",
    )

    sculpture = _sculpture(scene_time)
    arches = _arches(scene_time, beat)
    root = Group(
        floor, *floor_lines, platform, edge, *trusses, *(h.model for h in front), *(h.model for h in back),
        *speakers, *arches, name="stage",
    )  # fmt: skip
    if centerpiece:
        root.add(ball, sculpture)
    return Stage(root, front, back, sculpture, ball, arches, woofers, floor_lines)


def _sculpture(scene_time: RootSignal) -> Group:
    """A chrome ring turning about two axes around a white cube turning the
    other way, with six glossy spheres orbiting them, on a plinth. White
    and chrome, so the moving heads paint them."""
    plinth = Box(1.8, 1.8, 0.7, name="plinth", material=PbrMaterial("#15161c", roughness=0.4)).move_to((0, 0, 0.35))
    ring = Torus(
        1.25,
        0.16,
        name="ring",
        material=PbrMaterial("#d8dce6", metallic=1.0, roughness=0.18),
    )
    cube = Box(0.95, 0.95, 0.95, name="cube", material=PbrMaterial("#f2f2f2", roughness=0.35))
    for node, turn in ((ring, lambda t: quaternion_from_axis_angle((1, 0, 0), 0.7 * t) * quaternion_from_axis_angle((0, 1, 0), 0.45 * t)),
                       (cube, lambda t: quaternion_from_axis_angle((1, 1, 1), -0.9 * t))):  # fmt: skip
        node.move_to(SCULPTURE)
        node.rotation = derive(turn, scene_time, kind=ROTATION, name=f"{node.name}.rotation")
    orbiters = []
    mesh = sphere_mesh(0.2, 24, 12)
    for i in range(6):
        sphere = Shape3D(
            mesh,
            f"orbiter-{i}",
            material=PbrMaterial("#ffffff", roughness=0.15, metallic=0.2),
        )
        phase, tilt = i * math.tau / 6, 0.5 * math.sin(i * 1.7)

        def where(t: float, axis: int, phase: float = phase, tilt: float = tilt) -> float:
            angle = 0.9 * t + phase
            point = (
                2.1 * math.cos(angle),
                2.1 * math.sin(angle),
                SCULPTURE[2] + 0.6 * math.sin(2 * angle + tilt),
            )
            return point[axis]

        for axis, name in enumerate("xyz"):
            setattr(
                sphere.position,
                name,
                derive(
                    lambda t, a=axis, w=where: w(t, a),
                    scene_time,
                    name=f"orbiter-{i}.{name}",
                ),
            )
        orbiters.append(sphere)
    return Group(plinth, ring, cube, *orbiters, name="sculpture")


def _arches(scene_time: RootSignal, beat: Callable[[float], float]) -> list[Torus]:
    """Glowing half rings over the aisle the camera flies down (their lower
    halves are under the floor). A wave of light runs through them towards
    the stage on every kick."""
    arches = []
    mesh = torus_mesh(4.2, 0.07, 96, 12)
    for i, y in enumerate(np.arange(-30.0, -14.0, 3.0)):
        arch = Shape3D(
            mesh,
            f"arch-{i}",
            color="#000000",
            emissive=MAGENTA if i % 2 else CYAN,
            cast_shadows=False,
        )
        arch.move_to((0.0, y, 0.0)).rotate(math.pi / 2, (1, 0, 0))
        delay = 0.06 * i
        arch.emissive_strength = derive(
            lambda t, d=delay: 0.5 + 2.0 * beat(t - d),
            scene_time,
            name=f"arch-{i}.strength",
        )
        arches.append(arch)
    return arches


# -- the ManimX logo ---------------------------------------------------------------------


@dataclass(eq=False)
class Logo:
    """The Manim logo with an X: ``circle``, ``square``, ``triangle`` and the
    ``m`` are Shape2Ds with the logo's exact colors (as on dark
    backgrounds); the X is two emissive Box slabs, so it glows (bloom
    works on lit meshes)."""

    root: Group
    circle: Shape2D
    square: Shape2D
    triangle: Shape2D
    m: Shape2D
    x: Group
    bars: list[Box]


def manimx_logo() -> Logo:
    """Built like ``manimx.examples.manim_logo`` in the logo's own plane
    (x right, y up), centered at the origin; :func:`stand_up` puts it on
    the stage."""
    circle = Circle(name="circle", color=LOGO_GREEN, fill_opacity=1.0).move_to((-1.0, 0.0, 0.0))
    square = Square(name="square", color=LOGO_BLUE, fill_opacity=1.0).move_to((0.0, 1.0, 0.0))
    triangle = Triangle(name="triangle", color=LOGO_RED, fill_opacity=1.0).move_to((1.0, 0.0, 0.0))
    m = Shape2D(
        np.load(M_GLYPH_POINTS),
        "m",
        color=LOGO_LIGHT,
        fill_opacity=1.0,
        stroke_width=0.0,
    )
    m.move_to((-2.25, 1.5, 0.0)).set_scale((7.0, 7.0, 1.0))
    bars = [
        Box(
            0.42,
            3.1,
            0.12,
            name=f"x-bar-{i}",
            color="#000000",
            emissive=X_GLOW,
            emissive_strength=0.0,
            cast_shadows=False,
        ).rotate(angle, (0, 0, 1))
        for i, angle in enumerate((math.pi / 5.5, -math.pi / 5.5))
    ]
    x = Group(*bars, name="x").move_to((3.35, 0.45, -0.1))
    shapes = Group(triangle, square, circle, m, name="manim")
    root = Group(shapes, x, name="manimx-logo")
    root.shift((-0.55, -0.6, 0.0))  # center the whole mark (the X widens it to the right)
    return Logo(root, circle, square, triangle, m, x, bars)


def stand_up(logo: Logo, center: Sequence[float] = tuple(LOGO_CENTER), scale: float = 1.0) -> Logo:
    """Stand the logo upright at ``center``, facing the audience (-y)."""
    holder = Group(logo.root, name="logo-stand").set_scale(scale).rotate(math.pi / 2, (1, 0, 0)).move_to(center)
    logo.root = holder
    return logo


__all__ = [
    "Aim",
    "Drive",
    "Logo",
    "MovingHead",
    "Stage",
    "build_stage",
    "fly",
    "hue",
    "manimx_logo",
    "on_floor",
    "rgb",
    "stand_up",
]
