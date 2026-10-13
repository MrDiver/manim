"""PortalOrbit: the camera circles the portal cube.

The scene's camera (``self.camera``) is animated like any node: ``Orbit``
below is an ordinary animation that moves it on a circle around a point
while keeping it aimed at that point. The cube's worlds bring their own
lights; the torus in the right world spins while the camera passes.

Render it alone with ``manimx render PortalOrbit``, or the whole timeline with
``manimx render``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from manimx.animation import Animation, Camera, Change, Scene, Target
from manimx.examples.portal_cube import CENTER, portal_cube
from manimx.renderer import Camera as RenderCamera
from manimx.userland import TAU, Rotate
from manimx.userland.rate_functions import linear, smooth


@dataclass(eq=False)
class Orbit(Animation):
    """Move ``camera`` on a sphere around ``center``, looking at it.

    Azimuth and elevation (degrees) and the distance go from their first to
    their second value; azimuth 0 looks from +x, -90 from -y (the front).
    """

    camera: Camera
    azimuth: tuple[float, float]
    elevation: tuple[float, float] = (22.0, 22.0)
    distance: tuple[float, float] = (11.0, 11.0)
    center: Sequence[float] = tuple(CENTER)
    run_time: float = 1.0
    rate: Any = smooth

    def targets(self) -> Sequence[Target]:
        axes = self.camera.position.axes
        return [*(Target(axis, Change.ABSOLUTE) for axis in axes), Target(self.camera.rotation, Change.ABSOLUTE)]

    def contributions(self, alpha: float, start_values: Sequence[Any]) -> Sequence[Any]:
        eye = self.eye(alpha)
        return [*(float(value) for value in eye), RenderCamera.look_at(tuple(eye), tuple(self.center)).transform.rotation]

    def eye(self, alpha: float) -> np.ndarray:
        def at(pair: tuple[float, float]) -> float:
            return pair[0] + (pair[1] - pair[0]) * alpha

        a, e = math.radians(at(self.azimuth)), math.radians(at(self.elevation))
        direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        return np.asarray(self.center) + at(self.distance) * direction


class PortalOrbit(Scene):
    def construct(self) -> None:
        cube = portal_cube()
        self.add(cube)
        self.camera.set_field_of_view(math.radians(40))
        self.camera.near = 0.3
        self.play(Orbit(self.camera, azimuth=(-90, -90), run_time=0))  # start in front of the cube

        self.wait(1)
        # Once around; the torus in the right world spins meanwhile.
        self.play(
            Orbit(self.camera, azimuth=(-90, 270), run_time=10, rate=linear),
            Rotate(cube.find("torus"), TAU, axis=(0, 0, 1), run_time=10, rate=linear),
        )
        # Rise and come closer to the corner where two worlds meet.
        self.play(Orbit(self.camera, azimuth=(270, 315), elevation=(22, 35), distance=(11, 10), run_time=2.5))
        self.wait(1.5)
