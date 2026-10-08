"""A navigation badge: a map seen through a circle, with rings pulsing out.

A unit circle grows from nothing. Inside it, clipped to the circle, a
street map scrolls past a navigation arrow that stays in the middle. Rings
appear at the circle's edge and pulse outwards once, one after another.
Finally everything collapses to nothing and the scene is empty again.

What it exercises:

* GrowFromCenter / ShrinkToCenter on a group (the clip mask scales with
  its content, since both live in the group)
* a scene-side ClippingGroup (the map only shows inside the circle)
* continuous motion without animations: the map's y position is a
  derived signal of ``scene_time``, so the streets keep passing while
  other animations play
* a custom animation (:class:`PulseOut`): any class with ``targets`` and
  ``contribution`` can be placed on the timeline
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..animation import (
    Animation,
    Change,
    Circle,
    ClippingGroup,
    GrowFromCenter,
    Group,
    Mobject,
    Polyline,
    RootSignal,
    Scene,
    ShrinkToCenter,
    Target,
    derive,
)
from ..animation.rate_functions import ease_out_cubic, linear
from ..renderer.config import Camera, RenderConfig

BACKGROUND = "#070b1a"
MAP_GROUND = "#0f1c38"
STREET = "#2c4470"
MAIN_ROAD = "#4b6a9e"
ROUTE = "#3ee0ff"
ARROW = "#ffffff"
RIM = "#3ee0ff"

SCROLL_SPEED = 0.45
"""World units per second the streets move past the arrow."""


@dataclass(eq=False)
class PulseOut(Animation):
    """A ring that appears, expands to ``scale`` times its size and fades,
    all in one sweep, then leaves the scene."""

    mobject: Mobject
    scale: float = 1.8
    run_time: float = 1.4
    rate: Any = staticmethod(ease_out_cubic)

    def targets(self) -> Sequence[Target]:
        return [Target(self.mobject.alpha, Change.ABSOLUTE)] + [
            Target(axis, Change.RELATIVE) for axis in self.mobject.scale.axes[:2]
        ]

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> float:
        if signal is self.mobject.alpha:
            # Appear quickly, then fade out while expanding.
            return min(1.0, alpha * 6.0) * (1.0 - alpha)
        return self.scale**alpha

    def mobjects(self) -> Sequence[Mobject]:
        return (self.mobject,)

    def removes(self) -> Sequence[Mobject]:
        return (self.mobject,)


def street_map(length: float = 12.0) -> Group:
    """A strip of streets around a vertical route, from y = -1.5 upwards."""
    rng = np.random.default_rng(11)
    streets = []
    y = -1.5
    while y < length:  # side streets crossing the route at irregular gaps
        tilt = rng.uniform(-0.25, 0.25)
        streets.append(Polyline([(-1.6, y - tilt), (1.6, y + tilt)], stroke_color=STREET, stroke_width=5))
        y += rng.uniform(0.35, 0.7)
    for x in (-0.9, 0.85):  # parallel avenues with a kink now and then
        corners = [(x + rng.uniform(-0.1, 0.1), y) for y in np.arange(-1.5, length, 1.5)]
        streets.append(Polyline(corners, stroke_color=STREET, stroke_width=5))
    main_road = Polyline([(0.0, -1.5), (0.0, length)], stroke_color=MAIN_ROAD, stroke_width=14)
    route = Polyline([(0.0, -1.5), (0.0, length)], stroke_color=ROUTE, stroke_width=5)
    return Group(*streets, main_road, route, name="streets")


def navigation_arrow() -> Polyline:
    """A chevron pointing up, centered on the origin."""
    corners = [(0.0, 0.2), (0.16, -0.16), (0.0, -0.08), (-0.16, -0.16)]
    return Polyline(corners, closed=True, name="arrow", color=ARROW, fill_opacity=1.0, stroke_width=2)


class Navigation(Scene):
    def construct(self) -> None:
        ground = Circle(1.0, name="ground", fill_color=MAP_GROUND, fill_opacity=1.0, stroke_width=0)
        streets = street_map()
        # The streets pass by on their own: a derived signal of scene time,
        # not an animation, so it keeps running while anything else plays.
        streets.position.y = derive(lambda t: -SCROLL_SPEED * t, self.time.scene_time, name="streets.y")
        mask = Circle(1.0, name="mask", fill_opacity=1.0, stroke_width=0)
        map_view = ClippingGroup(mask, Group(ground, streets, name="map"), name="map-view")
        rim = Circle(1.0, name="rim", color=RIM, stroke_width=6)
        badge = Group(map_view, navigation_arrow(), rim, name="badge")

        self.wait(0.3)
        self.play(GrowFromCenter(badge), run_time=1.2)
        self.wait(1.0)

        rings = [Circle(1.0, name=f"ring-{i}", color=RIM, stroke_width=5) for i in range(3)]
        self.play(*(PulseOut(ring) for ring in rings), lag_ratio=0.3, rate=linear)
        self.wait(0.4)

        self.play(ShrinkToCenter(badge), run_time=0.9)
        self.wait(0.5)


def config(pixel_width: int = 1920, pixel_height: int = 1080) -> RenderConfig:
    """A close-up: 4.2 units vertically, room for the rings at 1.8x the
    badge's radius."""
    return RenderConfig(
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        background_color=BACKGROUND,
        camera=Camera(frame_height=4.2),
    )


__all__ = ["Navigation", "PulseOut", "config"]
