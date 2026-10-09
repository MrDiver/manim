"""A 3D test scene in the style of Blender's default viewport.

World ``+z`` is up, like Blender and Manim's 3D scenes. The scene contains

* a ground grid on the XY plane with Blender-colored X (red) and Y (green)
  axes and a vertical Z axis (blue),
* the Manim logo standing upright in the middle, turned about ``z``,
* the renderer's stress-test shapes (joints, caps, per-point styles,
  transparency, clipping, a mirrored back side) placed around it in
  different orientations.

It exists to check cameras, 3D transforms and depth: render it with each
camera from :func:`views` and every view must look plausible.
"""

from __future__ import annotations

import math

import numpy as np

from ..animation import ClippingGroup, Group, Shape2D
from ..animation.geometry import arc_points, rectangle_points, regular_polygon_points
from ..robjects import CapStyle, JointType, RShape2D, Transform
from ..renderer.config import Camera, RenderConfig
from ..renderer.scene_graph import bounding_box
from ..utils.color import BLUE, GREEN, ORANGE, PURPLE, RED, WHITE, YELLOW
from ._nodes import placed
from .manim_logo import manim_logo

BACKGROUND = "#3d3d3d"  # Blender's viewport gray
GRID_COLOR = "#6b6b6b"
AXIS_X = "#ff3352"
AXIS_Y = "#8bdc00"
AXIS_Z = "#2890ff"

GRID_EXTENT = 10
TO_UPRIGHT = (1.0, 0.0, 0.0)
"""Rotating a shape by 90 degrees about x stands its XY plane up (local y -> world z)."""


def blender_scene() -> Group:
    return Group(_ground(), _upright_logo(), _stress_ring(), name="blender-scene")


def views(pixel_width: int = 1920, pixel_height: int = 1080) -> dict[str, RenderConfig]:
    """Named cameras around the scene."""
    perspective = {"field_of_view": math.radians(40.0), "near": 0.5, "far": 200.0}
    cameras = {
        # Blender's default camera direction, pulled back to fit the scene.
        "default": Camera.look_at((16.5, -15.5, 11.0), (0.0, 0.0, 2.0), **perspective),
        "front": Camera.look_at((0.0, -24.0, 4.0), (0.0, 0.0, 3.0), **perspective),
        "side": Camera.look_at((24.0, 0.0, 6.0), (0.0, 0.0, 3.0), **perspective),
        "top": Camera.look_at((0.0, 0.0, 26.0), up=(0.0, 1.0, 0.0), **perspective),
        "below": Camera.look_at((10.0, 12.0, -9.0), (0.0, 0.0, 1.0), **perspective),
        "isometric": Camera.look_at((20.0, -20.0, 20.0), frame_height=18.0, near=1.0, far=100.0),
    }
    return {
        name: RenderConfig(pixel_width=pixel_width, pixel_height=pixel_height, background_color=BACKGROUND, camera=camera)
        for name, camera in cameras.items()
    }


def _corners(corners: list[tuple[float, float]]) -> np.ndarray:
    return RShape2D.from_corners(corners).points


def _ground() -> Group:
    """Grid lines and axes on the XY plane, plus the vertical Z axis."""
    coordinates = [c for c in range(-GRID_EXTENT, GRID_EXTENT + 1) if c != 0]
    lines = [((c, -GRID_EXTENT), (c, GRID_EXTENT)) for c in coordinates]
    lines += [((-GRID_EXTENT, c), (GRID_EXTENT, c)) for c in coordinates]
    # One shape, one open subpath per grid line.
    grid = Shape2D(np.vstack([_corners(line) for line in lines]), "grid", stroke_color=GRID_COLOR, stroke_width=2.0)
    extent = (-GRID_EXTENT, GRID_EXTENT)
    x_axis = Shape2D(_corners([(extent[0], 0), (extent[1], 0)]), "x-axis", stroke_color=AXIS_X, stroke_width=3.0)
    y_axis = Shape2D(_corners([(0, extent[0]), (0, extent[1])]), "y-axis", stroke_color=AXIS_Y, stroke_width=3.0)
    z_axis = placed(
        Shape2D(_corners([(0, 0), (0, 8)]), "z-axis", stroke_color=AXIS_Z, stroke_width=3.0),
        Transform().rotated(math.pi / 2, TO_UPRIGHT),
    )
    return Group(grid, x_axis, y_axis, z_axis, name="ground")


def _upright_logo() -> Group:
    """The logo standing on the ground at the origin, turned towards the default camera."""
    logo = manim_logo()
    low, high = bounding_box(logo.snapshot())
    height = high[1] - low[1]
    scale = 1.6
    placement = Transform(position=(0.0, 0.0, scale * height / 2 + 0.2), scale=(scale, scale, scale))
    return placed(Group(logo, name="upright-logo"), placement.rotated(math.pi / 2, TO_UPRIGHT).rotated(math.radians(30.0), (0, 0, 1)))


def _stress_ring() -> Group:
    """Stress-test shapes on a circle around the logo, each oriented differently."""
    zigzag = Group(
        *(
            placed(
                Shape2D(
                    _corners([(-1.0, -0.5), (-0.5, 0.5), (0.0, -0.5), (0.5, 0.5), (1.0, -0.5)]),
                    f"zigzag-{joint.name.lower()}",
                    stroke_color=WHITE,
                    stroke_width=20,
                    stroke_joint=joint,
                    stroke_cap=cap,
                ),
                Transform(position=(0.0, 1.5 - 1.5 * i, 0.0)),
            )
            for i, (joint, cap) in enumerate(
                [(JointType.MITER, CapStyle.BUTT), (JointType.ROUND, CapStyle.ROUND), (JointType.BEVEL, CapStyle.SQUARE)]
            )
        ),
        name="zigzags",
    )
    rainbow = Shape2D(arc_points(1.4), "rainbow", stroke_color=[RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED], stroke_width=[6, 30, 6, 30, 6])
    pentagon = Shape2D(regular_polygon_points(5, 1.5), "pentagon", fill_color=[RED, BLUE], fill_opacity=0.7, stroke_color=YELLOW, stroke_width=10, stroke_opacity=0.6)
    squares = Group(
        placed(Shape2D(rectangle_points(2.0, 2.0), "red-square", fill_color=RED, fill_opacity=0.5, stroke_color=WHITE, stroke_width=6), Transform(position=(-0.5, 0.5, 0.0))),
        placed(Shape2D(rectangle_points(2.0, 2.0), "blue-square", fill_color=BLUE, fill_opacity=0.5, stroke_color=WHITE, stroke_width=6), Transform(position=(0.5, -0.5, 0.0))),
        name="squares",
    )
    clipped = ClippingGroup(
        Shape2D(regular_polygon_points(6, 1.5), "hexagon-mask", fill_opacity=1.0, stroke_width=0),
        Group(
            *(
                placed(
                    Shape2D(rectangle_points(0.5, 0.5), f"stripe-{i}", fill_color=color, fill_opacity=1.0, stroke_width=0),
                    Transform(position=(-1.75 + 0.5 * i, 0.0, 0.0), scale=(1.0, 7.0, 1.0)),
                )
                for i, color in enumerate([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, RED, ORANGE])
            ),
            name="stripes",
        ),
        "clipped",
    )
    mirrored = Group(
        Shape2D(arc_points(), "green-disc", fill_color=GREEN, fill_opacity=1.0, stroke_color=WHITE, stroke_width=8),
        Shape2D(rectangle_points(0.8, 0.8), "red-square", fill_color=RED, fill_opacity=1.0, stroke_width=0),
        name="mirrored",
    )

    # (shape, angle on the ring, extra orientation applied before placement)
    placements = [
        (zigzag, 0.0, Transform().rotated(math.pi / 2, TO_UPRIGHT)),
        (rainbow, 60.0, Transform().rotated(math.pi / 2, TO_UPRIGHT).rotated(math.radians(40), (0, 0, 1))),
        (pentagon, 120.0, Transform(position=(0.0, 0.0, -1.3))),  # lying flat
        (squares, 180.0, Transform().rotated(math.radians(60), TO_UPRIGHT)),  # leaning back
        (clipped, 240.0, Transform().rotated(math.pi / 2, TO_UPRIGHT).rotated(math.radians(-70), (0, 0, 1))),
        (mirrored, 300.0, Transform().rotated(math.pi / 2, TO_UPRIGHT).rotated(math.pi, (0, 0, 1))),  # back side to the center
    ]
    radius = 6.5
    children = []
    for shape, degrees, orientation in placements:
        angle = math.radians(degrees)
        position = (radius * math.cos(angle), radius * math.sin(angle), 1.8)
        children.append(
            placed(shape, Transform(position=tuple(np.add(position, orientation.position)), rotation=orientation.rotation, scale=orientation.scale))
        )
    return Group(*children, name="stress-ring")
