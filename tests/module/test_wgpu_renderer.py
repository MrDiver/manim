"""The wgpu renderer draws render object trees: 2D shapes, a lit mesh,
groups, and the frame held when nothing changed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import quaternion

from manim import _native as native
from manim.renderer.protocol import CapabilityException, RenderConfig
from manim.utils.transform import Transform

WIDTH, HEIGHT = 320, 180
BACKGROUND = (30, 30, 46)


def square_points(half: float) -> np.ndarray:
    """A square around the origin as cubic Bezier curves (straight lines)."""
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    points = []
    for index in range(4):
        a, b = np.array(corners[index]), np.array(corners[(index + 1) % 4])
        points += [a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]
    return np.array(points)


def box_mesh(half: float) -> native.MeshData:
    """A cube around the origin, four vertices and a normal per face."""
    positions, normals, indices = [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            u = np.roll(np.eye(3), 1, axis=0)[axis]
            v = np.roll(np.eye(3), 2, axis=0)[axis]
            if sign < 0:
                u, v = v, u
            first = len(positions)
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                positions.append(half * (normal + du * u + dv * v))
                normals.append(normal)
            indices += [first, first + 1, first + 2, first, first + 2, first + 3]
    return native.MeshData(np.array(positions), np.array(indices), np.array(normals))


def scene(square_x: float = -3.5) -> native.RGroup:
    square = native.RShape2D(
        Transform(position=(square_x, 0.0, 0.0)),
        uid=1,
        points=square_points(1.2),
        fill=native.Fill(color=[0.345, 0.769, 0.867], opacity=1.0),
        stroke=native.Stroke(width=0.0),
    )
    box = native.RShape3D(
        Transform(
            position=(3.5, 0.0, 0.0),
            rotation=quaternion.from_rotation_vector([0.6, 0.8, 0.0]),
        ),
        uid=2,
        mesh=box_mesh(1.0),
        material=native.PbrMaterial(base_color=[0.514, 0.757, 0.404]),
    )
    return native.RGroup(uid=3, children=[square, box])


def config() -> RenderConfig:
    return RenderConfig(
        pixel_width=WIDTH,
        pixel_height=HEIGHT,
        background_color="#1E1E2E",
    )


def pixel(x: float, y: float) -> tuple[int, int]:
    """The pixel `(row, column)` of the scene point `(x, y)` under the
    default camera (8 units high, centered on the origin).
    """
    scale = HEIGHT / 8
    return int(HEIGHT / 2 - y * scale), int(WIDTH / 2 + x * scale)


def test_a_frame_shows_shapes_and_a_lit_mesh_over_the_background():
    frame = native.WgpuRenderer().render(scene(), config())
    assert frame.shape == (HEIGHT, WIDTH, 4)
    assert frame.dtype == np.uint8
    assert tuple(frame[pixel(0.0, 3.5)][:3]) == BACKGROUND
    # The square's fill: sRGB as given.
    assert np.abs(frame[pixel(-3.5, 0.0)][:3] - [88, 196, 221]).max() <= 1
    # The box is lit: neither background nor black.
    box = frame[pixel(3.5, 0.0)][:3].astype(int)
    assert tuple(box) != BACKGROUND
    assert box.sum() > 0
    assert (frame[..., 3] == 255).all()


def test_an_unchanged_tree_gives_the_same_frame_and_a_moved_shape_moves():
    renderer = native.WgpuRenderer()
    first = renderer.render(scene(), config())
    assert np.array_equal(renderer.render(scene(), config()), first)
    moved = renderer.render(scene(square_x=-1.0), config())
    assert tuple(moved[pixel(-3.5, 0.0)][:3]) == BACKGROUND
    assert np.abs(moved[pixel(-1.0, 0.0)][:3] - [88, 196, 221]).max() <= 1


def test_a_new_background_renders_the_scene_again():
    renderer = native.WgpuRenderer()
    renderer.render(scene(), config())
    red = RenderConfig(
        pixel_width=WIDTH, pixel_height=HEIGHT, background_color="#FF0000"
    )
    frame = renderer.render(scene(), red)
    assert tuple(frame[pixel(0.0, 3.5)][:3]) == (255, 0, 0)
    assert np.abs(frame[pixel(-3.5, 0.0)][:3] - [88, 196, 221]).max() <= 1


def test_a_rotated_group_turns_what_it_holds():
    turned = native.RGroup(
        Transform(rotation=quaternion.from_rotation_vector([0, 0, math.pi])),
        children=[scene()],
    )
    frame = native.WgpuRenderer().render(turned, config())
    # Turned half around: the square is now on the right.
    assert np.abs(frame[pixel(3.5, 0.0)][:3] - [88, 196, 221]).max() <= 1


def test_what_the_renderer_cannot_draw_raises_a_capability_exception():
    mesh_in_mask = native.RClippingGroup(
        mask=native.RShape3D(mesh=box_mesh(1.0)),
        content=native.RShape2D(points=square_points(1.0)),
    )
    with pytest.raises(CapabilityException, match="part of a mask"):
        native.WgpuRenderer().render(mesh_in_mask, config())
