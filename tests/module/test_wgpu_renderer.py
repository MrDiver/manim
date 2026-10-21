"""The wgpu renderer draws render object trees: 2D shapes, a lit mesh,
groups, and the frame held when nothing changed.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import quaternion

from manim import _native as native
from manim.renderer.protocol import (
    CapabilityException,
    RenderConfig,
    ToneMapOperator,
)
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


def edge_shades(frame: np.ndarray) -> int:
    """How many shades of red lie between the background and a white fill."""
    red = frame[..., 0].astype(int)
    return len(np.unique(red[(red > BACKGROUND[0]) & (red < 255)]))


def test_more_samples_give_smoother_edges_and_minus_one_takes_the_most():
    def frame(samples: int) -> np.ndarray:
        turned = native.RShape2D(
            Transform(rotation=quaternion.from_rotation_vector([0, 0, 0.3])),
            points=square_points(2.0),
            fill=native.Fill(color=[1.0, 1.0, 1.0], opacity=1.0),
            stroke=native.Stroke(width=0.0),
        )
        settings = RenderConfig(
            pixel_width=WIDTH,
            pixel_height=HEIGHT,
            background_color="#1E1E2E",
            msaa_samples=samples,
        )
        return native.WgpuRenderer().render(turned, settings)

    assert edge_shades(frame(1)) == 0
    assert edge_shades(frame(-1)) >= edge_shades(frame(4)) > 0


def test_a_sample_count_the_gpu_lacks_raises_a_capability_exception():
    settings = RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT, msaa_samples=3)
    with pytest.raises(CapabilityException, match="3 MSAA samples"):
        native.WgpuRenderer().render(scene(), settings)


def test_both_gpu_memory_choices_render_and_others_are_refused():
    for memory in ("compact", "fast"):
        settings = RenderConfig(
            pixel_width=WIDTH,
            pixel_height=HEIGHT,
            background_color="#1E1E2E",
            gpu_memory=memory,
        )
        frame = native.WgpuRenderer().render(scene(), settings)
        assert np.abs(frame[pixel(-3.5, 0.0)][:3] - [88, 196, 221]).max() <= 1
    with pytest.raises(ValueError, match="compact"):
        native.WgpuRenderer().render(scene(), RenderConfig(gpu_memory="huge"))


def test_the_renderer_reads_every_render_config_field():
    """A field the renderer never reads is a setting that does nothing."""

    class Recorder:
        """A RenderConfig that notes each field read, also through its
        properties (``background_rgba`` reads ``background_color``).
        """

        def __init__(self, config: RenderConfig):
            object.__setattr__(self, "config", config)
            object.__setattr__(self, "read", set())

        def __getattr__(self, name: str):
            attribute = getattr(type(self.config), name, None)
            if isinstance(attribute, property):
                return attribute.fget(self)
            self.read.add(name)
            return getattr(self.config, name)

    recorder = Recorder(RenderConfig(pixel_width=WIDTH, pixel_height=HEIGHT))
    native.WgpuRenderer().render(scene(), recorder)
    fields = {field.name for field in dataclasses.fields(RenderConfig)}
    assert fields - recorder.read == set()


def test_the_tone_map_changes_lit_meshes_and_leaves_2d_shapes():
    renderer = native.WgpuRenderer()

    def frame(tone_map: ToneMapOperator) -> np.ndarray:
        settings = RenderConfig(
            pixel_width=WIDTH,
            pixel_height=HEIGHT,
            background_color="#1E1E2E",
            tone_map=tone_map,
        )
        return renderer.render(scene(), settings)

    neutral, clamped = frame(ToneMapOperator.PBR_NEUTRAL), frame(ToneMapOperator.NONE)
    assert not np.array_equal(neutral[pixel(3.5, 0.0)], clamped[pixel(3.5, 0.0)])
    assert np.array_equal(neutral[pixel(-3.5, 0.0)], clamped[pixel(-3.5, 0.0)])
