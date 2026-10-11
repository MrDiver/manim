"""What the core (manimx._core, see CORE.md) asks Python for while packing.

The core packs records per frame in Rust. A few records depend only on
geometry or style and change rarely: a stroke's curve links, a mesh's
interleaved vertices, a material entry. The core builds those once per
version of the signals they come from by calling these functions, which
reuse the reference implementation (so both paths produce the same
bytes), and keeps the result until a version changes. Fills are
tessellated in Rust for both (core/src/tessellation.rs), and the core
builds stroke records itself (core/src/strokes.rs) whenever the style
values are plain numbers; :func:`stroke_records` is the reference it is
tested against and the fallback for other values. Every function
returns plain bytes in the record layouts of vector_batch.py and
mesh_batch.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..animation import style
from ..robjects import PbrMaterial, RCurve3D, RDirectionalLight, RShape2D
from ..robjects.material import (
    DEFAULT_SURFACE_SHADER,  # noqa: F401 - read by the core (mesh_pass.rs)
    linear_rgb,
)
from ..robjects.mesh import MeshData
from ..robjects.transform import Transform, rotation_towards
from .webgpu.mesh_batch import _interleave, _material_entry
from .webgpu.mesh_pass import _mip_chain
from .webgpu.shadow_views import _box
from .webgpu.vector_batch import _StrokeTopology


def fill_colors(color: np.ndarray, opacity: np.ndarray, count: int) -> tuple[bytes, bool, bool]:
    """Per-point fill RGBA (float32), whether any of it is visible, and
    whether all of it is opaque.
    """
    fill = style.fill(color, opacity)
    return fill.rgba(count).astype(np.float32).tobytes(), bool(np.any(fill.opacity > 0.0)), bool(np.all(fill.opacity >= 1.0))


def stroke_records(points: np.ndarray, color: np.ndarray, opacity: np.ndarray, width: np.ndarray, joint: Any, cap: Any) -> tuple[bytes, bytes, bool, bool]:
    """A path's stroke curves (local point indices) and points, whether it
    is visible, and whether it is opaque.
    """
    stroke = style.stroke(color, opacity, width, joint, cap)
    shape = (RShape2D if points.shape[1] == 2 else RCurve3D)(points=points, stroke=stroke)
    visible = bool(np.any(stroke.width > 0.0) and np.any(stroke.opacity > 0.0))
    if not visible:
        return b"", b"", False, False
    topology = _StrokeTopology.build(shape)
    return topology.curves.tobytes(), topology.points(shape).tobytes(), True, bool(np.all(stroke.opacity >= 1.0))


def path_center(points: np.ndarray) -> tuple[float, float, float]:
    """The mean of a path's points in 3D (its distance to the camera sorts it)."""
    points_3d = points if points.shape[1] == 3 else np.column_stack([points, np.zeros(len(points))])
    x, y, z = points_3d.mean(axis=0)
    return float(x), float(y), float(z)


def path_box(points: np.ndarray) -> bytes:
    """The 8 corners of the box around a path's points (float64)."""
    points_3d = points if points.shape[1] == 3 else np.column_stack([points, np.zeros(len(points))])
    return _box(points_3d).astype(np.float64).tobytes()


def mesh_records(mesh: MeshData) -> tuple[bytes, bytes, int, bytes]:
    """A mesh's interleaved vertices (MESH_VERTEX_DTYPE), its indices
    (uint32), its triangle count and the corners of its box (float64).
    """
    if mesh.triangle_count == 0:
        return b"", b"", 0, b""
    return _interleave(mesh).tobytes(), np.ascontiguousarray(mesh.indices, np.uint32).tobytes(), mesh.triangle_count, _box(mesh.positions).astype(np.float64).tobytes()


def material_entry(*values: Any) -> bytes:
    """The MATERIAL_DTYPE entry of a Shape3D's material signal values (in
    the order of ``Shape3D._material_signals()``), clamped as the
    snapshot clamps them.
    """
    base_color, opacity, metallic, roughness, emissive, strength, normal_scale, occlusion, *textures, _surface_shader = values
    material = PbrMaterial(
        base_color=np.array(base_color),
        opacity=_unit(opacity),
        metallic=_unit(metallic),
        roughness=_unit(roughness),
        emissive=np.array(emissive),
        emissive_strength=max(0.0, strength),
        normal_scale=normal_scale,
        occlusion_strength=_unit(occlusion),
        base_color_texture=textures[0],
        metallic_roughness_texture=textures[1],
        normal_texture=textures[2],
        occlusion_texture=textures[3],
        emissive_texture=textures[4],
    )
    return _material_entry(material).tobytes()


def light_color(color: tuple[float, ...]) -> tuple[float, float, float]:
    """A light's color in linear RGB."""
    return linear_rgb(np.array(color))


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def default_light(camera_matrix: np.ndarray) -> tuple[list[float], tuple[float, float, float], float, float, bool]:
    """The key light of a scene without lights (``default_lights``): its
    world matrix (row-major), linear color, intensity, size and whether it
    casts shadows.
    """
    right, up, backwards = camera_matrix[:3, 0], camera_matrix[:3, 1], camera_matrix[:3, 2]
    direction = -(1.0 * up - 0.6 * right + 0.8 * backwards)
    light = RDirectionalLight(transform=Transform(rotation=rotation_towards(tuple(direction))), intensity=3.0)
    return light.transform.matrix().reshape(-1).tolist(), linear_rgb(light.color), light.intensity, light.size, light.cast_shadows


def texture_levels(texture: Any) -> list[tuple[int, int, bytes]]:
    """A material texture's image and its smaller mip levels down to 1x1,
    as ``(width, height, RGBA bytes)`` (the core's renderer uploads them).
    """
    return [(level.shape[1], level.shape[0], np.ascontiguousarray(level).tobytes()) for level in _mip_chain(texture.image)]


def default_texture_levels(rgba: tuple[int, int, int, int]) -> list[tuple[int, int, bytes]]:
    """A 1x1 texture of one color (missing material textures)."""
    return [(1, 1, bytes(rgba))]
