"""What the core (manimx._core) asks Python for while packing.

The core packs records per frame in Rust. A few records depend only on
geometry or style and change rarely: a mesh's interleaved vertices, a
material entry, a texture's mip levels. The core builds those once per
version of the signals they come from by calling these functions, and
keeps the result until a version changes. Fills are tessellated in Rust
(core/src/tessellation.rs), and stroke records are built in Rust
(core/src/strokes.rs) whenever the style values are plain numbers;
:func:`stroke_records` builds them for other values and is what the
Rust version is tested against. Every function returns plain bytes in
the record layouts below (they mirror core/src/records.rs and the
shaders).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..animation.nodes import style
from ..robjects import PbrMaterial, RDirectionalLight
from ..robjects.material import (
    DEFAULT_SURFACE_SHADER,  # noqa: F401 - read by the core (mesh_pass.rs)
    linear_rgb,
)
from ..robjects.mesh import MeshData
from ..robjects.transform import Transform, rotation_towards

POINTS_PER_CURVE = 4
TOLERANCE_FOR_POINT_EQUALITY = 1e-6
STROKE_WIDTH_SCALE = 0.01
"""World units per unit of stroke width (Manim: width 4 = 0.04 units)."""
NO_NEIGHBOR = -1

# Mirror the structs of core/src/records.rs (and the shaders reading them).
STROKE_POINT_DTYPE = np.dtype([("position", "<f4", 3), ("width", "<f4"), ("color", "<f4", 4)])
STROKE_CURVE_DTYPE = np.dtype(
    [
        ("first_point", "<u4"),
        ("prev_point", "<i4"),
        ("next_point", "<i4"),
        ("start_joint", "<u4"),
        ("end_joint", "<u4"),
        ("start_cap", "<u4"),
        ("end_cap", "<u4"),
        ("_pad", "<u4"),
    ]
)
MESH_VERTEX_DTYPE = np.dtype([("position", "<f4", 3), ("u", "<f4"), ("normal", "<f4", 3), ("v", "<f4"), ("tangent", "<f4", 4)])
MATERIAL_DTYPE = np.dtype(
    [
        ("base_color", "<f4", 4),
        ("emissive", "<f4", 4),
        ("metallic", "<f4"),
        ("roughness", "<f4"),
        ("normal_scale", "<f4"),
        ("occlusion_strength", "<f4"),
    ]
)


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
    visible = bool(np.any(stroke.width > 0.0) and np.any(stroke.opacity > 0.0))
    if not visible:
        return b"", b"", False, False
    curves = _link_curves(points, [j.value for j in stroke.joints(len(points) // POINTS_PER_CURVE)], stroke.cap.value)
    records = np.zeros(len(points), STROKE_POINT_DTYPE)
    records["position"] = points if points.shape[1] == 3 else np.column_stack([points, np.zeros(len(points))])
    records["width"] = stroke.widths(len(points)) * STROKE_WIDTH_SCALE
    records["color"] = stroke.rgba(len(points))
    # A resampled per-point style list assigns slightly different values
    # to the end anchor of one curve and the start anchor of the next
    # (separate entries in the point array). Averaging them avoids a
    # visible step at every joint. Every anchor is in at most one pair.
    linked = curves["next_point"] != NO_NEIGHBOR
    ends = (curves["first_point"][linked] + POINTS_PER_CURVE - 1).astype(np.intp)
    starts = curves["next_point"][linked].astype(np.intp)
    for name in ("width", "color"):
        values = records[name]
        mean = (values[ends] + values[starts]) / 2
        values[ends] = mean
        values[starts] = mean
    return curves.tobytes(), records.tobytes(), True, bool(np.all(stroke.opacity >= 1.0))


def _link_curves(points: np.ndarray, joints: list[int], cap: int) -> np.ndarray:
    """Stroke curve records with previous/next links inside each subpath
    (a subpath continues while a curve starts where the last one ended).
    Closed subpaths link their last curve back to the first one, so the
    closing corner gets a joint instead of two caps.
    """
    count = len(points) // POINTS_PER_CURVE
    curves = np.zeros(count, STROKE_CURVE_DTYPE)
    starts = POINTS_PER_CURVE * np.arange(count)
    curves["first_point"] = starts
    curves["end_joint"] = joints
    curves["start_cap"] = cap
    curves["end_cap"] = cap
    curves["prev_point"] = NO_NEIGHBOR
    curves["next_point"] = NO_NEIGHBOR
    corners = points.reshape(count, POINTS_PER_CURVE, -1)
    gaps = np.linalg.norm(corners[1:, 0] - corners[:-1, -1], axis=1)
    breaks = [0, *(np.flatnonzero(gaps > TOLERANCE_FOR_POINT_EQUALITY) + 1).tolist(), count]
    for begin, end in zip(breaks[:-1], breaks[1:], strict=True):
        for curve in range(begin, end):
            if curve > begin:
                curves["prev_point"][curve] = starts[curve - 1]
                curves["start_joint"][curve] = joints[curve - 1]
            if curve < end - 1:
                curves["next_point"][curve] = starts[curve + 1]
        if np.linalg.norm(corners[begin, 0] - corners[end - 1, -1]) <= TOLERANCE_FOR_POINT_EQUALITY:
            curves["prev_point"][begin] = starts[end - 1]
            curves["start_joint"][begin] = joints[end - 1]
            curves["next_point"][end - 1] = starts[begin]
    return curves


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


def _mip_chain(image: np.ndarray) -> list[np.ndarray]:
    """The image and successive half-size versions down to 1x1."""
    levels = [image]
    current = Image.fromarray(image)
    while current.width > 1 or current.height > 1:
        current = current.resize((max(1, current.width // 2), max(1, current.height // 2)), Image.Resampling.BOX)
        levels.append(np.asarray(current))
    return levels


def default_texture_levels(rgba: tuple[int, int, int, int]) -> list[tuple[int, int, bytes]]:
    """A 1x1 texture of one color (missing material textures)."""
    return [(1, 1, bytes(rgba))]


def _box(points: np.ndarray) -> np.ndarray:
    """The 8 corners of the axis-aligned box around ``points``."""
    low, high = points.min(axis=0), points.max(axis=0)
    return np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])


def _interleave(mesh: MeshData) -> np.ndarray:
    """A mesh's vertices as MESH_VERTEX_DTYPE records (tangents computed
    when it has none)."""
    mesh = mesh.with_tangents()
    vertices = np.zeros(mesh.vertex_count, MESH_VERTEX_DTYPE)
    vertices["position"] = mesh.positions
    vertices["normal"] = mesh.normals
    if mesh.uvs is not None:
        vertices["u"], vertices["v"] = mesh.uvs[:, 0], mesh.uvs[:, 1]
    if mesh.tangents is not None:
        vertices["tangent"] = mesh.tangents
    return vertices


def _material_entry(material: PbrMaterial) -> np.ndarray:
    entry = np.zeros((), MATERIAL_DTYPE)
    entry["base_color"] = material.base_color_linear
    entry["emissive"][:3] = material.emissive_linear
    entry["metallic"] = material.metallic
    entry["roughness"] = material.roughness
    entry["normal_scale"] = material.normal_scale
    entry["occlusion_strength"] = material.occlusion_strength
    return entry
