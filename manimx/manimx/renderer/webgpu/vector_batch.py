"""Pack all Shape2D draws of a frame into shared GPU-ready arrays.

This module is pure numpy: it builds the "super buffers" that ``vector.wgsl``
reads and the indirect draw commands that index into them. Keeping it free of
GPU calls makes the data layout testable on its own.

Buffers (dtypes mirror the WGSL structs one to one):

* ``objects``        one entry per Shape2D: model matrix and depth layer
* ``instances``      what each draw instance generates (fill or stroke curve)
* ``fill_vertices``  all tessellated fill triangles, concatenated
* ``stroke_points``  all control points with per-point stroke width / color
* ``stroke_curves``  one entry per cubic curve with neighbour links for joints

Draws are split into an opaque list (any order, depth test resolves overlap)
and a transparent list sorted back to front.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...objects import Shape2D
from ...objects.shape2d import POINTS_PER_CURVE, TOLERANCE_FOR_POINT_EQUALITY
from ..cache import MobjectDataCache
from ..scene_graph import DrawItem
from ..tessellation import FillMesh, tessellate_fill

STROKE_WIDTH_SCALE = 0.01
"""World units per unit of stroke width (Manim: width 4 = 0.04 units)."""

KIND_FILL = 0
KIND_STROKE = 1
NO_NEIGHBOR = -1

# Keep in sync with STROKE_VERTEX_COUNT in vector.wgsl:
# SEGMENTS * 6 ribbon vertices + 2 fans of FAN_TRIANGLES * 3 vertices.
STROKE_VERTEX_COUNT = 24 * 6 + 2 * 8 * 3

OBJECT_DTYPE = np.dtype(
    [("model", "<f4", (4, 4)), ("layer", "<f4"), ("_pad", "<f4", 3)]
)
INSTANCE_DTYPE = np.dtype(
    [("kind", "<u4"), ("object", "<u4"), ("curve", "<u4"), ("_pad", "<u4")]
)
FILL_VERTEX_DTYPE = np.dtype(
    [("position", "<f4", 2), ("uv", "<f4", 2), ("color", "<f4", 4), ("side", "<f4"), ("_pad", "<f4", 3)]
)
STROKE_POINT_DTYPE = np.dtype(
    [("position", "<f4", 2), ("width", "<f4"), ("_pad", "<f4"), ("color", "<f4", 4)]
)
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
# WebGPU's DrawIndirectArgs.
DRAW_INDIRECT_DTYPE = np.dtype(
    [("vertex_count", "<u4"), ("instance_count", "<u4"), ("first_vertex", "<u4"), ("first_instance", "<u4")]
)

assert OBJECT_DTYPE.itemsize == 80
assert INSTANCE_DTYPE.itemsize == 16
assert FILL_VERTEX_DTYPE.itemsize == 48
assert STROKE_POINT_DTYPE.itemsize == 32
assert STROKE_CURVE_DTYPE.itemsize == 32
assert DRAW_INDIRECT_DTYPE.itemsize == 16


@dataclass(slots=True)
class _Draw:
    """One indirect draw before encoding, with what it needs for sorting."""

    args: tuple[int, int, int, int]
    opaque: bool
    camera_distance: float
    layer: int


@dataclass(slots=True)
class VectorBatch:
    objects: np.ndarray
    instances: np.ndarray
    fill_vertices: np.ndarray
    stroke_points: np.ndarray
    stroke_curves: np.ndarray
    opaque_draws: np.ndarray
    transparent_draws: np.ndarray

    @staticmethod
    def build(
        items: list[DrawItem], cache: MobjectDataCache, camera_position: np.ndarray
    ) -> VectorBatch:
        builder = _BatchBuilder(cache, camera_position)
        for item in items:
            if isinstance(item.obj, Shape2D):
                builder.add(item.obj, item.world, item.order)
        return builder.finish()


@dataclass(slots=True)
class _BatchBuilder:
    cache: MobjectDataCache
    camera_position: np.ndarray
    objects: list[np.ndarray] = field(default_factory=list)
    instances: list[tuple[int, int, int, int]] = field(default_factory=list)
    fill_meshes: list[FillMesh] = field(default_factory=list)
    stroke_points: list[np.ndarray] = field(default_factory=list)
    stroke_curves: list[np.ndarray] = field(default_factory=list)
    draws: list[_Draw] = field(default_factory=list)
    fill_vertex_count: int = 0
    stroke_point_count: int = 0
    stroke_curve_count: int = 0

    def add(self, shape: Shape2D, world: np.ndarray, order: int) -> None:
        if shape.num_curves == 0:
            return
        object_index = len(self.objects)
        # Two layers per object: the fill at 2 * order, its stroke one above.
        layer = 2 * order
        entry = np.zeros((), OBJECT_DTYPE)
        entry["model"] = world.T  # WGSL matrices are column-major
        entry["layer"] = layer
        self.objects.append(entry)

        center = world @ np.append(shape.points.mean(axis=0), [0.0, 1.0])
        distance = float(np.linalg.norm(center[:3] - self.camera_position))
        self._add_fill(shape, object_index, distance, layer)
        self._add_stroke(shape, object_index, distance, layer + 1)

    # -- fill ---------------------------------------------------------------

    def _add_fill(self, shape: Shape2D, object_index: int, distance: float, layer: int) -> None:
        opacity = shape.fill.opacity
        if not np.any(opacity > 0.0):
            return
        mesh = self.cache.get_or_create(
            shape.uid,
            "fill-mesh",
            shape.version("points", "fill"),
            lambda: tessellate_fill(shape.points, shape.fill_rgba()),
        )
        if len(mesh) == 0:
            return
        first_instance = self._add_instance(KIND_FILL, object_index)
        self.draws.append(
            _Draw(
                args=(len(mesh), 1, self.fill_vertex_count, first_instance),
                opaque=bool(np.all(opacity >= 1.0)),
                camera_distance=distance,
                layer=layer,
            )
        )
        self.fill_meshes.append(mesh)
        self.fill_vertex_count += len(mesh)

    # -- stroke -------------------------------------------------------------

    def _add_stroke(self, shape: Shape2D, object_index: int, distance: float, layer: int) -> None:
        stroke = shape.stroke
        if not (np.any(stroke.width > 0.0) and np.any(stroke.opacity > 0.0)):
            return
        first_point = self.stroke_point_count
        first_curve = self.stroke_curve_count

        points = np.zeros(len(shape.points), STROKE_POINT_DTYPE)
        points["position"] = shape.points
        points["width"] = shape.stroke_widths() * STROKE_WIDTH_SCALE
        points["color"] = shape.stroke_rgba()
        curves = _link_curves(shape, first_point)
        _match_joint_styles(points, curves, first_point)

        first_instance = len(self.instances)
        for curve in range(shape.num_curves):
            self._add_instance(KIND_STROKE, object_index, first_curve + curve)
        self.draws.append(
            _Draw(
                args=(STROKE_VERTEX_COUNT, shape.num_curves, 0, first_instance),
                opaque=bool(np.all(stroke.opacity >= 1.0)),
                camera_distance=distance,
                layer=layer,
            )
        )
        self.stroke_points.append(points)
        self.stroke_curves.append(curves)
        self.stroke_point_count += len(points)
        self.stroke_curve_count += len(curves)

    def _add_instance(self, kind: int, object_index: int, curve: int = 0) -> int:
        self.instances.append((kind, object_index, curve, 0))
        return len(self.instances) - 1

    # -- encoding -----------------------------------------------------------

    def finish(self) -> VectorBatch:
        opaque = [d for d in self.draws if d.opaque]
        # Transparent draws: farthest first; coplanar shapes in painter's order.
        transparent = sorted(
            (d for d in self.draws if not d.opaque),
            key=lambda d: (-round(d.camera_distance, 6), d.layer),
        )
        return VectorBatch(
            objects=_concat(self.objects, OBJECT_DTYPE),
            instances=np.array(self.instances, dtype=np.uint32).view(INSTANCE_DTYPE).reshape(-1)
            if self.instances
            else np.zeros(0, INSTANCE_DTYPE),
            fill_vertices=_fill_vertices(self.fill_meshes),
            stroke_points=_concat(self.stroke_points, STROKE_POINT_DTYPE),
            stroke_curves=_concat(self.stroke_curves, STROKE_CURVE_DTYPE),
            opaque_draws=_draw_args(opaque),
            transparent_draws=_draw_args(transparent),
        )


def _link_curves(shape: Shape2D, first_point: int) -> np.ndarray:
    """Stroke curve records with previous/next links inside each subpath.

    Closed subpaths link their last curve back to the first one so the
    closing corner gets a proper joint instead of two caps.
    """
    count = shape.num_curves
    curves = np.zeros(count, STROKE_CURVE_DTYPE)
    starts = first_point + POINTS_PER_CURVE * np.arange(count)
    curves["first_point"] = starts
    joints = [j.value for j in shape.stroke_joints()]
    curves["end_joint"] = joints
    curves["start_cap"] = shape.stroke.cap.value
    curves["end_cap"] = shape.stroke.cap.value
    curves["prev_point"] = NO_NEIGHBOR
    curves["next_point"] = NO_NEIGHBOR

    begin = 0
    for subpath in shape.subpaths():
        end = begin + len(subpath) // POINTS_PER_CURVE
        for curve in range(begin, end):
            if curve > begin:
                curves["prev_point"][curve] = starts[curve - 1]
                curves["start_joint"][curve] = joints[curve - 1]
            if curve < end - 1:
                curves["next_point"][curve] = starts[curve + 1]
        if np.linalg.norm(subpath[0] - subpath[-1]) <= TOLERANCE_FOR_POINT_EQUALITY:
            curves["prev_point"][begin] = starts[end - 1]
            curves["start_joint"][begin] = joints[end - 1]
            curves["next_point"][end - 1] = starts[begin]
        begin = end
    return curves


def _match_joint_styles(points: np.ndarray, curves: np.ndarray, first_point: int) -> None:
    """Give both copies of a shared anchor the same width and color.

    A resampled per-point style list assigns slightly different values to the
    end anchor of one curve and the start anchor of the next (they are
    separate entries in the point array). Averaging them avoids a visible
    step at every joint.
    """
    for curve in curves:
        if curve["next_point"] == NO_NEIGHBOR:
            continue
        end = curve["first_point"] + POINTS_PER_CURVE - 1 - first_point
        start = curve["next_point"] - first_point
        for name in ("width", "color"):
            mean = (points[name][end] + points[name][start]) / 2
            points[name][end] = mean
            points[name][start] = mean


def _fill_vertices(meshes: list[FillMesh]) -> np.ndarray:
    if not meshes:
        return np.zeros(0, FILL_VERTEX_DTYPE)
    vertices = np.zeros(sum(len(m) for m in meshes), FILL_VERTEX_DTYPE)
    vertices["position"] = np.concatenate([m.positions for m in meshes])
    vertices["uv"] = np.concatenate([m.uv for m in meshes])
    vertices["color"] = np.concatenate([m.colors for m in meshes])
    vertices["side"] = np.concatenate([m.side for m in meshes])
    return vertices


def _concat(parts: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype)
    return np.concatenate([np.atleast_1d(p) for p in parts])


def _draw_args(draws: list[_Draw]) -> np.ndarray:
    return np.array([d.args for d in draws], dtype=np.uint32).reshape(-1, 4).view(DRAW_INDIRECT_DTYPE).reshape(-1)

