"""The retained vector scene: GPU records kept per object, patched per frame.

:meth:`VectorBatch.build` packs every shape's records into fresh arrays
for every render, and the vector pass uploads all of them, although from
one render to the next usually only a few transforms change. This module
keeps the packed records instead, in persistent arrays (*arenas*) that
mirror the GPU buffers, with one *slot* per shape (its uid, and whether it
is drawn as a mask). Each render diffs the flat scene against the slots,
using the render contract's identity and version tokens:

* unchanged shape, unchanged placement: nothing is written;
* new world matrix, layer, clip or alpha: the shape's 96-byte object
  entry;
* new fill colors or stroke style (version changed): its color or point
  records, in place;
* new points: its records are written to a new range (the old one
  becomes garbage), and records that point at them (curves, instances)
  are rewritten.

Only the written ranges are uploaded (:attr:`VectorBatch.dirty`). The
draw lists are rebuilt every frame from the slots; they are small, and
transparent draws must be re-sorted by view depth anyway. Shapes
that leave the scene free their slots; when garbage outweighs live
records, everything is repacked from scratch.

The records are the same as :meth:`VectorBatch.build` produces, only at
other positions in the buffers, so the rendered pixels are identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...robjects import RShape2D
from ...robjects.rbezier_path import RBezierPath
from ..cache import RNodeDataCache
from ..scene_graph import NO_CLIP, DrawItem, FlatScene
from ..tessellation import tessellate_fill
from .arena import Arena
from .vector_batch import (
    FILL_VERTEX_DTYPE,
    FLAG_CAMERA_FACING,
    FLAG_FRONT_ONLY,
    INSTANCE_DTYPE,
    KIND_FILL,
    KIND_STROKE,
    NO_NEIGHBOR,
    NO_VIEW_ENTRY,
    OBJECT_DTYPE,
    STROKE_CURVE_DTYPE,
    STROKE_POINT_DTYPE,
    STROKE_VERTEX_COUNT,
    CameraViews,
    VectorBatch,
    _Draw,
    _draw_args,
    _fill_vertices,
    _StrokeTopology,
    view_depth,
)

ARENAS = {
    "objects": OBJECT_DTYPE,
    "instances": INSTANCE_DTYPE,
    "fill_vertices": FILL_VERTEX_DTYPE,
    "fill_colors": np.dtype((np.float32, 4)),
    "stroke_points": STROKE_POINT_DTYPE,
    "stroke_curves": STROKE_CURVE_DTYPE,
}


@dataclass(eq=False)
class _Slot:
    """Where one shape's records live, and what they were built from."""

    object: int
    world: np.ndarray | None = None
    layer: int = -1
    clip: int = NO_CLIP - 1
    flags: int = -1
    alpha: float = -1.0
    view: object = None
    points_key: object = None
    """Points version the geometry ranges below were built from."""
    fill_mesh: object = None
    fill_first: int = 0
    fill_count: int = 0
    fill_instance: int = -1
    colors_key: object = None
    colors_first: int = 0
    colors_count: int = 0
    stroke_key: object = None
    points_first: int = 0
    points_count: int = 0
    curves_first: int = 0
    curves_count: int = 0
    stroke_instances: int = -1


@dataclass(eq=False)
class RetainedVectors:
    """Persistent vector records of one renderer, see the module docs."""

    arenas: dict[str, Arena] = field(default_factory=lambda: {name: Arena(dtype) for name, dtype in ARENAS.items()})
    slots: dict[tuple[int, int], _Slot] = field(default_factory=dict)
    free_objects: list[int] = field(default_factory=list)
    _views: CameraViews = field(default_factory=dict)
    """The camera views of the current build (see ``camera_view_entries``)."""

    def build(self, scene: FlatScene, cache: RNodeDataCache, world_to_view: np.ndarray, views: CameraViews | None = None) -> VectorBatch:
        if self._too_much_garbage():
            self.__init__()  # repack everything from scratch
        self._views = views or {}
        draws: list[_Draw] = []
        seen: set[tuple[int, int]] = set()
        for item in scene.items:
            self._add(item, cache, world_to_view, draws, seen)
        for index, scope in enumerate(scene.clips):
            for item in scope.mask:
                self._add(item, cache, world_to_view, draws, seen, mask_of=index, front_only=scope.group.one_sided)
        for key in [key for key in self.slots if key not in seen]:
            self._free(self.slots.pop(key))
        return self._batch(scene, draws)

    # -- per shape ----------------------------------------------------------

    def _add(
        self,
        item: DrawItem,
        cache: RNodeDataCache,
        world_to_view: np.ndarray,
        draws: list[_Draw],
        seen: set[tuple[int, int]],
        mask_of: int = NO_CLIP,
        front_only: bool = False,
    ) -> None:
        shape = item.obj
        if not isinstance(shape, RBezierPath) or shape.num_curves == 0:
            return
        key = (shape.uid, mask_of)
        seen.add(key)
        slot = self.slots.get(key)
        if slot is None:
            slot = self.slots[key] = _Slot(self._object_slot())
        points_key = shape.version("points")
        if slot.points_key != points_key:
            # New geometry: every range built from the points moves.
            self._release_geometry(slot)
            slot.points_key = points_key

        layer = 2 * item.order
        flags = (0 if isinstance(shape, RShape2D) else FLAG_CAMERA_FACING) | (FLAG_FRONT_ONLY if front_only else 0)
        entry = self.arenas["objects"].data[slot.object : slot.object + 1]
        view = self._views.get(shape.uid, NO_VIEW_ENTRY)
        placement = (layer, item.clip, flags, item.alpha, view)
        if slot.world is None or not np.array_equal(slot.world, item.world) or (slot.layer, slot.clip, slot.flags, slot.alpha, slot.view) != placement:
            entry["model"] = item.world.T  # WGSL matrices are column-major
            entry["layer"], entry["clip"], entry["flags"], entry["alpha"] = placement[:4]
            entry["view"], entry["view_scale"] = view
            self.arenas["objects"].touch(slot.object, slot.object + 1)
            slot.world = item.world
            slot.layer, slot.clip, slot.flags, slot.alpha, slot.view = placement
        if item.alpha <= 0.0:
            return  # invisible: the slot stays, nothing is drawn

        local_center = cache.get_or_create(shape.uid, "center", points_key, lambda: np.append(shape.points_3d().mean(axis=0), 1.0))
        distance = view_depth(world_to_view, item.world @ local_center)
        clipped = item.clip != NO_CLIP
        for draw in (self._fill(shape, slot, cache, distance, layer), self._stroke(shape, slot, cache, distance, layer + 1)):
            if draw is None:
                continue
            draw.opaque &= not clipped and item.alpha >= 1.0
            draw.mask_of = mask_of
            draw.casts_shadow = shape.cast_shadows and mask_of == NO_CLIP
            draw.clip = item.clip
            draws.append(draw)

    def _fill(self, shape: RBezierPath, slot: _Slot, cache: RNodeDataCache, distance: float, layer: int) -> _Draw | None:
        if not isinstance(shape, RShape2D):
            return None
        opacity = shape.fill.opacity
        if not np.any(opacity > 0.0):
            return None
        mesh = cache.get_or_create(shape.uid, "fill-mesh", shape.version("points"), lambda: tessellate_fill(shape.points))
        if len(mesh) == 0:
            return None
        if slot.fill_mesh is not mesh:
            vertices = cache.get_or_create(shape.uid, "fill-vertices", shape.version("points"), lambda: _fill_vertices([mesh]))
            slot.fill_first = self._place("fill_vertices", vertices, slot.fill_first, slot.fill_count)
            slot.fill_count = len(vertices)
            slot.fill_mesh = mesh
        colors_key = (shape.version("points"), shape.version("fill"))
        if slot.colors_key != colors_key:
            colors = shape.fill_rgba().astype(np.float32)
            first = self._place("fill_colors", colors, slot.colors_first, slot.colors_count)
            if first != slot.colors_first or slot.colors_count == 0:
                self.arenas["objects"].data[slot.object]["fill_colors"] = first
                self.arenas["objects"].touch(slot.object, slot.object + 1)
            slot.colors_first, slot.colors_count = first, len(colors)
            slot.colors_key = colors_key
        if slot.fill_instance < 0:
            slot.fill_instance = self._allocate("instances", 1)
            self.arenas["instances"].write(slot.fill_instance, _instances(KIND_FILL, slot.object, np.zeros(1, np.uint32)))
        return _Draw(
            args=(slot.fill_count, 1, slot.fill_first, slot.fill_instance),
            # A picture may be see-through anywhere (a transparent background).
            opaque=bool(np.all(opacity >= 1.0)) and shape.uid not in self._views,
            camera_distance=distance,
            layer=layer,
        )

    def _stroke(self, shape: RBezierPath, slot: _Slot, cache: RNodeDataCache, distance: float, layer: int) -> _Draw | None:
        stroke = shape.stroke
        if not (np.any(stroke.width > 0.0) and np.any(stroke.opacity > 0.0)):
            return None
        points_version = shape.version("points")
        topology = cache.get_or_create(
            shape.uid, "stroke-topology", (points_version, stroke.joint, stroke.cap), lambda: _StrokeTopology.build(shape)
        )
        stroke_key = (points_version, shape.version("stroke"), id(topology))
        if slot.stroke_key != stroke_key:
            points = cache.get_or_create(
                shape.uid, "stroke-points", (points_version, shape.version("stroke")), lambda: topology.points(shape)
            )
            slot.points_first = self._place("stroke_points", points, slot.points_first, slot.points_count)
            slot.points_count = len(points)
            curves = topology.curves.copy()
            curves["first_point"] += slot.points_first
            for link in ("prev_point", "next_point"):
                linked = curves[link] != NO_NEIGHBOR
                curves[link][linked] += slot.points_first
            first = self._place("stroke_curves", curves, slot.curves_first, slot.curves_count)
            if first != slot.curves_first or len(curves) != slot.curves_count or slot.stroke_instances < 0:
                rows = _instances(KIND_STROKE, slot.object, first + np.arange(len(curves), dtype=np.uint32))
                old = slot.curves_count if slot.stroke_instances >= 0 else 0
                slot.stroke_instances = self._place("instances", rows, slot.stroke_instances, old)
            slot.curves_first, slot.curves_count = first, len(curves)
            slot.stroke_key = stroke_key
        return _Draw(
            args=(STROKE_VERTEX_COUNT, shape.num_curves, 0, slot.stroke_instances),
            opaque=bool(np.all(stroke.opacity >= 1.0)),
            camera_distance=distance,
            layer=layer,
        )

    # -- slots and ranges ---------------------------------------------------

    def _object_slot(self) -> int:
        if self.free_objects:
            return self.free_objects.pop()
        return self._allocate("objects", 1)

    def _allocate(self, arena: str, count: int) -> int:
        return self.arenas[arena].allocate(count)

    def _place(self, arena: str, records: np.ndarray, old_first: int, old_count: int) -> int:
        return self.arenas[arena].place(records, old_first, old_count)

    def _release_geometry(self, slot: _Slot) -> None:
        """New points: everything built from them is rebuilt (in place when
        the size still fits, see _place)."""
        slot.fill_mesh = None
        slot.stroke_key = None
        slot.colors_key = None

    def _free(self, slot: _Slot) -> None:
        self.free_objects.append(slot.object)
        arenas = self.arenas
        arenas["fill_vertices"].garbage += slot.fill_count
        arenas["stroke_points"].garbage += slot.points_count
        arenas["stroke_curves"].garbage += slot.curves_count
        arenas["instances"].garbage += slot.curves_count + (slot.fill_instance >= 0)
        arenas["fill_colors"].garbage += slot.colors_count

    def _too_much_garbage(self) -> bool:
        return any(arena.garbage > max(4096, arena.used // 2) for arena in self.arenas.values())

    # -- the frame's batch ------------------------------------------------------

    def _batch(self, scene: FlatScene, draws: list[_Draw]) -> VectorBatch:
        scope_count = len(scene.clips)
        frame = [d for d in draws if d.mask_of == NO_CLIP]
        opaque = [d for d in frame if d.opaque]
        transparent = sorted((d for d in frame if not d.opaque), key=lambda d: (-round(d.camera_distance, 6), d.layer))
        masks = [_draw_args(sorted((d for d in draws if d.mask_of == scope), key=lambda d: d.layer)) for scope in range(scope_count)]
        arenas = self.arenas
        return VectorBatch(
            objects=arenas["objects"].view(),
            instances=arenas["instances"].view(),
            fill_vertices=arenas["fill_vertices"].view(),
            fill_colors=arenas["fill_colors"].view(),
            stroke_points=arenas["stroke_points"].view(),
            stroke_curves=arenas["stroke_curves"].view(),
            opaque_draws=_draw_args(opaque),
            transparent_draws=_draw_args(transparent),
            mask_draws=masks,
            shadow_draws={
                scope: _draw_args([d for d in frame if d.casts_shadow and scene.in_scope(d.clip, scope)])
                for scope in (NO_CLIP, *range(scope_count))
            },
            dirty={name: arena.ranges() for name, arena in arenas.items()},
        )


def _instances(kind: int, object_index: int, curves: np.ndarray) -> np.ndarray:
    rows = np.zeros((len(curves), 4), np.uint32)
    rows[:, 0] = kind
    rows[:, 1] = object_index
    rows[:, 2] = curves
    return rows.view(INSTANCE_DTYPE).reshape(-1)
