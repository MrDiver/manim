"""The packing stage on the core: a planned frame in, a packet out.

:class:`CorePacker` is the counterpart of :class:`~.packer.FramePacker`
for :class:`~manimx.animation.plan.PlannedFrame` views: the core
(``manimx._core.Packer``, see CORE.md) flattens the scene graph with the
frame's values and packs the records in Rust, and this module wraps them
in the same :class:`~.packer.FramePacket`, so :class:`~.renderer.WGPURenderer`
renders it unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ... import _core
from ..config import RenderConfig
from .frame import FRAME_DTYPE, camera_frame
from .mesh_batch import (
    DRAW_INDEXED_DTYPE,
    LIGHT_DTYPE,
    MATERIAL_DTYPE,
    MESH_OBJECT_DTYPE,
    MESH_VERTEX_DTYPE,
    DrawGroup,
    MeshBatch,
)
from .packer import FramePacket
from .shadow_views import SHADOW_VIEW_DTYPE, ShadowViews
from .vector_batch import (
    DRAW_INDIRECT_DTYPE,
    FILL_VERTEX_DTYPE,
    INSTANCE_DTYPE,
    OBJECT_DTYPE,
    STROKE_CURVE_DTYPE,
    STROKE_POINT_DTYPE,
    VectorBatch,
)

_VECTOR_DTYPES = {
    "objects": OBJECT_DTYPE,
    "instances": INSTANCE_DTYPE,
    "fill_vertices": FILL_VERTEX_DTYPE,
    "fill_colors": np.dtype(np.float32),  # (K, 4) RGBA, reshaped below
    "stroke_points": STROKE_POINT_DTYPE,
    "stroke_curves": STROKE_CURVE_DTYPE,
}


class CorePacker:
    """Packs planned frames on the core, keeping its records across calls."""

    def __init__(self) -> None:
        self._packer = _core.Packer()

    def pack(self, frame: Any, config: RenderConfig) -> FramePacket:
        """The packet for ``frame`` (a PlannedFrame) seen with ``config``."""
        packed = self._packer.pack(frame.graph, frame.state, config_values(config))
        vectors = packed["vectors"]
        meshes = packed["meshes"]
        shadows = packed["shadows"]
        scope_parents = np.array(packed["scope_parents"], dtype=np.int32)
        return FramePacket(
            config=config,
            vectors=VectorBatch(
                **{name: vectors[name].view(dtype) for name, dtype in _VECTOR_DTYPES.items() if name != "fill_colors"},
                fill_colors=vectors["fill_colors"].view(np.float32).reshape(-1, 4),
                opaque_draws=_records(vectors["opaque_draws"], DRAW_INDIRECT_DTYPE),
                transparent_draws=_records(vectors["transparent_draws"], DRAW_INDIRECT_DTYPE),
                mask_draws=[_records(draws, DRAW_INDIRECT_DTYPE) for draws in vectors["mask_draws"]],
                shadow_draws={scope: _records(draws, DRAW_INDIRECT_DTYPE) for scope, draws in vectors["shadow_draws"]},
                dirty=vectors["dirty"],
            ),
            meshes=None if meshes is None else _mesh_batch(meshes),
            shadows=None if shadows is None else _shadow_views(shadows),
            frame=camera_frame(config, packed["light_count"]),
            scope_parents=scope_parents,
            mask_count=len(scope_parents),
            source=self,
        )


def config_values(config: RenderConfig) -> tuple:
    """What the core's packer reads from ``config`` (see parse_settings in
    core/src/packer.rs).
    """
    camera = config.camera
    return (
        config.time,
        config.layer_depth_step,
        config.exposure,
        tuple(config.ambient_linear),
        config.shadow_softness,
        config.shadow_samples,
        config.shadow_resolution,
        config.shadows,
        config.default_lights,
        camera.transform.matrix().reshape(-1).tolist(),
        [(view.uid, 1.0 / view.width, 1.0 / view.height) for view in config.camera_views],
        config.hide_camera_views,
        camera.focus_distance,
        camera.aperture,
        config.shadow_filter,
    )


def _records(data: bytes, dtype: np.dtype) -> np.ndarray:
    return np.frombuffer(data, dtype=dtype)


def _mesh_batch(meshes: dict) -> MeshBatch:
    return MeshBatch(
        objects=_records(meshes["objects"], MESH_OBJECT_DTYPE),
        vertices=meshes["vertices"].view(MESH_VERTEX_DTYPE),
        indices=meshes["indices"].view(np.uint32),
        materials=_records(meshes["materials"], MATERIAL_DTYPE),
        lights=np.frombuffer(meshes["lights"], LIGHT_DTYPE).copy(),
        light_items=[],
        groups=[
            DrawGroup(textures, surface_shader, list(map(tuple, _records(draws, DRAW_INDEXED_DTYPE).tolist())))
            for textures, surface_shader, draws in meshes["groups"]
        ],
        shadow_draws={scope: _records(draws, DRAW_INDEXED_DTYPE) for scope, draws in meshes["shadow_draws"]},
        dirty=meshes["dirty"],
    )


def _shadow_views(shadows: dict) -> ShadowViews:
    return ShadowViews(
        frames=_records(shadows["frames"], FRAME_DTYPE),
        views=_records(shadows["views"], SHADOW_VIEW_DTYPE),
        scopes=list(shadows["scopes"]),
        changed=list(shadows["changed"]),
    )
