"""The packing stage: a flat scene in, GPU-ready records out.

The render chain has four stages, each with one input and one output::

    RNode tree --SceneFlattener--> FlatScene
               --FramePacker-----> FramePacket
               --WGPURenderer----> RenderLayers
               --FrameComposer---> frame

:class:`FramePacker` is the third arrow's input side: it turns a
:class:`~..scene_graph.FlatScene` into the numpy records the shaders read
(their dtypes mirror the WGSL structs) and owns every cache that makes
this cheap: the per-object :class:`~..cache.RNodeDataCache` and the
retained vector records. It never touches the GPU, so it can be tested
and measured on its own, and :class:`~.renderer.WGPURenderer` never sees
a render node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...robjects import RShape3D
from ..cache import RNodeDataCache
from ..config import RenderConfig
from ..scene_graph import FlatScene
from .frame import camera_frame
from .mesh_batch import MeshBatch, RetainedMeshes
from .retained import RetainedVectors
from .shadow_views import ShadowLayers, ShadowViews, build_shadow_views
from .vector_batch import VectorBatch

CACHE_MAX_AGE = 1024
"""Packs after which unused per-object cache entries are dropped."""


@dataclass(slots=True, eq=False)
class FramePacket:
    """Everything :class:`~.renderer.WGPURenderer` needs for one render.

    Vector records, mesh vertices and mesh indices come with the ranges
    changed since the previous packet of the same packer (:attr:`source`);
    a renderer that was last fed by another packer uploads them in full.
    """

    config: RenderConfig
    vectors: VectorBatch
    meshes: MeshBatch | None
    """None when the scene has no RShape3D: lights, shadows and the
    deferred passes only affect meshes, so they are skipped."""
    shadows: ShadowViews | None
    """Shadow views of the lights (None without meshes)."""
    frame: np.ndarray
    """The camera's Frame record (FRAME_DTYPE)."""
    scope_parents: np.ndarray
    """Per clip scope, the index of the enclosing one (-1 for none)."""
    mask_count: int
    """Number of clip scopes, each drawing one mask layer."""
    source: object
    """The packer that built this packet (see the class docs)."""


class FramePacker:
    """Turns flat scenes into packets, keeping per-object data across calls."""

    def __init__(self, cache: RNodeDataCache | None = None) -> None:
        self.cache = cache if cache is not None else RNodeDataCache()
        self._vectors = RetainedVectors()
        self._meshes = RetainedMeshes()
        self._shadow_layers = ShadowLayers()

    def pack(self, scene: FlatScene, config: RenderConfig) -> FramePacket:
        if self.cache.next_generation() % CACHE_MAX_AGE == 0:
            self.cache.sweep(CACHE_MAX_AGE)
        vectors = self._vectors.build(scene, self.cache, config.camera.position)
        meshes = shadows = None
        if any(isinstance(item.obj, RShape3D) for item in scene.items):
            if self._meshes.too_much_garbage():
                self._meshes = RetainedMeshes()  # pack everything again
            meshes = MeshBatch.build(scene, self.cache, config, self._meshes)
            # Fills in each light row's shadow view index.
            shadows = build_shadow_views(scene, meshes.light_items, meshes.lights, self.cache, config)
            self._shadow_layers.mark(scene, shadows)
        return FramePacket(
            config=config,
            vectors=vectors,
            meshes=meshes,
            shadows=shadows,
            frame=camera_frame(config, len(meshes.lights) if meshes else 0),
            scope_parents=np.array([scope.parent for scope in scene.clips], dtype=np.int32),
            mask_count=len(scene.clips),
            source=self,
        )
