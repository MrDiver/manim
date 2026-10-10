"""The render service: the render chain of one process, kept alive.

    service = RenderService.shared()
    image = service.render(tree, RenderConfig())   # (H, W, 4) uint8

The chain (see packer.py) runs a tree through its stages::

    RNode tree --SceneFlattener--> FlatScene
               --FramePacker-----> FramePacket
               --WGPURenderer----> RenderLayers
               --FrameComposer---> frame

Each stage is its own object with one input and one output, so any of
them can be used alone (a :class:`~.renderer.WGPURenderer` renders
packets from any source). The service owns one of each, plus a composer
per effect chain, and keeps them across calls: GPU resources, per-object
data and retained records survive, which keeps a long-lived web UI cheap.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import wgpu

from ...robjects.planned_frame import PlannedFrame
from ...robjects import RNode
from ..base import Renderer
from ..config import Frame, RenderConfig
from ..effects import Effect
from ..scene_graph import SceneFlattener
from .composer import FrameComposer
from .core_packer import CorePacker
from .gpu import GPUContext
from .packer import FramePacker
from .renderer import WGPURenderer
from .targets import RenderLayers

_shared: RenderService | None = None
_shared_lock = threading.Lock()


class RenderService(Renderer):
    """One render chain, meant to live as long as the process.

    A process should render through one instance, :meth:`shared`, instead
    of creating services per call. Rendering is not thread-safe; one
    thread renders at a time.
    """

    @classmethod
    def shared(cls) -> RenderService:
        """The process's render service, created on first use."""
        global _shared
        with _shared_lock:
            if _shared is None:
                _shared = cls()
            return _shared

    def __init__(self, context: GPUContext | None = None, profile: bool = False) -> None:
        super().__init__()
        self.flattener = SceneFlattener()
        self.packer = FramePacker(self.cache)
        self.core_packer = CorePacker()
        """Packs PlannedFrames (frames of the core, see CORE.md)."""
        self.renderer = WGPURenderer(context, profile)
        self._composers: dict[tuple[Effect, ...], FrameComposer] = {}

    def render(self, frame: Frame, config: RenderConfig) -> np.ndarray:
        """Render ``frame`` (a tree, or views from ``frame_at``) to
        ``(H, W, 4)`` uint8 RGBA with default compositing.
        """
        return self.composer().compose(frame, config)

    def composer(self, effects: Sequence[Effect] = ()) -> FrameComposer:
        """The composer for an effect chain, kept like everything else (its
        pipelines and textures survive across calls).
        """
        key = tuple(effects)
        composer = self._composers.get(key)
        if composer is None:
            composer = self._composers[key] = FrameComposer(self, key)
        return composer

    def render_layers(self, root: RNode | PlannedFrame, config: RenderConfig, encoder: wgpu.GPUCommandEncoder | None = None) -> RenderLayers:
        """Run ``root`` through flattener, packer and renderer; a
        PlannedFrame is flattened and packed by the core instead."""
        if isinstance(root, PlannedFrame):
            return self.renderer.render_layers(self.core_packer.pack(root, config), encoder)
        return self.renderer.render_layers(self.packer.pack(self.flattener.flatten(root), config), encoder)
