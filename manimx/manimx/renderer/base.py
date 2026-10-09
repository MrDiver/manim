"""The interface every renderer backend implements.

A backend turns a render-object tree into an image. Everything else
(timelines, animation, compositing, output files) lives outside of it, so a
WebGPU rasterizer, an offline path tracer or a Blender bridge are
interchangeable:

    renderer: Renderer = WGPURenderer()      # or BlenderRenderer(), ...
    image = renderer.render(scene, config)
    renderer.submit(RemoveObjects.subtree(old_group))

Backends may keep per-object data in :attr:`Renderer.cache`; lifecycle
commands reach it through :meth:`Renderer.submit`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..objects import RNode
from .cache import MobjectDataCache
from .commands import CacheCommand
from .config import RenderConfig


class Renderer(ABC):
    def __init__(self) -> None:
        self.cache = MobjectDataCache()

    @abstractmethod
    def render(self, root: RNode, config: RenderConfig) -> np.ndarray:
        """Render ``root`` as seen by ``config.camera``.

        Returns straight-alpha RGBA pixels, shape ``(pixel_height,
        pixel_width, 4)``, dtype ``uint8``, first row at the top.
        """

    def submit(self, command: CacheCommand) -> None:
        """Apply a lifecycle command (e.g. object removal) to the caches.

        Backends with more than one cache override this and forward the
        command to each of them.
        """
        self.cache.execute(command)
