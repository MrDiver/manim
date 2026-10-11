"""Single frames, rendered on demand: for the preview, scripts and tests.

    image = render_image(Square(2.0, fill_opacity=1.0), config=RenderConfig(pixel_width=640, pixel_height=360))
    image = render_image(MyScene, time=2.5)

:class:`StillRenderer` keeps one :class:`~.core_pipeline.CoreFramePipeline`
per configuration (size, effects, lighting; not the time), so asking for
frames of the same size again reuses its GPU resources.
:func:`render_image` renders through one shared StillRenderer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..animation.nodes.node import Node
from ..animation.plan import planned_frames
from ..animation.scene import PreparedScene, Scene
from ..renderer.config import Frame, RenderConfig
from ..renderer.effects import Effect
from .core_pipeline import CoreFramePipeline


class StillRenderer:
    """Single frames at any configuration: one :class:`CoreFramePipeline`
    per configuration, the last ``keep`` of them kept, each with one frame
    in flight. Not thread-safe: render from one thread at a time."""

    def __init__(self, keep: int = 4) -> None:
        self.keep = keep
        self._pipelines: dict[tuple, CoreFramePipeline] = {}

    def render(self, scene: Frame | Callable[[float], Frame], config: RenderConfig, effects: tuple[Effect, ...] = ()) -> np.ndarray:
        """``scene`` (views, or a function of time returning them) at
        ``config.time``: ``(H, W, 4)`` uint8 RGBA."""
        key = (tuple(effects), repr(dataclasses.replace(config, time=0.0)))
        pipeline = self._pipelines.pop(key, None)
        if pipeline is None:
            pipeline = CoreFramePipeline(tuple(effects), config, in_flight=1)
        self._pipelines[key] = pipeline  # the most recent last
        while len(self._pipelines) > self.keep:
            del self._pipelines[next(iter(self._pipelines))]
        pipeline.config = config
        return pipeline.render(scene, config.time)


_shared: StillRenderer | None = None


def render_image(
    source: Any,
    time: float = 0.0,
    config: RenderConfig | None = None,
    effects: Iterable[Effect] = (),
) -> np.ndarray:
    """``source`` at ``time`` as ``(H, W, 4)`` uint8 RGBA pixels.

    ``source`` is a Scene class or instance, a prepared scene, a timeline,
    or one node or a list of them. Scenes look through their own camera;
    nodes through ``config.camera``.
    """
    global _shared
    config = dataclasses.replace(config or RenderConfig(), time=float(time))
    frame_at = _frames(source, config)
    if _shared is None:
        _shared = StillRenderer()
    return _shared.render(frame_at, config, tuple(effects))


def _frames(source: Any, config: RenderConfig) -> Callable[[float], Frame]:
    if isinstance(source, Node) or (isinstance(source, (list, tuple)) and all(isinstance(node, Node) for node in source)):
        nodes = [source] if isinstance(source, Node) else list(source)
        planned = planned_frames(_still(nodes))
        # Nodes have no scene to bring a camera: use the config's.
        return lambda t: tuple(dataclasses.replace(view, camera=config.camera) for view in planned.frame_at(t))
    if isinstance(source, type) and issubclass(source, Scene):
        source = source()
    if isinstance(source, Scene):
        source = source.prepare()
    planned = planned_frames(source)
    if planned is None:
        raise TypeError(f"cannot render {source!r}: give nodes, a Scene, a prepared scene or a timeline")
    return planned.frame_at


def _still(nodes: list[Node]) -> PreparedScene:
    """A scene that shows ``nodes`` and nothing happens in."""

    class Still(Scene):
        def construct(self) -> None:
            self.add(*nodes)

    return Still().prepare()


__all__ = ["StillRenderer", "render_image"]
