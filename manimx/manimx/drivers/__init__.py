"""Drivers: everything that turns a clock into ``state_at(t)`` queries and
the results into output (section 10 of docs/animation_engine_design.md).

For now the offline driver: render every frame of a scene, timeline or
view into a video file or an image sequence, pipelined on the GPU and
optionally across worker processes.
"""

from .core_pipeline import CoreFramePipeline
from .offline import FrameSource, RenderSettings, RenderStats, load_source, render
from .sinks import (
    FrameFormat,
    FrameList,
    FrameSink,
    ImageSequence,
    Tee,
    VideoFile,
    Window,
)
from .stills import StillRenderer, render_image

__all__ = [
    "FrameFormat",
    "FrameList",
    "CoreFramePipeline",
    "FrameSink",
    "FrameSource",
    "ImageSequence",
    "RenderSettings",
    "RenderStats",
    "Tee",
    "VideoFile",
    "StillRenderer",
    "Window",
    "load_source",
    "render",
    "render_image",
]
