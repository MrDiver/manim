"""Drivers: everything that turns a clock into ``state_at(t)`` queries and
the results into output (section 10 of docs/animation_engine_design.md).

For now the offline driver: render every frame of a scene, timeline or
view into a video file or an image sequence, pipelined on the GPU and
optionally across worker processes.
"""

from .offline import FrameSource, RenderSettings, RenderStats, load_source, render
from .pipeline import FramePipeline
from .sinks import FrameFormat, FrameList, FrameSink, ImageSequence, Tee, VideoFile, Window

__all__ = [
    "FrameFormat",
    "FrameList",
    "FramePipeline",
    "FrameSink",
    "FrameSource",
    "ImageSequence",
    "RenderSettings",
    "RenderStats",
    "Tee",
    "VideoFile",
    "Window",
    "load_source",
    "render",
]
