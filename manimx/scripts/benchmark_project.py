"""Benchmark rendering a manimx project, stage by stage.

    uv run python scripts/benchmark_project.py example-project
    uv run python scripts/benchmark_project.py example-project --save-reference /tmp/ref.npz
    uv run python scripts/benchmark_project.py example-project --check-reference /tmp/ref.npz

Renders the project's timeline in one process with the project's output
settings (and once without motion blur), and prints:

* frames per second and milliseconds per frame,
* the wall time of each renderer stage (measured by wrapping the stage
  functions, no profiler, so Python is not slowed down),
* the GPU time of each render pass (timestamp queries, when supported).

``--save-reference`` / ``--check-reference`` store or compare a set of
small frames across the timeline, so optimizations can be checked to be
bit-identical. Results go to PERFORMANCE.md by hand, one row per step.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import time
from collections import defaultdict

import numpy as np

from manimx.animation import timing
from manimx.drivers.pipeline import FramePipeline
from manimx.project import Project
from manimx.renderer import FrameComposer, WGPURenderer
from manimx.renderer.webgpu import composer as composer_module
from manimx.renderer.webgpu import mesh_batch, renderer, retained

STAGES = [
    (timing.Timeline, "state_at", "state_at (evaluate + snapshot)"),
    (retained.RetainedVectors, "build", "vector batch (retained)"),
    (mesh_batch.MeshBatch, "build", "mesh batch"),
    (renderer, "build_shadow_views", "shadow views"),
    (renderer.WGPURenderer, "render_layers", "render_layers (all)"),
    (composer_module.FrameComposer, "_composite", "composite"),
    (FramePipeline, "_finish_oldest", "wait + readback"),
]


def wrap_stages(spent: dict[str, float]) -> None:
    for owner, name, label in STAGES:
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(*args, _original=original, _label=label, **kwargs):
            start = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                spent[_label] += time.perf_counter() - start

        setattr(owner, name, timed)


def render_all(timeline, settings, times, profile: bool = False) -> tuple[float, WGPURenderer]:
    renderer_ = WGPURenderer(profile=profile)
    pipeline = FramePipeline(FrameComposer(renderer_, settings.effects), settings.config, settings.in_flight)
    for t in times[:5]:  # warm up: shaders, caches, buffers
        pipeline.submit(0, t, timeline.state_at)
    pipeline.flush()
    start = time.perf_counter()
    for index, t in enumerate(times):
        pipeline.submit(index, t, timeline.state_at)
    pipeline.flush()
    return time.perf_counter() - start, renderer_


def reference_frames(project: Project, timeline) -> np.ndarray:
    output = dataclasses.replace(project.output, width=320, height=180)
    settings = project.render_settings(output)
    times = np.linspace(0, timeline.duration, 48)
    renderer_ = WGPURenderer()
    composer = FrameComposer(renderer_, settings.effects)
    return np.stack([composer.compose(timeline.state_at, dataclasses.replace(settings.config, time=float(t))) for t in times])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project")
    parser.add_argument("--save-reference")
    parser.add_argument("--check-reference")
    args = parser.parse_args()

    project = Project.load(args.project)
    started = time.perf_counter()
    timeline = project.build_timeline()
    print(f"prepare: {time.perf_counter() - started:.2f} s")

    if args.save_reference or args.check_reference:
        frames = reference_frames(project, timeline)
        if args.save_reference:
            np.savez_compressed(args.save_reference, frames=frames)
            print(f"saved {len(frames)} reference frames")
        else:
            expected = np.load(args.check_reference)["frames"]
            different = [i for i in range(len(frames)) if not np.array_equal(frames[i], expected[i])]
            print("reference: identical" if not different else f"reference: {len(different)} frames differ: {different}")
            if different:
                print(f"  max difference {np.abs(frames.astype(int) - expected.astype(int)).max()}")

    settings = project.render_settings()
    times = settings.frame_times(timeline.duration)
    sharp = project.render_settings(dataclasses.replace(project.output, motion_blur=1))
    sharp_wall, _ = render_all(timeline, sharp, times)

    spent: dict[str, float] = defaultdict(float)
    wrap_stages(spent)
    wall, _ = render_all(timeline, settings, times)
    count = len(times)
    out = project.output
    print(f"{out.width}x{out.height}, {count} frames, motion blur {out.motion_blur}: {count / wall:6.1f} fps  {wall / count * 1e3:6.2f} ms/frame")
    print(f"{out.width}x{out.height}, {count} frames, no motion blur: {count / sharp_wall:6.1f} fps  {sharp_wall / count * 1e3:6.2f} ms/frame")
    for label, seconds in sorted(spent.items(), key=lambda item: -item[1]):
        print(f"  {label:<32} {seconds / count * 1e3:6.2f} ms/frame  {seconds / wall * 100:5.1f}%")

    _, profiled = render_all(timeline, settings, times[:: max(1, count // 60)], profile=True)
    profiled.profiler.read()
    gpu: dict[str, float] = defaultdict(float)
    sample = times[:: max(1, count // 60)]
    pipeline = FramePipeline(FrameComposer(profiled, settings.effects), settings.config, 1)
    for t in sample:
        pipeline.submit(0, t, timeline.state_at)
        pipeline.flush()
        for name, ms in profiled.profiler.read().items():
            gpu[name] += ms
    if gpu:
        total = sum(gpu.values()) / len(sample)
        passes = ", ".join(f"{name} {ms / len(sample):.2f}" for name, ms in sorted(gpu.items(), key=lambda item: -item[1]))
        print(f"GPU: {total:.2f} ms/frame ({passes})")


if __name__ == "__main__":
    main()
