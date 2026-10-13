"""Benchmark rendering a manimx project.

    uv run python scripts/benchmark_project.py tests/projects/benchmark
    uv run python scripts/benchmark_project.py tests/projects/benchmark --save-reference /tmp/ref.npz
    uv run python scripts/benchmark_project.py tests/projects/benchmark --check-reference /tmp/ref.npz

Renders the project's timeline in one process with the project's output
settings (and once without motion blur), and prints:

* frames per second and milliseconds per frame,
* the wall time of evaluating frames and of handing them to the core
  (measured by wrapping those functions, no profiler),
* the core's CPU time per stage (pack, draw, submit, read-back),
* the GPU time of each render pass (timestamp queries, when supported).

``--save-reference`` / ``--check-reference`` store or compare a set of
small frames across the timeline, so optimizations can be checked to be
bit-identical. Results go to PERFORMANCE.md by hand, one row per step.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import time
from collections import defaultdict

import numpy as np
from manimx.animation.timing import Timeline
from manimx.drivers.core_pipeline import CoreFramePipeline
from manimx.drivers.stills import StillRenderer
from manimx.project import Project

# Stages that are timed.
STAGES = [
    (Timeline, "frame_at", "frame_at (evaluate, snapshot)"),
    (CoreFramePipeline, "submit", "submit (pack, render, composite)"),
    (CoreFramePipeline, "flush", "flush"),
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


def render_all(frame_at, settings, times) -> tuple[float, dict[str, float]]:
    """Seconds for every frame, and the core's CPU milliseconds per stage."""
    pipeline = CoreFramePipeline(settings.effects, settings.config, settings.in_flight)
    for t in times[:5]:  # warm up: shaders, caches, buffers
        pipeline.submit(0, t, frame_at)
    pipeline.flush()
    pipeline.core.stage_timings()
    start = time.perf_counter()
    for index, t in enumerate(times):
        pipeline.submit(index, t, frame_at)
    pipeline.flush()
    return time.perf_counter() - start, pipeline.core.stage_timings()


def reference_frames(project: Project, frame_at, duration: float) -> np.ndarray:
    small = copy.deepcopy(project.settings)
    small.output.width, small.output.height = 320, 180
    settings = project.render_settings(small)
    renderer = StillRenderer()
    configs = (dataclasses.replace(settings.config, time=float(t)) for t in np.linspace(0, duration, 48))
    return np.stack([renderer.render(frame_at, config, settings.effects) for config in configs])


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
        frames = reference_frames(project, timeline.frame_at, timeline.duration)
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
    sharp_settings = copy.deepcopy(project.settings)
    sharp_settings.output.samples = 1
    sharp_wall, _ = render_all(timeline.frame_at, project.render_settings(sharp_settings), times)

    spent: dict[str, float] = defaultdict(float)
    wrap_stages(spent)
    wall, stages = render_all(timeline.frame_at, settings, times)
    count = len(times)
    out = project.output
    print(f"{out.width}x{out.height}, {count} frames, {project.settings.output.samples} samples: {count / wall:6.1f} fps  {wall / count * 1e3:6.2f} ms/frame")
    print(f"{out.width}x{out.height}, {count} frames, 1 sample: {count / sharp_wall:6.1f} fps  {sharp_wall / count * 1e3:6.2f} ms/frame")
    for label, seconds in ((label, spent[label]) for _, _, label in STAGES if label in spent):
        print(f"  {label:<36} {seconds / count * 1e3:6.2f} ms/frame  {seconds / wall * 100:5.1f}%")
    print("core CPU: " + ", ".join(f"{name} {ms / count:.2f}" for name, ms in stages.items()) + " ms/frame")

    gpu: dict[str, float] = defaultdict(float)
    sample = times[:: max(1, count // 60)]
    pipeline = CoreFramePipeline(settings.effects, settings.config, 1, profile=True)
    for t in sample:
        pipeline.submit(0, t, timeline.frame_at)
        pipeline.flush()
        for name, ms in pipeline.gpu_timings().items():
            gpu[name] += ms
    if gpu:
        total = sum(gpu.values()) / len(sample)
        passes = ", ".join(f"{name} {ms / len(sample):.2f}" for name, ms in sorted(gpu.items(), key=lambda item: -item[1]))
        print(f"GPU: {total:.2f} ms/frame ({passes})")


if __name__ == "__main__":
    main()
