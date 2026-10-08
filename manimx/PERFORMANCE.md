# Rendering performance

Measurements of the renderer and animation engine, one row per optimization
step, so every change is justified by numbers and regressions show up.

## How we measure

```sh
uv run python scripts/benchmark_project.py example-project --check-reference /tmp/ref.npz
```

- **Workload:** the example project (Intro: the Manim logo, Bounce: five
  simulated balls, Outro: a template animation): 1114 frames, 1920×1080, 60 fps,
  motion blur with 8 sub-frames, all rendered in one process with 3 frames in
  flight. No encoding: frames go nowhere, because H.264 encoding runs in
  parallel in ffmpeg and was measured to add nothing.
- **Stage times** come from wrapping the stage functions with timers (no
  profiler, which roughly doubles Python's cost). Stages nest:
  `render_layers` contains the vector batch, the mesh batch and the shadow
  views.
- **GPU times** come from timestamp queries on a sample of frames, rendered one
  at a time.
- **Correctness:** `--check-reference` renders 48 frames across the timeline
  and compares them with frames saved before the first optimization. Every step
  below is bit-identical.
- **Machine:** Linux, Vulkan; numbers are comparable within this table only.
  Run-to-run noise is about ±0.5 fps.

## Results

| Step | fps (blur 8) | ms/frame | fps (no blur) | state_at | vector batch | render_layers | GPU ms/frame |
|---|---|---|---|---|---|---|---|
| 0. Baseline | 74.5 | 13.43 | 407.6 | 2.12 | 3.91 | 9.24 | 3.68 |
| A. Cached stroke records | 87.5 | 11.43 | 452.2 | 2.09 | 1.97 | 7.29 | 4.06* |

\* GPU time is sampled on 60 frames with a full sync per frame and varies by
about ±10% between runs. Step A does not change any GPU work.

## Baseline analysis

What a frame cost before any optimization (13.4 ms):

- **CPU-bound:** the GPU needs 3.7 ms per frame, so it idled about 75% of
  the time.
- **Motion blur multiplies everything:** without it a frame takes 2.5 ms.
  Each of the 8 sub-frames evaluates the scene and runs the renderer's whole
  CPU side again.
- **Within a sub-frame, from the largest cost down:**
  - **Vector batch (29%):** every shape's stroke records (curve links,
    joints, per-point width and color) were rebuilt in Python loops for
    every sub-frame, even for shapes that only move.
  - **WebGPU call overhead (about 28%):** views, render pass descriptors
    and structs were recreated for every sub-frame, about 330k binding
    calls for 557 frames.
  - **`state_at` (16%):** evaluating signals and building render objects,
    including re-parsing colors that are already floats.
  - **3D stages with no 3D content (about 12% CPU, 1.9 ms GPU):** mesh
    batch, default lights, shadow views, and the geometry and lighting
    passes ran although the project has no meshes.
- **Everything else is small:** preparing all scenes takes 0.6 s once,
  mixing the audio 5 ms, and importing manimx 0.34 s.

## Steps

### A. Cached stroke records

Stroke records depend on a shape's points and stroke style, never on its
transform, but were rebuilt for every sub-frame. Now:

- **Topology is cached** per shape, keyed by points version, joints and cap:
  curve records linked inside each subpath, and the pairs of anchors shared
  by consecutive curves.
- **Per-point records are cached** (position, width, color with shared anchors
  matched), keyed by points and stroke version. A fade, which changes the
  stroke's opacity, rebuilds only these, vectorized. The old per-curve loop
  that averaged shared anchors is a single numpy operation, and gives the same
  float32 results, because every anchor is in at most one pair.
- **Each shape's local center** (for sorting transparent draws) is cached by
  points version.
- **Curve instances** are added as one array per shape instead of a Python
  loop per curve.

Moving shapes now only offset the cached records by their position in the
frame's buffers.
