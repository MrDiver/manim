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
| B. No 3D stages without meshes | 119.6 | 8.36 | 556.3 | 2.02 | 1.88 | 4.32 | 1.72 |
| C. Renderer as a service | 134.9 | 7.41 | 603.9 | 1.99 | 1.87 | 3.51 | 1.75 |

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

### B. No 3D stages without meshes

Lights, shadows and the deferred passes (geometry, lighting) only affect
meshes (Shape3D). The example project has none, yet every sub-frame built
a mesh batch with the default key light, built shadow views for 2D shapes
that cast shadows onto nothing, and ran a full-screen lighting pass over
an empty G-buffer: 1.6 ms of CPU and 1.9 ms of GPU per frame.

Now a frame without meshes skips all of that. Two things still have to be
right:

- **The G-buffer's depth must be empty.** Vector shapes are depth-tested
  against the meshes' depth, so stale depth from an earlier frame with
  meshes would hide them.
- **The lit layer must be empty**, because the composite reads it.

The renderer therefore clears both once, with load-op clears and no draws,
and remembers it for the current targets. Following mesh-free frames skip
those passes completely; a frame with meshes, or new targets (a size
change), resets the state.
`test_frames_without_meshes_after_meshes_start_from_empty_3d_layers`
renders flat, then flat with a sphere, then flat again, and checks that the
last frame equals a fresh renderer's. Breaking the clear makes it fail.

Result: +37% (87.5 → 119.6 fps). GPU time drops from about 3.7 to 1.7 ms per
frame, and `render_layers` from 7.3 to 4.3 ms.

### C. The renderer as a service

The renderer is meant to be a long-lived service: built once per process,
keeping everything it creates across all calls. Within one render call it
already lived for all frames, but every call (each driver `render()`,
`manimx still`, each worker process) created its own renderer with its own
GPU device. Even within a call, these were recreated for every sub-frame:

- **render target views and attachments** for the lit, vector and mask
  scratch targets and the vector depth;
- **in the composer:** a bind group and texture views for every fullscreen
  pass (composite, finalize, bloom, shader effects);
- **command encoders and submissions:** two per sub-frame.

Now:

- **`WGPURenderer.shared()`** is the process's renderer service: one device,
  created on first use. Drivers render through it, so the CLI, tests and
  later the web UI keep shaders, pipelines, buffers, targets and per-object
  data across calls. Rendering stays single-threaded per renderer.
- **`renderer.composer(effects)`** keeps one composer per effect chain, with
  its pipelines and texture pool.
- **`ResourceCache`**, owned by the renderer, keeps views per texture (weakly,
  so they go with their texture) and bind groups keyed by their layout and
  the identity of everything they bind (LRU-bounded, 512 entries).
  Fullscreen pipelines keep their bind group layout, because each
  `get_bind_group_layout()` call returns a new object, which would defeat
  the cache.
- **Render targets** build their attachments once.
- **One command encoder and one submission per sub-frame:** the renderer
  records into the composer's encoder. All buffer writes of a sub-frame land
  before its submission, and the previous sub-frame is already submitted, so
  sharing the same buffers stays correct.
- **The per-object cache is bounded:** a long-lived cache would otherwise
  keep data for objects that never come back (a rebuilt timeline has new
  uids). Every render is a generation; every 1024 renders, entries unused
  for 1024 renders are dropped.

New tests cover the shared instance and composers per effect chain, reuse
and LRU eviction in the resource cache, renders after size changes matching
a fresh renderer, and cache sweeping. The test suite runs faster too (11.2 →
9.6 s), since driver and CLI tests now share one renderer.

Result: +13% (119.6 → 134.9 fps). The composite drops from 0.72 to 0.37 ms
per frame, and `render_layers` from 4.3 to 3.5 ms.
