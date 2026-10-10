# manimx core: the timeline as data, evaluated in Rust

Plan and status of moving everything below the authoring API out of
Python. Read ARCHITECTURE.md first; this file only describes what changes.

## Why

Profile of `light-show` (1080p, motion blur 8, 69.8 ms CPU per frame, GPU
busy 19 ms): 25% `MeshBatch`, 24% evaluation and snapshots, 16% shadow
views, 12% wgpu-py bindings, 10% `SceneFlattener`, 10% other renderer
Python. The GPU waits 2.3% of the time. Every one of those costs is Python
touching every object on every (sub-)frame. Caching (identity keys, frame
hold, retained arenas) only works around it.

## The decision

`Scene.construct()` keeps running once in Python and keeps its API.
`Scene.prepare()` already turns it into tracks of placements per
`RootSignal`. That result becomes **data** (a `ScenePlan`), and a native
core (`manimx._core`, Rust, PyO3) evaluates it at any time and packs the
GPU records directly: no `RNode` tree, no `FlatScene`, no per-object Python
per frame.

Python stays per frame only where the user wrote Python that must run per
frame:

* a `DerivedSignal` with a Python function (`derive(...)`), unless it is
  one the core knows (`follow()` offsets);
* an `Animation` whose `contribution()` has no data form (custom
  animations);
* a rate function the core does not know.

The core calls those back with the GIL it already holds, only for the
signals that need them, so the per-frame cost grows with the user's
Python, not with the number of objects.

The `RNode` path stays as the reference implementation: tests compare the
core's values and pixels with it, until it can be retired.

## What stays

* The authoring API: `Node` classes, signals, `Animation`, `Scene`,
  `Timeline`, `Clip`, events, cues, `Simulation` (already baked to data).
* The split: scene side and render side meet in one contract; today the
  `RNode` tree, in the core a `ScenePlan` plus evaluated values.
* The render chain's stages and the GPU records (`OBJECT_DTYPE`,
  `MESH_OBJECT_DTYPE`, ...): the WGSL shaders do not change.
* Purity: `frame_at(t)` depends only on `t`.

## Phases

Every phase ends with the tests green, a pixel comparison with the `RNode`
path, a measurement (`scripts/benchmark_project.py`, light-show 1 s
profile) and a commit. Only measured gains stay (PERFORMANCE.md).

### 0. Crate and build

* `manimx/core/`: crate `manimx-core`, PyO3 (abi3, Python >= 3.11) and
  `numpy`, built as the extension module `manimx._core`.
* `pyproject.toml`: build backend maturin (mixed Python/Rust project).
  Development: `maturin develop --release` into the active venv.

### 1. The timeline as data

* `ScenePlan` (Python, `manimx/animation/plan.py`) is compiled from a
  `PreparedScene`: a table of signals (slot, `SignalKind`, value), the
  tracks with their entries, the derived signals, and the nodes (class,
  parent, children, which slot feeds which render field).
* **Contributions as data.** `Animation.contribution()` stays the
  definition. Built-in animations also describe it as data
  (`Animation.plan(signal)` returns a `Contribution` value, or `None` for
  "call me"): to a value, by an amount, a progress ramp `a + b * alpha`,
  a rotation by an angle about an axis, keyframes (`Simulation`), a morph
  between aligned paths, a partial path. Both forms must agree; a test
  evaluates every built-in both ways.
* **Rates as data:** `linear`, `smooth` and the userland rate functions
  the core implements are recognized by identity; others are called back.
* **Rust:** `ScenePlan` is loaded once; `evaluate(t, global_time)` folds
  every track (same algorithm as `SignalTrack._fold`) into typed value
  arrays and marks which slots changed since the last call (versions,
  like today).
* Test: for every scene in the test suite, `example-project`, `benchmark`
  and `light-show`, the core's values equal `SignalTrack.value_at` at many
  times.

### 2. Morph and Write

New userland animations, defined in Python (`contribution()`) and as data,
so they work on both paths:

* `Morph(source, target)` (Manim's `Transform`/`ReplacementTransform`):
  the paths are aligned once when the animation is created (curves
  subdivided to equal counts per subpath, subpaths paired, missing ones
  grown from a point, like Manim's `align_points`); per frame it is a
  point-wise interpolation of points, fill and stroke.
* `Create`/`Uncreate`: a partial path (the first `alpha` of the curves,
  the last one cut with de Casteljau).
* `Write`/`DrawBorderThenFill`: stroke drawn as a partial path, then the
  fill fades in, lagged over the shapes of a group (Manim's timing).
* Typst and LaTeX (later) produce `Group`s of `Shape2D`, so they get
  `Write` and `Morph` for free.

### 3. The core packs 2D

* The core keeps the node table, computes world matrices and alphas
  (today's `_walk`), and keeps the retained vector records itself
  (`RetainedVectors` in Rust: slots, arenas, dirty ranges).
* Fill tessellation and stroke topology are ported (earcut port must
  triangulate exactly like `mapbox_earcut`; until then they are called
  back and cached per points version, which is rare).
* Output: a `FramePacket` whose arrays are views of the core's arenas;
  `WGPURenderer` renders it unchanged. `FrameComposer` takes a source of
  packets instead of a function returning trees; the frame hold becomes
  "no slot changed".

### 4. The core packs 3D

`Shape3D` meshes and materials, `Particles`, lights, shadow views and
`ShadowLayers`, `ClippingGroup` scopes, `Curve3D`. Scene bounds are kept
incrementally.

### 5. Switch over

`drivers.render()`, `manimx ui` and the CLI use the core by default; the
`RNode` path remains for tests and as a fallback with `--no-core`.

### 6. The renderer in Rust

`WGPURenderer`, the passes and `FrameComposer` ported to wgpu in Rust with
the same WGSL; readback and the frame pipeline in Rust. This removes the
wgpu-py bindings (12%) and the remaining per-pass Python.

### 7. Later

* Motion blur sub-frames in one submission.
* Frames rendered in parallel, frame cache by content across runs.
* Hardware video encoding without readback.
* The core compiled to WASM with WebGPU for `manimx ui` playback.

## Status

| Phase | State |
|---|---|
| 0 Crate and build | done |
| 1 Timeline as data | done: evaluator, ScenePlan, data contributions |
| 2 Morph and Write | done: `Morph`, `Create`, `Uncreate`, `DrawBorderThenFill`, `Write` on the `Between` and `Partial` data forms |
| 3 Core packs 2D | done: graph, retained vector records |
| 4 Core packs 3D | done: meshes, particles, lights, shadow views and layers |
| 5 Switch over | done: `performance.core` (on by default) for `drivers.render()`, workers and `manimx ui` |
| 6 Renderer in Rust | |
| 7 Later | |
