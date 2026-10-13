# manimx-core

The native core of manimx, imported in Python as `manimx._core`. Why it
exists and where it is going: [../CORE.md](../CORE.md).

## Build and test

```sh
# from manimx/, with the environment manimx is installed in active
maturin develop --release        # builds and installs manimx._core
pytest tests/test_plan.py        # the core against the Python reference
cargo clippy --release           # in core/
```

Python must be rebuilt into the environment after every change to the
Rust sources; `import manimx._core` otherwise still loads the old module.

## Modules

Three parts, one folder each, and one interface between them: the render
object tree (`robjects/`). The renderer is given a tree and its settings
and nothing else; it never sees signals, slots or timelines. Every
folder's `mod.rs` says in a few lines what it holds.

```text
src/
  lib.rs           the Python module manimx._core
  evaluation/      signals at a time, and the render object tree of them
  robjects/        the render object tree: the interface
  render/          a tree and its settings drawn into a frame
    pipeline.rs    RenderPipeline: runs the stages
    pack/          Pack: the tree as GPU records (shapes/: fills, strokes)
    draw/          Draw: geometry/ (what is drawn), passes/ (in order)
    composite/     Composite: effects, compositing, accumulating
    output/        Output: RGBA or NV12, read back
    gpu/           device, shaders, buffers, fullscreen passes, timing
  python/          the Python classes of robjects/ and render/
shaders/           the WGSL shaders, compiled into the library
```

```text
evaluation/   signals at a time ──snapshot──▶  robjects/  ◀──reads──  render/
```

| Module | What it is | Python counterpart |
|---|---|---|
| `lib.rs` | the Python module `manimx._core` | |
| `evaluation/evaluator.rs` | `Evaluator`: all signals of a scene in slots, set to their values at a time; `FrameState` | `PreparedScene.evaluate()`, `ScenePlan` (animation/plan.py) |
| `evaluation/scene_graph.rs` | `SceneGraph`: the scene's nodes and the slots of their fields; the render object tree of a `FrameState` | `Node.render_kind`, `Node.render_fields()`, `Node.snapshot()` |
| `evaluation/track.rs` | one signal's track and its entries, folded at a time | `SignalTrack`, `Entry`, `Placement` (animation/signals/signal_tracks.py) |
| `evaluation/contribution.rs` | what one animation does to one signal at a progress | `Animation.contribution()`, animation/signals/contributions.py |
| `evaluation/rates.rs` | rate functions | animation/signals/rates.py |
| `evaluation/value.rs` | signal values and the operations of their kinds | animation/signals/kinds.py |
| `evaluation/paths.rs` | the part of a path drawn so far | animation/signals/paths.py |
| `robjects/mod.rs` | `RNode` and its kinds, `Transform`, uids and versions | `manimx.robjects` |
| `robjects/values.rs` | points, `Fill`, `Stroke`, `Mesh`, `Material`, `Texture`, `SurfaceShader`, particles | `Fill`, `Stroke`, `MeshData`, `PbrMaterial`, ... |
| `python/robjects.rs` | the Python classes of the render objects | `manimx.robjects` |
| `python/render_pipeline.rs` | `RenderPipeline`, the Python class of the renderer; reads a `RenderConfig` | drivers/core_pipeline.py drives it |
| `render/pipeline.rs` | `RenderPipeline`: sub-frames and views through the stages, held frames, stage timing | |
| `render/settings.rs` | `ViewSettings`: camera, camera views, lighting and shadow settings | `RenderConfig`, `Camera` (renderer/config.py) |
| `render/pack/packer.rs` | `Packer`: one view's tree in, a `Packet` of draws and changed records out | |
| `render/pack/flatten.rs` | the tree's drawable leaves with world matrices, clip scopes and lights | |
| `render/pack/vectors.rs` | retained records of `RShape2D` and `RCurve3D` | |
| `render/pack/meshes.rs` | records of `RShape3D` and `RParticles` | |
| `render/pack/lights.rs`, `shadows.rs` | light records and the default light; shadow views, scene bounds, which shadow layers changed | |
| `render/pack/arena.rs`, `records.rs` | growable record arrays mirroring GPU buffers; the GPU record structs (bytemuck), as the WGSL structs declare them | shaders/ |
| `render/pack/shapes/` | fills as triangles (lyon, nonzero rule), strokes as linked curve records | |
| `render/draw/mod.rs` | `Drawer`: prepares the geometry, runs the passes of one view | |
| `render/draw/geometry/` | `VectorGeometry`, `MeshGeometry`: buffers, pipelines, draws per role; images, camera views, material textures | |
| `render/draw/passes/` | `mask`, `shadow`, `gbuffer`, `lighting`, `vector`, `translucent` | |
| `render/draw/targets.rs` | every texture a view is drawn into | |
| `render/composite/` | `Compositor`: lit-stage effects, compositing, depth of field, accumulating, frame-stage effects; `effects/` | renderer/effects.py (settings) |
| `render/output/` | `Output`, `PixelFormat`: the frame as RGBA or NV12, staging buffers, read-back | drivers/sinks.py (`PixelFormat`) |
| `render/gpu/` | the device, the shader table, growing buffers, fullscreen passes, GPU and CPU timing, reading textures back | |

Math is [glam](https://docs.rs/glam) (`DMat4`, `DQuat`, `DVec3` in f64,
converted to f32 only when a record is written); records are `#[repr(C)]`
structs written with [bytemuck](https://docs.rs/bytemuck).

## The renderer: target architecture

Agreed on 2026-10-13. Done: the stage folders and names, Output, the
geometry and passes of Draw, the shaders compiled in. Still to come:
the settings in Rust (`OutputConfig`, `ViewConfig`) and a queued way to
run the stages, which the measurements so far do not call for
(PERFORMANCE.md).

### A pipeline of stages

A frame passes through named stages, each with one input and one output
and the state it keeps between frames. Evaluating the scene is not one
of them: that is the scene side (`evaluation/`, and Python for user
code). The renderer starts at a finished render object tree.

```text
drivers.render():  Evaluate ─▶ [ RenderPipeline: Pack ─▶ Draw ─▶ Composite ─▶ Output ]
                   evaluation/   render/ (no Python, no signals)
```

| Stage | Struct | In | Out | Keeps between frames | Runs on |
|---|---|---|---|---|---|
| Evaluate | `Evaluator`, `SceneGraph` (evaluation/) | a time | `FrameScene`: per sub-frame its weight and views, each a tree and its `ViewConfig` | the slots | the caller (Python worker processes) |
| Pack | `Packer` | `FrameScene` | `Packet`: draw lists and the record ranges that changed | path and mesh caches, the arenas the GPU buffers mirror | CPU |
| Draw | `Drawer` | `Packet` | `Layers`: lit (HDR), vector, depths | GPU buffers, pipelines, render targets | GPU |
| Composite | `Compositor` | `Layers` of every view and sub-frame | `Image`: the accumulated frame after its effects | effect textures | GPU |
| Output | `Output` | `Image` | the frame where it goes | staging buffers, an encoder | GPU, then CPU |

`RenderPipeline` owns the stages and passes frames between them:
`submit(index, frame)`, then `receive()`. The Python class is a thin
wrapper over it.

Names follow the stages: a struct is named for what its stage does, and
its module for the struct (`pack/`, `draw/`, `composite/`, `output/`).
There is no struct called just "renderer".

### Queues and concurrency

Each stage is a plain struct with one method from its input to its
output, so the same stages run two ways:

* **in sequence:** `RenderPipeline` calls Pack, Draw, Composite and
  Output one after the other on one thread (frames still overlap on the
  GPU, as `in_flight` does now). This is the reference, and what the
  browser runs (WASM threads need cross-origin isolation).
* **queued:** Pack on its own thread, bounded queues between it and the
  GPU stages, Output's read-back and encoding on another thread. Pack
  works ahead on frames N+1, N+2 while the GPU draws frame N.

Pack may run ahead although it keeps arenas: a `Packet` carries the
record ranges that changed, and Draw uploads them in packet order before
its passes, so the GPU buffers go through the same states as in
sequence. Draw and Composite share the GPU queue and stay in order.

Which way is faster is measured, not assumed (PERFORMANCE.md).

### Inside Draw: geometry and passes

Two kinds of things, kept apart:

* **Geometry** (`draw/geometry/`): one module per kind of drawn object,
  holding its GPU buffers, pipelines and bind groups, and a
  `draw(pass, role)` that draws it in a role (color, mask, shadow,
  G-buffer). `VectorGeometry` draws `RShape2D` and `RCurve3D`,
  `MeshGeometry` draws `RShape3D` and `RParticles`.
* **Passes** (`draw/passes/`): one per step, in the order they run:
  mask, shadow, gbuffer, lighting, vector, translucent. A pass decides
  the target and which geometry it draws in which role, and nothing
  else. Shadows are only
  in `ShadowPass`; camera view pictures are an input of `VectorPass`.

The shape math of packing (fills, strokes) is CPU work and lives in
`pack/shapes/`.

### Output

Output finishes the frame for where it goes, so read-back and encoding
are one stage. Today `Output` (output/) is the first row below; the
others would join it as kinds of output:

| Output | What it does | Where |
|---|---|---|
| `FramesOutput` | RGBA or NV12 read back to the caller | Python sinks, preview, tests |
| `PipeOutput` | NV12 read back and piped to ffmpeg | desktop video |
| `WebCodecsOutput` | a `VideoFrame` from the canvas texture, no read-back | browser |

Encoding on the GPU without read-back on the desktop (Vulkan Video,
NVENC through CUDA) would be another `Output`; it is not planned.

### Settings and shaders

* The settings are defined in Rust and exposed to Python as classes:
  `OutputConfig` (for the whole pipeline: size, MSAA, background, tone
  map, effects, shadow resolution, pixel format) and `ViewConfig` (per
  view: camera, camera views, lights and shadow settings, time, jitter),
  with `Camera`, `Bloom`, `ShaderEffect` and `ToneMap`. They replace
  `RenderConfig`, `RendererOptions`, `ViewSettings` and the converters in
  python/frame_renderer.rs.
* The WGSL shaders live in `core/shaders/` and are compiled into the
  library with `include_str!`; `#include` lines are expanded from that
  table, with no folder to read at run time (there is none in the
  browser).

### Former names

| Before | Now |
|---|---|
| `FrameRenderer` (render/frame_renderer.rs) | `RenderPipeline` |
| `Packer` (render/packing/) | `Packer` (pack/) |
| `Renderer` (render/renderer.rs) and render/passes/ | `Drawer` (draw/), its geometry and passes |
| `Composer` (render/composer.rs), render/effects/ | `Compositor` (composite/) |
| read-back in `FrameRenderer`, drivers/sinks.py | `Output` (output/) |
| `RenderConfig`, `RendererOptions`, `ViewSettings` | `RenderConfig`, `PipelineOptions`, `ViewSettings`; to become `OutputConfig`, `ViewConfig` |
| manimx/renderer/shaders/ | core/shaders/ |

## Rules

* The renderer reads render object trees and `ViewSettings`, nothing
  else (no `evaluation`, no `pyo3`): any tree, built by the core or by
  hand in Python, is drawn the same way. The renderer is checked against
  reference images (tests/references/).
* Where Python computes the same thing (signal values), the Python code
  is the reference. Ported functions compute in the same order, so both
  agree to the last bits where possible; tests/test_plan.py compares
  them.
* Whatever the core does not know (a custom `SignalKind`, a rate function,
  an animation without a data form, a `derive()` function) stays a Python
  object and is called back.
