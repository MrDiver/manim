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
folder's `mod.rs` starts with a table of its modules.

```text
src/
  lib.rs           the Python module manimx._core
  evaluation/      signals at a time, and the render object tree of them
  robjects/        the render object tree: the interface
  render/          a tree and its settings drawn into a frame
    packing/       the tree as GPU records
    geometry/      fills and strokes, used by packing
    passes/        the passes of one view
    effects/       bloom, custom shaders, depth of field
  python/          the Python classes of robjects/ and render/
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
| `python/frame_renderer.rs` | `FrameRenderer`, the Python class of the renderer; reads a `RenderConfig` | drivers/core_pipeline.py drives it |
| `render/frame_renderer.rs` | the render chain: frames in flight, held frames, read-back | |
| `render/settings.rs` | `ViewSettings`: camera, camera views, lighting and shadow settings | `RenderConfig`, `Camera` (renderer/config.py) |
| `render/packing/packer.rs` | `Packer`: one view's tree in, the records of a packet out | |
| `render/packing/flatten.rs` | the tree's drawable leaves with world matrices, clip scopes and lights | |
| `render/packing/vectors.rs` | retained records of `RShape2D` and `RCurve3D` | |
| `render/packing/meshes.rs` | records of `RShape3D` and `RParticles` | |
| `render/packing/lights.rs` | light records, the default light | |
| `render/packing/shadows.rs` | shadow views, scene bounds, which shadow layers changed | |
| `render/packing/arena.rs` | growable record arrays mirroring GPU buffers | |
| `render/packing/records.rs` | the GPU record structs (bytemuck), as the WGSL structs declare them | core/shaders/ |
| `render/geometry/tessellation.rs` | fills triangulated with exact curves | |
| `render/geometry/strokes.rs` | a path's stroke records (curve links, per-point width and color) | |
| `render/composer.rs` | sub-frames and views into one frame | |
| `render/renderer.rs` | one view: the passes into the lit and vector layers | |
| `render/passes/` | camera views, vectors and masks, shadow maps, meshes, lighting, texture mip levels | |
| `render/effects/` | bloom, shader effects, depth of field | renderer/effects.py (settings) |
| `render/fullscreen.rs` | fullscreen passes, named textures, uniform slots | |
| `render/targets.rs` | every texture a frame renders into | |
| `render/context.rs`, `buffer.rs`, `color.rs`, `profiler.rs`, `inspect.rs`, `error.rs` | device and shaders, growing buffers, sRGB, GPU timing, reading textures back, errors | |

Math is [glam](https://docs.rs/glam) (`DMat4`, `DQuat`, `DVec3` in f64,
converted to f32 only when a record is written); records are `#[repr(C)]`
structs written with [bytemuck](https://docs.rs/bytemuck).

## The renderer: target architecture

Agreed on 2026-10-13; the code moves there step by step. Until then the
module table above describes what exists.

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
| Output | `Output` (a trait) | `Image` | the frame where it goes | staging buffers, an encoder | GPU, then CPU |

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
  `MaskPass`, `ShadowPass`, `GeometryPass`, `LightingPass`,
  `VectorPass`, `TranslucentPass`. A pass decides the target and which
  geometry it draws in which role, and nothing else. Shadows are only
  in `ShadowPass`; camera view pictures are an input of `VectorPass`.

The shape math of packing (fills, strokes) is CPU work and belongs to
`pack/shapes/`.

### Output

Output finishes the frame for where it goes, so read-back and encoding
are one stage:

| `Output` | What it does | Where |
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

### From today's names

| Today | Target |
|---|---|
| `FrameRenderer` (render/frame_renderer.rs) | `RenderPipeline` |
| `Packer` (render/packing/) | `Packer` (pack/) |
| `Renderer` (render/renderer.rs) and render/passes/ | `Drawer` (draw/), its geometry and passes |
| `Composer` (render/composer.rs), render/effects/ | `Compositor` (composite/) |
| read-back in `FrameRenderer`, drivers/sinks.py | `Output` (output/) |
| `RenderConfig`, `RendererOptions`, `ViewSettings` | `OutputConfig`, `ViewConfig` |
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
