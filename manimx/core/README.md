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
| `render/packing/records.rs` | the GPU record structs (bytemuck), as the WGSL structs declare them | renderer/shaders/ |
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
