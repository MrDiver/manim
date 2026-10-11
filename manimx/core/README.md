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

A frame goes through three stages, one folder each. Every folder's
`mod.rs` starts with a table of its modules.

```text
src/
  lib.rs           the Python module manimx._core
  evaluation/      signals at a time
  packing/         the scene at that time as GPU records
  geometry/        fills and strokes, used by packing
  render/          the records drawn into a frame
    passes/        the passes of one view
    effects/       bloom, custom shaders, depth of field
```

| Module | What it is | Python counterpart |
|---|---|---|
| `lib.rs` | the Python module `manimx._core` | |
| `evaluation/evaluator.rs` | `Evaluator`: all signals of a scene in slots, set to their values at a time; `FrameState`, `Slots` | `PreparedScene.evaluate()`, `ScenePlan` (animation/plan.py) |
| `evaluation/track.rs` | one signal's track and its entries, folded at a time | `SignalTrack`, `Entry`, `Placement` (animation/signal_tracks.py) |
| `evaluation/contribution.rs` | what one animation does to one signal at a progress | `Animation.contribution()`, animation/contributions.py |
| `evaluation/rates.rs` | rate functions | animation/rates.py |
| `evaluation/value.rs` | signal values and the operations of their kinds | animation/kinds.py |
| `evaluation/paths.rs` | the part of a path drawn so far | animation/paths.py |
| `packing/packer.rs` | `Packer`: a scene graph and a `FrameState` in, the records of a packet out | |
| `packing/scene_graph.rs` | `SceneGraph`: nodes, nesting, which slot feeds which field; flattening to drawable leaves | `Node.snapshot()` |
| `packing/vectors.rs` | retained records of `Shape2D` and `Curve3D` | |
| `packing/meshes.rs` | records of `Shape3D` and `Particles` | |
| `packing/lights.rs` | light records, the default light | |
| `packing/shadows.rs` | shadow views, scene bounds, which shadow layers changed | |
| `packing/arena.rs` | growable record arrays mirroring GPU buffers | |
| `packing/records.rs` | the GPU record structs (bytemuck), as the WGSL structs declare them | renderer/shaders/ |
| `packing/callbacks.rs` | the Python functions called once per version | renderer/core_callbacks.py |
| `geometry/tessellation.rs` | fills triangulated with exact curves | |
| `geometry/strokes.rs` | a path's stroke records (curve links, per-point width and color) | `stroke_records` (renderer/core_callbacks.py) |
| `render/frame_renderer.rs` | `FrameRenderer`: frames in flight, held frames, read-back | drivers/core_pipeline.py drives it |
| `render/composer.rs` | sub-frames and views into one frame | |
| `render/renderer.rs` | one view: the passes into the lit and vector layers | |
| `render/passes/` | camera views, vectors and masks, shadow maps, meshes, lighting | |
| `render/effects/` | bloom, shader effects, depth of field | renderer/effects.py (settings) |
| `render/fullscreen.rs` | fullscreen passes, named textures, uniform slots | |
| `render/targets.rs` | every texture a frame renders into | |
| `render/context.rs`, `buffer.rs`, `profiler.rs`, `inspect.rs` | device and shaders, growing buffers, GPU timing, reading textures back | |

Math is [glam](https://docs.rs/glam) (`DMat4`, `DQuat`, `DVec3` in f64,
converted to f32 only when a record is written); records are `#[repr(C)]`
structs written with [bytemuck](https://docs.rs/bytemuck).

## Rules

* Where Python computes the same thing (signal values, stroke records),
  the Python code is the reference. Ported functions compute in the same
  order, so both agree to the last bits where possible; the tests compare
  them (tests/test_plan.py, tests/test_core_strokes.py). The renderer is
  checked against reference images (tests/references/).
* Whatever the core does not know (a custom `SignalKind`, a rate function,
  an animation without a data form, a `derive()` function) stays a Python
  object and is called back.
