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

Every module ports one piece of the Python code and names it; read the
Python first, then the module.

| Module | What it is | Python counterpart |
|---|---|---|
| `lib.rs` | the Python module `manimx._core` | |
| `evaluator.rs` | `Evaluator`: all signals of a scene in slots, set to their values at a time | `PreparedScene.evaluate()`, `ScenePlan` (animation/plan.py) |
| `track.rs` | one signal's track and its entries, folded at a time | `SignalTrack`, `Entry`, `Placement` (animation/signal_tracks.py) |
| `contribution.rs` | what one animation does to one signal at a progress | `Animation.contribution()`, animation/contributions.py |
| `rates.rs` | rate functions | animation/rates.py |
| `value.rs` | signal values and the operations of their kinds | animation/kinds.py |
| `graph.rs` | `SceneGraph`: nodes, nesting, which slot feeds which field; flattening to drawable leaves | `Node.snapshot()`, `SceneFlattener` (renderer/scene_graph.py) |
| `packer.rs` | `Packer`: a scene graph and a `FrameState` in, the records of a packet out | `FramePacker` (renderer/webgpu/packer.py), wrapped by `CorePacker` (core_packer.py) |
| `vectors.rs` | retained records of `Shape2D` and `Curve3D` | `RetainedVectors` (renderer/webgpu/retained.py) |
| `meshes.rs` | records of `Shape3D` and `Particles` | `MeshBatch`, `RetainedMeshes` (mesh_batch.py) |
| `lights.rs` | light records, the default light | `light_array`, `default_lights` (mesh_batch.py) |
| `shadows.rs` | shadow views, scene bounds, which shadow layers changed | shadow_views.py, `frame_data` (frame.py) |
| `arena.rs` | growable record arrays mirroring GPU buffers, in numpy byte arrays | `Arena` (renderer/webgpu/arena.py) |
| `records.rs` | the GPU record structs (bytemuck) | the dtypes of vector_batch.py, mesh_batch.py, frame.py |
| `callbacks.rs` | the Python functions called once per version | renderer/core_callbacks.py |
| `paths.rs` | the part of a path drawn so far | animation/paths.py |
| `gpu/` | the renderer: passes, composer, readback (its `mod.rs` maps it) | renderer/webgpu/, drivers/pipeline.py |

Math is [glam](https://docs.rs/glam) (`DMat4`, `DQuat`, `DVec3` in f64,
converted to f32 only when a record is written); records are `#[repr(C)]`
structs written with [bytemuck](https://docs.rs/bytemuck).

## Rules

* The Python code is the reference. Ported functions compute in the same
  order, so both paths agree to the last bits where possible; the tests
  compare them.
* Whatever the core does not know (a custom `SignalKind`, a rate function,
  an animation without a data form, a `derive()` function) stays a Python
  object and is called back.
