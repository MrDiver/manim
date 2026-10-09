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
| `quat.rs` | quaternion arithmetic as numpy-quaternion does it | numpy-quaternion |

## Rules

* The Python code is the reference. Ported functions compute in the same
  order, so both paths agree to the last bits where possible; the tests
  compare them.
* Whatever the core does not know (a custom `SignalKind`, a rate function,
  an animation without a data form, a `derive()` function) stays a Python
  object and is called back.
