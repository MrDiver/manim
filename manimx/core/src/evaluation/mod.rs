//! Evaluation: every signal of a prepared scene set to its value at any
//! time, in any order, and the render object tree for those values. The
//! Rust side of `ScenePlan` (manimx/animation/plan.py).
//!
//! | Module | What it does |
//! |---|---|
//! | `evaluator` | `Evaluator`: all signals in one table of slots; `FrameState`, the values at one time |
//! | `scene_graph` | `SceneGraph`: the scene's nodes and their slots; the render object tree of a `FrameState` |
//! | `track` | one signal's track: its base value and the entries of the animations that change it |
//! | `contribution` | what one animation does to one signal at a progress |
//! | `rates` | rate functions (`linear`, `smooth`) |
//! | `value` | signal values and what each kind of value can do (add, interpolate, compare) |
//! | `paths` | the part of a path drawn so far (`Create`, `Write`) |

pub mod contribution;
pub mod evaluator;
pub mod paths;
pub mod rates;
pub mod scene_graph;
pub mod track;
pub mod value;
