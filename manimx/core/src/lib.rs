//! The native core of manimx, imported in Python as `manimx._core`.
//!
//! Three parts, one folder each, with one interface between them: the
//! render object tree (`robjects`), which is all a renderer is given
//! besides its settings.
//!
//! ```text
//! evaluation/   signals at a time ──snapshot──▶  robjects/  ◀──reads──  render/
//! ```
//!
//! 1. `evaluation`: every signal of a scene set to its value at a time
//!    (the `Evaluator`, `FrameState`), and the render object tree of
//!    those values (`SceneGraph::snapshot`).
//! 2. `robjects`: the render object tree, plain values. In Python it is
//!    `manimx.robjects` (python/robjects.rs).
//! 3. `render`: a tree and its settings drawn on the GPU into a frame
//!    (`FrameRenderer`); it knows nothing of signals or Python.
//!
//! `python` holds the Python classes of `robjects` and `render`.
//! CORE.md (in the folder above) explains why the core exists;
//! README.md here maps every module.

use pyo3::prelude::*;

mod evaluation;
mod python;
mod render;
mod robjects;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<evaluation::evaluator::Evaluator>()?;
    module.add_class::<evaluation::evaluator::FrameState>()?;
    module.add_class::<evaluation::scene_graph::SceneGraph>()?;
    module.add_class::<python::frame_renderer::FrameRenderer>()?;
    python::robjects::register(module)?;
    Ok(())
}
