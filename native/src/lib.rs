//! The native part of manim, imported in Python as `manim._native`.
//!
//! Stages, one folder each:
//!
//! 1. `robjects`: the render object tree, plain values; the interface
//!    between the scene side and every renderer.
//! 2. `resolve`: trees turned into scenes in world space, sent as the
//!    changes since the last tree (`Resolver`).
//! 3. `flat`: a scene in world space and its changes (`FlatScene`,
//!    `Update`): all a renderer is given besides its settings.
//! 4. `render`: the wgpu renderer, one scene seen with one view's
//!    settings rendered into layers; it knows nothing of mobjects,
//!    trees, frames or Python.
//!
//! `cache` holds `Memo`, the one shape every cache of the renderer has;
//! `python` holds the Python classes of these stages.

use pyo3::prelude::*;

mod cache;
mod error;
mod flat;
mod gpu;
mod python;
mod render;
mod resolve;
mod robjects;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    python::robjects::register(module)?;
    Ok(())
}
