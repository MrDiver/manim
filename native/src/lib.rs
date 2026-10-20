//! The native part of manim, imported in Python as `manim._native`.
//!
//! Stages, one folder each:
//!
//! 1. `robjects`: the render object tree, plain values; the interface
//!    between the scene side and every renderer.
//!
//! `python` holds the Python classes of these stages.

use pyo3::prelude::*;

mod python;
mod robjects;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    python::robjects::register(module)?;
    Ok(())
}
