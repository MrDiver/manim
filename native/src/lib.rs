//! The native part of manim, imported in Python as `manim._native`.

use pyo3::prelude::*;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
