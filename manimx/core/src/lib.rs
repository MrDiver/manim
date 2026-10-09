//! The native core of manimx (see CORE.md in the repository folder).

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
