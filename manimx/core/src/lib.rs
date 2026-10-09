//! The native core of manimx. CORE.md (in the folder above) explains
//! why it exists and what it will do; README.md here maps the modules.

use pyo3::prelude::*;

mod contribution;
mod evaluator;
mod quat;
mod rates;
mod track;
mod value;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<evaluator::Evaluator>()?;
    Ok(())
}
