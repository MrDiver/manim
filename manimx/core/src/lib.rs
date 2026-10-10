//! The native core of manimx. CORE.md (in the folder above) explains
//! why it exists and what it will do; README.md here maps the modules.

use pyo3::prelude::*;

mod arena;
mod callbacks;
mod contribution;
mod evaluator;
mod graph;
mod lights;
mod meshes;
mod packer;
mod rates;
mod records;
mod shadows;
mod track;
mod value;
mod vectors;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<evaluator::Evaluator>()?;
    module.add_class::<evaluator::FrameState>()?;
    module.add_class::<graph::SceneGraph>()?;
    module.add_class::<packer::Packer>()?;
    Ok(())
}
