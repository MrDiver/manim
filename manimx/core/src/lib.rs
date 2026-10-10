//! The native core of manimx. CORE.md (in the folder above) explains
//! why it exists and what it will do; README.md here maps the modules.

use pyo3::prelude::*;

mod arena;
mod callbacks;
mod contribution;
mod evaluator;
mod gpu;
mod graph;
mod lights;
mod meshes;
mod packer;
mod paths;
mod rates;
mod records;
mod shadows;
mod tessellation;
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
    module.add_class::<gpu::frame_renderer::FrameRenderer>()?;
    module.add_function(wrap_pyfunction!(tessellate_fill, module)?)?;
    Ok(())
}

/// The triangulated fill of 2D path points (`(N, 2)` floats, four per
/// cubic) as `FILL_VERTEX_DTYPE` records (renderer/tessellation.py).
#[pyfunction]
fn tessellate_fill<'py>(
    py: Python<'py>,
    points: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let points = points.as_array();
    let corners: Vec<glam::DVec2> = points
        .rows()
        .into_iter()
        .map(|row| glam::DVec2::new(row[0], row[1]))
        .collect();
    let vertices = tessellation::tessellate_fill(&corners);
    Ok(pyo3::types::PyBytes::new(
        py,
        bytemuck::cast_slice(&vertices),
    ))
}
