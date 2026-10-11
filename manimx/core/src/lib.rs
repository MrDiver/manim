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
mod strokes;
mod tessellation;
mod track;
mod value;
mod vectors;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<evaluator::Evaluator>()?;
    module.add_class::<evaluator::FrameState>()?;
    module.add_class::<evaluator::Slots>()?;
    module.add_class::<graph::SceneGraph>()?;
    module.add_class::<gpu::frame_renderer::FrameRenderer>()?;
    module.add_function(wrap_pyfunction!(stroke_records, module)?)?;
    Ok(())
}

/// A path's stroke records built in Rust (strokes.rs): `(curves, points,
/// visible, opaque)` like `stroke_records` in renderer/core_callbacks.py,
/// for points `(N, 2 or 3)`, RGB rows, opacities, widths, joint values
/// and a cap value.
#[pyfunction]
fn stroke_records<'py>(
    py: Python<'py>,
    points: numpy::PyReadonlyArray2<'py, f64>,
    colors: numpy::PyReadonlyArray2<'py, f64>,
    opacities: Vec<f64>,
    widths: Vec<f64>,
    joints: Vec<u32>,
    cap: u32,
) -> PyResult<(
    Bound<'py, pyo3::types::PyBytes>,
    Bound<'py, pyo3::types::PyBytes>,
    bool,
    bool,
)> {
    let points = points.as_array();
    let dimension = points.ncols();
    let flat: Vec<f64> = points.iter().copied().collect();
    let colors: Vec<[f64; 3]> = colors
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let style = strokes::StrokeStyle {
        colors: &colors,
        opacities: &opacities,
        widths: &widths,
        joints: &joints,
        cap,
    };
    let stroke = strokes::stroke_records(&flat, dimension, &style);
    Ok((
        pyo3::types::PyBytes::new(py, &stroke.curves),
        pyo3::types::PyBytes::new(py, &stroke.points),
        stroke.visible,
        stroke.opaque,
    ))
}
