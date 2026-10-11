//! The native core of manimx, imported in Python as `manimx._core`.
//!
//! A frame goes through three stages, one folder each:
//!
//! 1. `evaluation`: every signal of a scene set to its value at a time
//!    (the `Evaluator`, `FrameState`).
//! 2. `packing`: the scene graph with those values turned into the
//!    records the GPU reads (the `Packer`); `geometry` holds the shape
//!    math it uses (fills as triangles, strokes as curve records).
//! 3. `render`: the records drawn on the GPU into a frame
//!    (the `FrameRenderer`).
//!
//! CORE.md (in the folder above) explains why the core exists;
//! README.md here maps every module.

use pyo3::prelude::*;

mod evaluation;
mod geometry;
mod packing;
mod render;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<evaluation::evaluator::Evaluator>()?;
    module.add_class::<evaluation::evaluator::FrameState>()?;
    module.add_class::<evaluation::evaluator::Slots>()?;
    module.add_class::<packing::scene_graph::SceneGraph>()?;
    module.add_class::<render::frame_renderer::FrameRenderer>()?;
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
    let style = geometry::strokes::StrokeStyle {
        colors: &colors,
        opacities: &opacities,
        widths: &widths,
        joints: &joints,
        cap,
    };
    let stroke = geometry::strokes::stroke_records(&flat, dimension, &style);
    Ok((
        pyo3::types::PyBytes::new(py, &stroke.curves),
        pyo3::types::PyBytes::new(py, &stroke.points),
        stroke.visible,
        stroke.opaque,
    ))
}
