//! The Python functions the core calls back for records it builds once
//! per version (manimx/renderer/core_callbacks.py).

use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub struct Callbacks {
    pub path_center: Py<PyAny>,
    pub path_box: Py<PyAny>,
    pub fill_colors: Py<PyAny>,
    pub stroke_records: Py<PyAny>,
    pub mesh_records: Py<PyAny>,
    pub material_entry: Py<PyAny>,
    pub light_color: Py<PyAny>,
    pub default_light: Py<PyAny>,
}

impl Callbacks {
    pub fn load(py: Python<'_>) -> PyResult<Callbacks> {
        let module = py.import("manimx.renderer.core_callbacks")?;
        let function = |name: &str| -> PyResult<Py<PyAny>> { Ok(module.getattr(name)?.unbind()) };
        Ok(Callbacks {
            path_center: function("path_center")?,
            path_box: function("path_box")?,
            fill_colors: function("fill_colors")?,
            stroke_records: function("stroke_records")?,
            mesh_records: function("mesh_records")?,
            material_entry: function("material_entry")?,
            light_color: function("light_color")?,
            default_light: function("default_light")?,
        })
    }
}

/// The contents of a Python `bytes` object.
pub fn bytes_of(object: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    Ok(object.cast::<PyBytes>()?.as_bytes().to_vec())
}
