//! `manimx._core.FrameRenderer`: the core's renderer (render/) as a Python
//! class, driven by `CoreFramePipeline` (manimx/drivers/core_pipeline.py).
//! It takes what every renderer takes: render object trees and
//! `RenderConfig`s, and reads the settings it needs from the config.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::python::robjects::{node_of, transform_from_py};
use crate::render::effects::{Bloom, Effect, ShaderEffect};
use crate::render::error::Error;
use crate::render::frame_renderer::{self, RendererOptions, SubFrame, View};
use crate::render::settings::{Camera, CameraView, ViewSettings};

impl From<Error> for PyErr {
    fn from(error: Error) -> PyErr {
        match error {
            Error::Gpu(message) | Error::Shader(message) => PyRuntimeError::new_err(message),
            Error::Unsupported(message) => PyNotImplementedError::new_err(message),
            Error::Invalid(message) => PyValueError::new_err(message),
        }
    }
}

/// A view as Python hands it over: a render object tree and its
/// `RenderConfig`.
type PyView<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>);

#[pyclass(module = "manimx._core")]
pub struct FrameRenderer {
    renderer: frame_renderer::FrameRenderer,
    size: (usize, usize),
}

#[pymethods]
impl FrameRenderer {
    /// A renderer for frames of `config`'s pixel size, samples, shadow
    /// resolution and background. `effects` is a list of `("bloom",
    /// threshold, knee, intensity, radius, levels)` and `("shader", code,
    /// stage, values)`, in chain order; `tone_map` a `ToneMapOperator`
    /// value. Up to `in_flight` frames are on the GPU at once.
    #[new]
    #[pyo3(signature = (shader_folder, config, tone_map, effects, in_flight, profile=false))]
    fn new(
        shader_folder: PathBuf,
        config: &Bound<'_, PyAny>,
        tone_map: u32,
        effects: Vec<Bound<'_, PyTuple>>,
        in_flight: usize,
        profile: bool,
    ) -> PyResult<Self> {
        let width: u32 = config.getattr("pixel_width")?.extract()?;
        let height: u32 = config.getattr("pixel_height")?.extract()?;
        let background: (f32, f32, f32, f32) = config.getattr("background_rgba")?.extract()?;
        let options = RendererOptions {
            shader_folder,
            width,
            height,
            samples: config.getattr("msaa_samples")?.extract()?,
            shadow_resolution: config.getattr("shadow_resolution")?.extract()?,
            background: [background.0, background.1, background.2, background.3],
            tone_map,
            effects: effects
                .iter()
                .map(parse_effect)
                .collect::<PyResult<Vec<_>>>()?,
            in_flight,
            profile,
        };
        Ok(FrameRenderer {
            renderer: frame_renderer::FrameRenderer::new(options)?,
            size: (width as usize, height as usize),
        })
    }

    /// Render frame `index` from `sub_frames`: a list of `(weight, views)`,
    /// each view a `(root, config)` pair of a render object tree and the
    /// `RenderConfig` to see it with. Returns True when the frame shows
    /// exactly what the frame before showed: then nothing is rendered or
    /// read back for it.
    fn render(&mut self, index: u64, sub_frames: Vec<(f32, Vec<PyView<'_>>)>) -> PyResult<bool> {
        let sub_frames = sub_frames
            .into_iter()
            .map(|(weight, views)| {
                let views = views
                    .iter()
                    .map(|(root, config)| {
                        Ok(View {
                            root: node_of(root)?,
                            settings: view_settings(config)?,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(SubFrame { weight, views })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self.renderer.render(index, &sub_frames)?)
    }

    /// Frames finished so far, `(index, pixels)` in submission order, the
    /// pixels an `(H, W, 4)` uint8 RGBA array; with `wait`, waits for every
    /// frame on the GPU.
    #[pyo3(signature = (wait=false))]
    fn finished<'py>(&mut self, py: Python<'py>, wait: bool) -> PyResult<Bound<'py, PyList>> {
        let done = PyList::empty(py);
        let (width, height) = self.size;
        for (index, image) in self.renderer.finished(wait)? {
            // The array takes the bytes over: no copy.
            let pixels = image.into_pyarray(py).reshape([height, width, 4])?;
            done.append((index, pixels))?;
        }
        Ok(done)
    }

    /// A texture of the last frame as RGBA floats, to look at how it was
    /// made (see inspect.rs): `(width, height, bytes)`. Names: "lit",
    /// "vector", "albedo", "normal", "emissive", "scope", "depth",
    /// "masks" and "shadow_maps" (with `layer`), and the composer's
    /// "view distances" and "blur tiles". `shadow_layers()` says how many
    /// shadow map layers there are.
    #[pyo3(signature = (name, layer=0))]
    fn inspect<'py>(
        &mut self,
        py: Python<'py>,
        name: &str,
        layer: u32,
    ) -> PyResult<(u32, u32, Bound<'py, PyBytes>)> {
        let (width, height, texels) = self.renderer.inspect(name, layer)?;
        Ok((
            width,
            height,
            PyBytes::new(py, bytemuck::cast_slice(&texels)),
        ))
    }

    /// How many shadow map layers there are.
    fn shadow_layers(&self) -> u32 {
        self.renderer.shadow_layers()
    }

    /// GPU milliseconds per pass name since the last call (with `profile`).
    fn gpu_timings<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let timings = PyDict::new(py);
        for (name, ms) in self.renderer.gpu_timings() {
            timings.set_item(name, ms)?;
        }
        Ok(timings)
    }
}

/// What the renderer reads from a `RenderConfig`.
fn view_settings(config: &Bound<'_, PyAny>) -> PyResult<ViewSettings> {
    let get = |name: &str| config.getattr(name);
    let camera_views = get("camera_views")?
        .try_iter()?
        .map(|view| {
            let view = view?;
            Ok(CameraView {
                uid: view.getattr("uid")?.extract()?,
                camera: camera(&view.getattr("camera")?)?,
                width: view.getattr("width")?.extract()?,
                height: view.getattr("height")?.extract()?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let ambient: (f64, f64, f64) = get("ambient_linear")?.extract()?;
    let filter: (f32, f32, f32, f32) = get("shadow_filter")?.extract()?;
    let jitter: (f64, f64) = get("jitter")?.extract()?;
    Ok(ViewSettings {
        camera: camera(&get("camera")?)?,
        camera_views,
        hide_camera_views: get("hide_camera_views")?.extract()?,
        view_aspect: get("view_aspect")?.extract()?,
        jitter: [jitter.0, jitter.1],
        time: get("time")?.extract()?,
        layer_depth_step: get("layer_depth_step")?.extract()?,
        exposure: get("exposure")?.extract()?,
        ambient_linear: [ambient.0, ambient.1, ambient.2],
        default_lights: get("default_lights")?.extract()?,
        shadows: get("shadows")?.extract()?,
        shadow_softness: get("shadow_softness")?.extract()?,
        shadow_samples: get("shadow_samples")?.extract()?,
        shadow_resolution: get("shadow_resolution")?.extract()?,
        shadow_filter: [filter.0, filter.1, filter.2, filter.3],
    })
}

/// A `Camera` (manimx/renderer/config.py).
fn camera(camera: &Bound<'_, PyAny>) -> PyResult<Camera> {
    let get = |name: &str| camera.getattr(name);
    Ok(Camera {
        transform: transform_from_py(Some(&get("transform")?))?,
        frame_height: get("frame_height")?.extract()?,
        field_of_view: get("field_of_view")?.extract()?,
        near: get("near")?.extract()?,
        far: get("far")?.extract()?,
        focus_distance: get("focus_distance")?.extract()?,
        aperture: get("aperture")?.extract()?,
    })
}

fn parse_effect(effect: &Bound<'_, PyTuple>) -> PyResult<Effect> {
    let kind: String = effect.get_item(0)?.extract()?;
    match kind.as_str() {
        "bloom" => Ok(Effect::Bloom(Bloom {
            threshold: effect.get_item(1)?.extract()?,
            knee: effect.get_item(2)?.extract()?,
            intensity: effect.get_item(3)?.extract()?,
            radius: effect.get_item(4)?.extract()?,
            levels: effect.get_item(5)?.extract()?,
        })),
        "shader" => Ok(Effect::Shader(ShaderEffect {
            code: effect.get_item(1)?.extract()?,
            frame_stage: effect.get_item(2)?.extract::<String>()? == "frame",
            values: effect.get_item(3)?.extract()?,
        })),
        _ => Err(PyValueError::new_err(format!("unknown effect {kind:?}"))),
    }
}
