//! `manim._native.WgpuRenderer`: the wgpu renderer behind the `Renderer`
//! protocol (manim/renderer/protocol.py). It takes what every renderer
//! takes, the root of a render object tree and a `RenderConfig`, and
//! returns the frame.
//!
//! The trees are resolved here (resolve/), so the renderer itself is only
//! ever sent each scene's changes in world space. The compositor's
//! pipeline is made for the config's pixel size, samples, shadow
//! resolution, background and GPU, and made again when one of them
//! changes; the scenes are then sent whole again.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::composite::frames::{
    PipelineOptions, RenderPipeline, Samples, SubFrame, View, ViewSource,
};
use crate::composite::output::PixelFormat;
use crate::error::Error;
use crate::gpu::device::Memory;
use crate::python::robjects::{node_of, transform_from_py};
use crate::render::settings::{Camera, CameraView, ShadowMethod, ViewSettings};
use crate::resolve::Resolvers;

impl From<Error> for PyErr {
    fn from(error: Error) -> PyErr {
        match error {
            Error::Gpu(message) | Error::Shader(message) => PyRuntimeError::new_err(message),
            Error::Unsupported(message) => capability_exception(message),
            Error::Invalid(message) => PyValueError::new_err(message),
        }
    }
}

/// The protocol's `CapabilityException` with `message`: what a renderer
/// raises when it is asked for something it cannot do.
fn capability_exception(message: String) -> PyErr {
    Python::attach(|py| {
        match py
            .import("manim.renderer.protocol")
            .and_then(|protocol| protocol.getattr("CapabilityException"))
        {
            Ok(class) => PyErr::from_type(class.cast_into().expect("an exception class"), message),
            Err(error) => error,
        }
    })
}

/// The settings a pipeline is made for; a config with others needs a new
/// one.
#[derive(PartialEq)]
struct PipelineKey {
    width: u32,
    height: u32,
    samples: Samples,
    shadow_resolution: u32,
    gpu: String,
    gpu_memory: String,
}

#[pyclass(module = "manim._native", name = "WgpuRenderer")]
pub struct WgpuRenderer {
    pipeline: Option<(PipelineKey, RenderPipeline)>,
    /// Each scene's resolver: its trees turned into the changes sent.
    resolvers: Resolvers,
    /// The frame rendered last, returned again for a frame that shows the
    /// same (the pipeline does not render it).
    last: Option<Vec<u8>>,
    frames: u64,
}

#[pymethods]
impl WgpuRenderer {
    #[new]
    fn new() -> Self {
        WgpuRenderer {
            pipeline: None,
            resolvers: Resolvers::default(),
            last: None,
            frames: 0,
        }
    }

    /// The tree under `root` seen with `config`: an `(H, W, 4)` uint8
    /// array, sRGB, straight alpha, first row at the top.
    fn render<'py>(
        &mut self,
        py: Python<'py>,
        root: &Bound<'py, PyAny>,
        config: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = pipeline_key(config)?;
        let (width, height) = (key.width, key.height);
        if self.pipeline.as_ref().is_none_or(|(kept, _)| *kept != key) {
            let pipeline = RenderPipeline::new(pipeline_options(&key)?)?;
            // A new pipeline keeps no scene: they are all sent whole again.
            self.pipeline = Some((key, pipeline));
            self.resolvers = Resolvers::default();
            self.last = None;
        }
        let (_, pipeline) = self.pipeline.as_mut().expect("made above");
        let update = self.resolvers.sync(&node_of(root)?)?;
        for scene in self.resolvers.sweep() {
            pipeline.forget(scene);
        }
        let view = View {
            source: ViewSource::Scene(update),
            settings: view_settings(config)?,
            // The `ToneMapOperator` member's value (TONE_MAP_* in color.wgsl).
            tone_map: config.getattr("tone_map")?.getattr("value")?.extract()?,
        };
        let sub_frames = vec![SubFrame {
            weight: 1.0,
            views: vec![view],
        }];
        let (r, g, b, a): (f32, f32, f32, f32) = config.getattr("background_rgba")?.extract()?;
        self.frames += 1;
        let held = pipeline.render(self.frames, [r, g, b, a], sub_frames)?;
        if !held || self.last.is_none() {
            let (_, pixels) = pipeline
                .finished(true)?
                .pop()
                .ok_or_else(|| PyRuntimeError::new_err("the renderer returned no frame"))?;
            self.last = Some(pixels);
        }
        let pixels = self.last.clone().expect("rendered above");
        let array = pixels
            .into_pyarray(py)
            .reshape([height as usize, width as usize, 4])?;
        Ok(array.into_any())
    }
}

fn pipeline_key(config: &Bound<'_, PyAny>) -> PyResult<PipelineKey> {
    Ok(PipelineKey {
        width: config.getattr("pixel_width")?.extract()?,
        height: config.getattr("pixel_height")?.extract()?,
        samples: samples(config.getattr("msaa_samples")?.extract()?)?,
        shadow_resolution: config.getattr("shadow_resolution")?.extract()?,
        gpu: config.getattr("gpu")?.extract()?,
        gpu_memory: config.getattr("gpu_memory")?.extract()?,
    })
}

/// `RenderConfig.msaa_samples`: a sample count, or -1 for the most the
/// GPU supports.
fn samples(value: i64) -> PyResult<Samples> {
    match value {
        -1 => Ok(Samples::Most),
        count @ 1.. => Ok(Samples::Count(count as u32)),
        other => Err(PyValueError::new_err(format!(
            "msaa_samples must be a sample count or -1 (the most the GPU supports), got {other}"
        ))),
    }
}

fn pipeline_options(key: &PipelineKey) -> PyResult<PipelineOptions> {
    let gpu = match key.gpu.as_str() {
        "high-performance" => wgpu::PowerPreference::HighPerformance,
        "low-power" => wgpu::PowerPreference::LowPower,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown GPU choice {other:?}; use \"high-performance\" or \"low-power\""
            )))
        }
    };
    let memory = match key.gpu_memory.as_str() {
        "compact" => Memory::Compact,
        "fast" => Memory::Fast,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown GPU memory choice {other:?}; use \"compact\" or \"fast\""
            )))
        }
    };
    Ok(PipelineOptions {
        width: key.width,
        height: key.height,
        samples: key.samples,
        shadow_resolution: key.shadow_resolution,
        effects: Vec::new(),
        in_flight: 1,
        pixel_format: PixelFormat::Rgba,
        page_bytes: None,
        gpu,
        memory,
    })
}

/// What the renderer reads from a `RenderConfig` per render.
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
    let ambient: (f64, f64, f64) = get("ambient_rgb")?.extract()?;
    let jitter: (f64, f64) = get("jitter")?.extract()?;
    let shadow_method = match get("shadow_method")?
        .getattr("value")?
        .extract::<String>()?
        .as_str()
    {
        "pcss" => ShadowMethod::Pcss,
        "pcf" => ShadowMethod::Pcf,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown shadow method {other:?}"
            )))
        }
    };
    Ok(ViewSettings {
        camera: camera(&get("camera")?)?,
        camera_views,
        hide_camera_views: get("hide_camera_views")?.extract()?,
        view_aspect: get("view_aspect")?.extract()?,
        jitter: [jitter.0, jitter.1],
        time: get("time")?.extract()?,
        layer_depth_step: get("layer_depth_step")?.extract()?,
        exposure: get("exposure")?.extract()?,
        ambient_color: [ambient.0, ambient.1, ambient.2],
        ambient_intensity: get("ambient_intensity")?.extract()?,
        default_lights: get("default_lights")?.extract()?,
        shadows: get("shadows")?.extract()?,
        shadow_softness: get("shadow_softness")?.extract()?,
        shadow_samples: get("shadow_samples")?.extract()?,
        shadow_resolution: get("shadow_resolution")?.extract()?,
        shadow_method,
        shadow_share: get("shadow_share")?.extract()?,
        shadow_index: get("shadow_index")?.extract()?,
    })
}

/// A `Camera` (manim/renderer/protocol.py).
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
