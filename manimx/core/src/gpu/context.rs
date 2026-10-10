//! The GPU device and shader loading. Ports `GPUContext` and `load_wgsl`
//! (manimx/renderer/webgpu/gpu.py).
//!
//! The WGSL files stay in the Python package (renderer/webgpu/shaders);
//! the core reads them from there, so there is one copy of every shader.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Linear radiance of lit meshes, before tone mapping.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Color targets of the G-buffer (see shaders/gbuffer.wgsl).
pub const GBUFFER_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Rgba8UnormSrgb, // base color, occlusion
    wgpu::TextureFormat::Rgba16Float,    // world normal, roughness
    wgpu::TextureFormat::Rgba16Float,    // emissive radiance, metallic
    wgpu::TextureFormat::R16Float,       // clip scope (selects the lights), -1 for none
];

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub timestamps: bool,
    shader_folder: PathBuf,
}

/// The process's device, made on first use (see `Gpu::shared`).
static SHARED: Mutex<Option<Arc<Gpu>>> = Mutex::new(None);

impl Gpu {
    /// The device every renderer of this process uses: drivers refuse more
    /// than a few devices at once, and pipelines and textures are cheaper
    /// shared. Like `RenderService.shared()` on the Python side.
    pub fn shared(shader_folder: PathBuf) -> PyResult<Arc<Gpu>> {
        let mut shared = SHARED.lock().expect("not poisoned");
        if let Some(gpu) = shared.as_ref() {
            return Ok(gpu.clone());
        }
        let gpu = Arc::new(Gpu::create(shader_folder)?);
        *shared = Some(gpu.clone());
        Ok(gpu)
    }

    /// A high-performance device with `indirect-first-instance` (draws
    /// select their instance records with it), and timestamp queries when
    /// the adapter has them.
    pub fn create(shader_folder: PathBuf) -> PyResult<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|error| PyRuntimeError::new_err(format!("no GPU adapter: {error}")))?;
        let required = wgpu::Features::INDIRECT_FIRST_INSTANCE;
        if !adapter.features().contains(required) {
            return Err(PyRuntimeError::new_err(
                "GPU adapter lacks indirect-first-instance",
            ));
        }
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let features = if timestamps {
            required | wgpu::Features::TIMESTAMP_QUERY
        } else {
            required
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("manimx core"),
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|error| PyRuntimeError::new_err(format!("no GPU device: {error}")))?;
        Ok(Gpu {
            device,
            queue,
            timestamps,
            shader_folder,
        })
    }

    /// Compile a shader file after `#include` expansion and the given
    /// plain-text `replacements` (small variants).
    pub fn shader(
        &self,
        filename: &str,
        replacements: &[(&str, &str)],
    ) -> PyResult<wgpu::ShaderModule> {
        let mut code = self.load_wgsl(filename, &mut HashSet::new())?;
        for (old, new) in replacements {
            code = code.replace(old, new);
        }
        Ok(self.module(filename, &code))
    }

    pub fn module(&self, label: &str, code: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            })
    }

    /// `load_wgsl`: a file's source with its `#include "file.wgsl"` lines
    /// expanded, each file at most once.
    fn load_wgsl(&self, filename: &str, included: &mut HashSet<String>) -> PyResult<String> {
        included.insert(filename.to_string());
        let path = self.shader_folder.join(filename);
        let code = std::fs::read_to_string(&path).map_err(|error| {
            PyRuntimeError::new_err(format!("cannot read {}: {error}", path.display()))
        })?;
        self.expand_includes(&code, included)
    }

    /// `expand_includes` for WGSL given as a string.
    pub fn expand_includes(&self, code: &str, included: &mut HashSet<String>) -> PyResult<String> {
        let mut out = String::with_capacity(code.len());
        for line in code.split_inclusive('\n') {
            let trimmed = line.trim_end();
            let name = trimmed
                .strip_prefix("#include \"")
                .and_then(|rest| rest.strip_suffix('"'));
            match name {
                Some(name) if included.contains(name) => {
                    out.push_str(&format!("// {name} already included\n"))
                }
                Some(name) => {
                    out.push_str(&self.load_wgsl(name, included)?);
                    if line.ends_with('\n') {
                        out.push('\n');
                    }
                }
                None => out.push_str(line),
            }
        }
        Ok(out)
    }
}
