//! The GPU device and shader loading (`#include "file.wgsl"` lines are
//! expanded, each file at most once per shader).
//!
//! The WGSL files are read from a folder given when the device is made
//! (manimx passes its package's renderer/shaders), so there is one copy
//! of every shader.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::render::error::{Error, Result};

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Shadow maps: depth, and a stencil for the masks of clipped casters.
pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32FloatStencil8;
/// How much light translucent casters let through, per shadow map texel
/// (multiplied together; 1 without any).
pub const TRANSMITTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// The nearest translucent caster per shadow map texel, as a linear depth
/// (`translucent_depth` in mesh.wgsl); half floats blend on every device.
pub const TRANSLUCENT_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// The lighting pass's stencil, which marks the pixels on an edge.
pub const EDGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Stencil8;
/// Linear radiance of lit meshes, before tone mapping.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Color targets of the G-buffer (see shaders/gbuffer.wgsl).
pub const GBUFFER_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Rgba8UnormSrgb, // base color, occlusion
    wgpu::TextureFormat::Rgba16Float,    // world normal, roughness
    wgpu::TextureFormat::Rgba16Float,    // emissive radiance, metallic
    wgpu::TextureFormat::R16Float,       // clip scope (selects the lights), -1 for none
];

/// How a shadow pass draw uses the stencil of the shadow map, which holds
/// the masks of clip scopes (`ShadowStencil`, see shadow_pass.rs).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShadowStencil {
    /// An unclipped caster: depth only, the stencil is ignored.
    None,
    /// A clipped caster: only where the stencil equals the reference.
    Clipped,
    /// The outermost mask of a clip scope: writes the reference, no depth.
    Mask,
    /// A nested mask: where the stencil equals the reference, counts it up
    /// by one; no depth.
    InnerMask,
}

/// The depth-stencil state of a shadow pass pipeline (`shadow_depth_stencil`).
pub fn shadow_depth_stencil(stencil: ShadowStencil) -> wgpu::DepthStencilState {
    use wgpu::{CompareFunction as Compare, StencilOperation as Op};
    let (compare, pass_op) = match stencil {
        ShadowStencil::None => (Compare::Always, Op::Keep),
        ShadowStencil::Clipped => (Compare::Equal, Op::Keep),
        ShadowStencil::Mask => (Compare::Always, Op::Replace),
        ShadowStencil::InnerMask => (Compare::Equal, Op::IncrementClamp),
    };
    let face = wgpu::StencilFaceState {
        compare,
        fail_op: Op::Keep,
        depth_fail_op: Op::Keep,
        pass_op,
    };
    let mask = matches!(stencil, ShadowStencil::Mask | ShadowStencil::InnerMask);
    wgpu::DepthStencilState {
        format: SHADOW_FORMAT,
        depth_write_enabled: Some(!mask),
        depth_compare: Some(if mask { Compare::Always } else { Compare::Less }),
        stencil: wgpu::StencilState {
            front: face,
            back: face,
            read_mask: 0xFF,
            write_mask: 0xFF,
        },
        bias: wgpu::DepthBiasState {
            constant: 0,
            slope_scale: if mask { 0.0 } else { 1.5 },
            clamp: 0.0,
        },
    }
}

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
    /// shared.
    pub fn shared(shader_folder: PathBuf) -> Result<Arc<Gpu>> {
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
    pub fn create(shader_folder: PathBuf) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|error| Error::Gpu(format!("no GPU adapter: {error}")))?;
        // first_instance in indirect draws selects the per-draw instance
        // records; shadow maps have a stencil (SHADOW_FORMAT).
        let required =
            wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::DEPTH32FLOAT_STENCIL8;
        if !adapter.features().contains(required) {
            return Err(Error::Gpu(
                "GPU adapter lacks indirect-first-instance or depth32float-stencil8".into(),
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
        .map_err(|error| Error::Gpu(format!("no GPU device: {error}")))?;
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
    ) -> Result<wgpu::ShaderModule> {
        let mut code = self.load_wgsl(filename, &mut HashSet::new())?;
        for (old, new) in replacements {
            code = code.replace(old, new);
        }
        Ok(self.module(filename, &code))
    }

    /// `shader`, with errors in the code (e.g. a user's surface shader)
    /// returned instead of aborting the process.
    pub fn checked_shader(
        &self,
        filename: &str,
        replacements: &[(&str, &str)],
    ) -> Result<wgpu::ShaderModule> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.shader(filename, replacements)?;
        match pollster::block_on(scope.pop()) {
            None => Ok(module),
            Some(error) => Err(Error::Shader(format!("{filename}: {error}"))),
        }
    }

    pub fn module(&self, label: &str, code: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            })
    }

    /// A file's source with its `#include "file.wgsl"` lines
    /// expanded, each file at most once.
    fn load_wgsl(&self, filename: &str, included: &mut HashSet<String>) -> Result<String> {
        included.insert(filename.to_string());
        let path = self.shader_folder.join(filename);
        let code = std::fs::read_to_string(&path)
            .map_err(|error| Error::Gpu(format!("cannot read {}: {error}", path.display())))?;
        self.expand_includes(&code, included)
    }

    /// `expand_includes` for WGSL given as a string.
    pub fn expand_includes(&self, code: &str, included: &mut HashSet<String>) -> Result<String> {
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
