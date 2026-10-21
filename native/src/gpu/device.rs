//! The GPU device and shader loading: the shaders compiled into the
//! library (shaders.rs), their `#include "file.wgsl"` lines expanded,
//! each file at most once per shader.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::gpu::shaders;

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
/// Color targets of the G-buffer (see gbuffer.wgsl).
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

/// The formats rendered into with several samples per pixel (see
/// render/draw/targets.rs): a sample count must suit all of them.
const MULTISAMPLED_FORMATS: [wgpu::TextureFormat; 8] = [
    GBUFFER_FORMATS[0],
    GBUFFER_FORMATS[1],
    GBUFFER_FORMATS[2],
    GBUFFER_FORMATS[3],
    DEPTH_FORMAT,
    HDR_FORMAT,
    COLOR_FORMAT,
    EDGE_FORMAT,
];

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// The samples per pixel this device can render with, ascending: 1
    /// and 4 everywhere (WebGPU guarantees them), 2, 8 and 16 where the
    /// GPU has them for every multisampled format.
    pub sample_counts: Vec<u32>,
}

/// How the device's allocator reserves GPU memory
/// (`RenderConfig.gpu_memory`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    /// Blocks of 8 to 64 MiB: little memory held beyond what is used, so
    /// many processes (parallel tests, render workers) share one GPU.
    Compact,
    /// Blocks of 128 to 256 MiB (wgpu's default): fewer allocations.
    Fast,
}

impl Memory {
    fn hints(self) -> wgpu::MemoryHints {
        match self {
            Memory::Compact => wgpu::MemoryHints::MemoryUsage,
            Memory::Fast => wgpu::MemoryHints::Performance,
        }
    }
}

/// The process's devices, made on first use, one per kind of GPU and
/// memory use asked for (see `Gpu::shared`).
type SharedGpus = Vec<((wgpu::PowerPreference, Memory), Arc<Gpu>)>;
static SHARED: Mutex<SharedGpus> = Mutex::new(Vec::new());

impl Gpu {
    /// The device every renderer of this process that asks for this kind
    /// of GPU and memory use uses: drivers refuse more than a few devices
    /// at once, and pipelines and textures are cheaper shared.
    /// `preference` chooses the GPU: `HighPerformance` a dedicated one,
    /// `LowPower` an integrated one (when the dedicated one is busy, or to
    /// save power); whichever there is when there is only one.
    pub fn shared(preference: wgpu::PowerPreference, memory: Memory) -> Result<Arc<Gpu>> {
        let mut shared = SHARED.lock().expect("not poisoned");
        let kind = (preference, memory);
        if let Some((_, gpu)) = shared.iter().find(|(other, _)| *other == kind) {
            return Ok(gpu.clone());
        }
        let gpu = Arc::new(Gpu::create(preference, memory)?);
        shared.push((kind, gpu.clone()));
        Ok(gpu)
    }

    /// A device of the GPU `preference` chooses, with
    /// `indirect-first-instance` (draws select their instance records with
    /// it).
    pub fn create(preference: wgpu::PowerPreference, memory: Memory) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: preference,
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
        let mut features = required;
        // Sample counts beyond WebGPU's 1 and 4 (where the GPU has them)
        // need the adapter's own format features.
        let adapter_formats = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let sample_counts = if adapter.features().contains(adapter_formats) {
            features |= adapter_formats;
            [1, 2, 4, 8, 16]
                .into_iter()
                .filter(|&count| {
                    MULTISAMPLED_FORMATS.iter().all(|&format| {
                        adapter
                            .get_texture_format_features(format)
                            .flags
                            .sample_count_supported(count)
                    })
                })
                .collect()
        } else {
            vec![1, 4]
        };
        // The allocator keeps the blocks it reserved after the renderer
        // that needed them is gone, so their size decides what a process
        // holds (Memory).
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("manim"),
            required_features: features,
            required_limits: adapter.limits(),
            memory_hints: memory.hints(),
            ..Default::default()
        }))
        .map_err(|error| {
            // The usual cause is a GPU whose memory other processes hold.
            let hint = match memory {
                Memory::Fast => {
                    "; if the GPU is out of memory, try RenderConfig(gpu_memory=\"compact\")"
                }
                Memory::Compact => {
                    "; if the GPU is out of memory, free it or try RenderConfig(gpu=\"low-power\")"
                }
            };
            Error::Gpu(format!("no GPU device: {error}{hint}"))
        })?;
        Ok(Gpu {
            device,
            queue,
            sample_counts,
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
        self.expand_includes(&shaders::source(filename)?, included)
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
