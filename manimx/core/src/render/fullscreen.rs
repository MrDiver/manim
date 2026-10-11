//! Fullscreen passes: one triangle covering the target, a fragment
//! shader per pixel. Composite, effects and depth of field are all made of
//! them. Pipelines are made once per shader and kept.
//!
//! Also here: `TexturePool`, the named textures these passes render into,
//! and `UniformSlots`, the small uniform buffers they read their settings
//! from.

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::render::context::Gpu;
use crate::render::profiler::GpuProfiler;

const UNIFORM_STRIDE: u64 = 256; // minimum uniform buffer offset alignment
const UNIFORM_SLOTS: u64 = 64;

/// How a pass's output combines with what the target holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Blend {
    Replace,
    /// Sums: sub-frames into the accumulator, bloom mips.
    Additive,
    /// Premultiplied source over destination: views over each other.
    Over,
}

impl Blend {
    pub fn state(self) -> Option<wgpu::BlendState> {
        let add = |dst| wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: dst,
            operation: wgpu::BlendOperation::Add,
        };
        match self {
            Blend::Replace => None,
            Blend::Additive => Some(wgpu::BlendState {
                color: add(wgpu::BlendFactor::One),
                alpha: add(wgpu::BlendFactor::One),
            }),
            Blend::Over => Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        }
    }
}

/// Named textures reused across frames, made again when their size
/// changes.
#[derive(Default)]
pub struct TexturePool {
    textures: HashMap<&'static str, wgpu::Texture>,
}

impl TexturePool {
    /// The texture named `name`, if one was made.
    pub fn find(&self, name: &str) -> Option<&wgpu::Texture> {
        self.textures.get(name)
    }

    pub fn get(
        &mut self,
        device: &wgpu::Device,
        name: &'static str,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        mip_levels: u32,
        extra: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        let texture = self.textures.get(name);
        if texture.is_none_or(|t| {
            (t.width(), t.height(), t.mip_level_count()) != (size.0, size.1, mip_levels)
        }) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(name),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | extra,
                view_formats: &[],
            });
            self.textures.insert(name, texture);
        }
        self.textures[name].clone()
    }
}

/// Which fullscreen shader to run.
pub struct FullscreenPass<'a> {
    pub shader: &'a str,
    pub entry_point: &'static str,
    /// For a pass that reads the frame's depth buffers: their samples per
    /// pixel (the shader is written for 4, multisampled; see
    /// view_distance.wgsl).
    pub depth_samples: Option<u32>,
}

/// Shader, entry point, target format, blend and depth samples of a pipeline.
type PipelineKey = (
    String,
    &'static str,
    wgpu::TextureFormat,
    Blend,
    Option<u32>,
);

/// Fullscreen-triangle passes; pipelines are kept per shader, entry
/// point, target format and blend.
pub struct Fullscreen {
    pipelines: HashMap<PipelineKey, (wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    pub linear_clamp: wgpu::Sampler,
}

impl Fullscreen {
    pub fn new(device: &wgpu::Device) -> Fullscreen {
        Fullscreen {
            pipelines: HashMap::new(),
            linear_clamp: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
        }
    }

    /// Draw one fullscreen pass into `target`; `resources` are bindings 0, 1, 2, ...
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        shader: &str,
        entry_point: &'static str,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        resources: &[wgpu::BindingResource],
        blend: Blend,
        clear: bool,
        profiler: &mut GpuProfiler,
        timing: &'static str,
    ) -> PyResult<()> {
        let pass = FullscreenPass {
            shader,
            entry_point,
            depth_samples: None,
        };
        self.run_pass(
            gpu, encoder, pass, target, format, resources, blend, clear, profiler, timing,
        )
    }

    /// `run` for a variant of a shader (see `FullscreenPass`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_pass(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        pass: FullscreenPass,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        resources: &[wgpu::BindingResource],
        blend: Blend,
        clear: bool,
        profiler: &mut GpuProfiler,
        timing: &'static str,
    ) -> PyResult<()> {
        let FullscreenPass {
            shader,
            entry_point,
            depth_samples,
        } = pass;
        let key = (
            shader.to_string(),
            entry_point,
            format,
            blend,
            depth_samples,
        );
        if !self.pipelines.contains_key(&key) {
            let count = format!("const DEPTH_SAMPLES: i32 = {};", depth_samples.unwrap_or(4));
            let mut replacements = vec![("const DEPTH_SAMPLES: i32 = 4;", count.as_str())];
            if depth_samples == Some(1) {
                replacements.push(("texture_depth_multisampled_2d", "texture_depth_2d"));
            }
            let module = gpu.shader(shader, &replacements)?;
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(entry_point),
                    layout: None, // derived from the shader, like wgpu-py's "auto"
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_fullscreen"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(entry_point),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: blend.state(),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            let layout = pipeline.get_bind_group_layout(0);
            self.pipelines.insert(key.clone(), (pipeline, layout));
        }
        let (pipeline, layout) = &self.pipelines[&key];
        let entries: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: resource.clone(),
            })
            .collect();
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry_point),
            layout,
            entries: &entries,
        });
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(entry_point),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: profiler.pass_timing(timing),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

/// A uniform buffer of 64 slots of 256 bytes, used in turn.
pub struct UniformSlots {
    buffer: wgpu::Buffer,
    next: u64,
}

impl UniformSlots {
    pub fn new(device: &wgpu::Device, label: &str) -> UniformSlots {
        UniformSlots {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: UNIFORM_STRIDE * UNIFORM_SLOTS,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            next: 0,
        }
    }

    /// Write `data` into the next slot; returns its binding.
    pub fn write(&mut self, queue: &wgpu::Queue, data: &[u8]) -> wgpu::BindingResource<'_> {
        self.next = (self.next + 1) % UNIFORM_SLOTS;
        let offset = self.next * UNIFORM_STRIDE;
        queue.write_buffer(&self.buffer, offset, data);
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset,
            size: wgpu::BufferSize::new(data.len() as u64),
        })
    }
}
