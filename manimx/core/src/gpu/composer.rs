//! The composer: turns the renderer's layers into a frame. Ports the GPU
//! side of `FrameComposer` (manimx/renderer/webgpu/composer.py):
//!
//!     for each sub-frame (one, or the MotionBlur samples):
//!         for each view (one per active clip, lowest first):
//!             layers = render the view
//!             lit = lit-stage effects (Bloom, custom shaders) on layers.lit
//!             image = composite(tone_map(lit), layers.vector) over image
//!         accumulator += weight * image
//!     frame = frame-stage effects on the accumulator
//!     output = encode to sRGB
//!
//! Which sub-frames and views there are is decided in Python (they come
//! from the timeline); `compose` gets them ready to render.

use std::collections::{HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use pyo3::prelude::*;

use super::context::{Gpu, COLOR_FORMAT, HDR_FORMAT};
use super::profiler::GpuProfiler;

const UNIFORM_STRIDE: u64 = 256; // minimum uniform buffer offset alignment
const UNIFORM_SLOTS: u64 = 64;

/// `COMPOSITE_PARAMS_DTYPE` (composite.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParams {
    background: [f32; 4],
    tone_map: u32,
    weight: f32,
    _pad: [f32; 2],
}

/// `BLOOM_PARAMS_DTYPE` (bloom.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomParams {
    texel: [f32; 2],
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    _pad: [f32; 2],
}

/// `Bloom` (renderer/effects.py).
#[derive(Clone, Copy)]
pub struct Bloom {
    pub threshold: f32,
    pub knee: f32,
    pub intensity: f32,
    pub radius: f32,
    pub levels: u32,
}

/// `ShaderEffect` (renderer/effects.py).
pub struct ShaderEffect {
    pub code: String,
    /// "lit" or "frame".
    pub frame_stage: bool,
    pub values: Vec<f32>,
}

/// One effect of the chain, in order (`MotionBlur` is handled in Python:
/// it decides the sub-frames).
pub enum Effect {
    Bloom(Bloom),
    Shader(ShaderEffect),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Blend {
    Replace,
    /// Sums: sub-frames into the accumulator, bloom mips.
    Additive,
    /// Premultiplied source over destination: views over each other.
    Over,
}

impl Blend {
    fn state(self) -> Option<wgpu::BlendState> {
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

/// The texture to render the composited frame of `views` into, and how.
pub struct CompositeTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub weight: f32,
    pub clear: bool,
    pub background: bool,
    pub blend: Blend,
}

/// Named textures reused across frames, made again when their size
/// changes (`_TexturePool`).
#[derive(Default)]
pub struct TexturePool {
    textures: HashMap<&'static str, wgpu::Texture>,
}

impl TexturePool {
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

/// Fullscreen-triangle passes; pipelines are kept per shader, entry
/// point, target format and blend (`_Fullscreen`).
pub struct Fullscreen {
    pipelines: HashMap<
        (String, &'static str, wgpu::TextureFormat, Blend),
        (wgpu::RenderPipeline, wgpu::BindGroupLayout),
    >,
    pub linear_clamp: wgpu::Sampler,
}

impl Fullscreen {
    fn new(device: &wgpu::Device) -> Fullscreen {
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
        let key = (shader.to_string(), entry_point, format, blend);
        if !self.pipelines.contains_key(&key) {
            let module = gpu.shader(shader, &[])?;
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
struct UniformSlots {
    buffer: wgpu::Buffer,
    next: u64,
}

impl UniformSlots {
    fn new(device: &wgpu::Device, label: &str) -> UniformSlots {
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
    fn write(&mut self, queue: &wgpu::Queue, data: &[u8]) -> wgpu::BindingResource<'_> {
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

const SHADER_EFFECT_TEMPLATE: &str = r#"
#include "fullscreen.wgsl"
#include "frame.wgsl"
#include "color.wgsl"

struct EffectValues {
    values: array<vec4<f32>, 4>,
};

@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;
@group(0) @binding(3) var<uniform> effect_values: EffectValues;

fn value(index: u32) -> f32 {
    return effect_values.values[index / 4u][index % 4u];
}

{code}

@fragment
fn fs_effect(in: FullscreenVertex) -> @location(0) vec4<f32> {
    return effect(in.uv, vec2<i32>(floor(in.position.xy)));
}
"#;

pub struct Composer {
    pub effects: Vec<Effect>,
    pub tone_map: u32,
    pub textures: TexturePool,
    fullscreen: Fullscreen,
    composite_params: UniformSlots,
    bloom_params: UniformSlots,
    effect_values: UniformSlots,
    effect_layout: wgpu::BindGroupLayout,
    effect_sampler: wgpu::Sampler,
    effect_pipelines: HashMap<String, wgpu::RenderPipeline>,
    effect_uses: u64,
}

impl Composer {
    pub fn new(gpu: &Gpu, effects: Vec<Effect>, tone_map: u32) -> Composer {
        let device = &gpu.device;
        let both = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX;
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let effect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shader effect"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: both,
                    ty: uniform,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: both,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: both,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: both,
                    ty: uniform,
                    count: None,
                },
            ],
        });
        Composer {
            effects,
            tone_map,
            textures: TexturePool::default(),
            fullscreen: Fullscreen::new(device),
            composite_params: UniformSlots::new(device, "composite params"),
            bloom_params: UniformSlots::new(device, "bloom params"),
            effect_values: UniformSlots::new(device, "effect values"),
            effect_layout,
            effect_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            effect_pipelines: HashMap::new(),
            effect_uses: 0,
        }
    }

    pub fn has_shader_effects(&self) -> bool {
        self.effects
            .iter()
            .any(|effect| matches!(effect, Effect::Shader(_)))
    }

    /// The lit-stage effects (Bloom, lit shaders) on `lit`, in chain order.
    pub fn lit_effects(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        lit: &wgpu::Texture,
        frame_uniform: &wgpu::Buffer,
        profiler: &mut GpuProfiler,
    ) -> PyResult<wgpu::Texture> {
        let mut lit = lit.clone();
        for index in 0..self.effects.len() {
            lit = match &self.effects[index] {
                Effect::Bloom(bloom) => {
                    let bloom = *bloom;
                    self.bloom(gpu, encoder, &lit, &bloom, profiler)?
                }
                Effect::Shader(effect) if !effect.frame_stage => {
                    self.shader_effect(gpu, encoder, &lit, index, "lit", frame_uniform)?
                }
                Effect::Shader(_) => lit,
            };
        }
        Ok(lit)
    }

    /// The frame-stage shader effects on the accumulated frame.
    pub fn frame_effects(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::Texture,
        frame_uniform: &wgpu::Buffer,
    ) -> PyResult<wgpu::Texture> {
        let mut frame = frame.clone();
        for index in 0..self.effects.len() {
            if matches!(&self.effects[index], Effect::Shader(effect) if effect.frame_stage) {
                frame = self.shader_effect(gpu, encoder, &frame, index, "frame", frame_uniform)?;
            }
        }
        Ok(frame)
    }

    /// `fs_composite`: tone mapped lit layer and vector layer over the
    /// background, `weight` times into `target`.
    #[allow(clippy::too_many_arguments)]
    pub fn composite(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        lit: &wgpu::Texture,
        vector: &wgpu::Texture,
        target: CompositeTarget,
        background: [f32; 4],
        profiler: &mut GpuProfiler,
    ) -> PyResult<()> {
        let params = CompositeParams {
            background: if target.background {
                background
            } else {
                [0.0; 4]
            },
            tone_map: self.tone_map,
            weight: target.weight,
            _pad: [0.0; 2],
        };
        let lit_view = lit.create_view(&Default::default());
        let vector_view = vector.create_view(&Default::default());
        let params = self
            .composite_params
            .write(&gpu.queue, bytemuck::bytes_of(&params));
        let resources = [
            params,
            wgpu::BindingResource::TextureView(&lit_view),
            wgpu::BindingResource::TextureView(&vector_view),
        ];
        self.fullscreen.run(
            gpu,
            encoder,
            "composite.wgsl",
            "fs_composite",
            target.view,
            HDR_FORMAT,
            &resources,
            target.blend,
            target.clear,
            profiler,
            "composite",
        )
    }

    /// `fs_accumulate`: `weight` times a stack of views into the accumulator.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        stack: &wgpu::Texture,
        accumulator: &wgpu::TextureView,
        weight: f32,
        clear: bool,
        profiler: &mut GpuProfiler,
    ) -> PyResult<()> {
        let params = CompositeParams {
            background: [0.0; 4],
            tone_map: self.tone_map,
            weight,
            _pad: [0.0; 2],
        };
        let stack_view = stack.create_view(&Default::default());
        let params = self
            .composite_params
            .write(&gpu.queue, bytemuck::bytes_of(&params));
        let resources = [params, wgpu::BindingResource::TextureView(&stack_view)];
        self.fullscreen.run(
            gpu,
            encoder,
            "composite.wgsl",
            "fs_accumulate",
            accumulator,
            HDR_FORMAT,
            &resources,
            Blend::Additive,
            clear,
            profiler,
            "composite",
        )
    }

    /// `fs_finalize`: the accumulated linear frame encoded to sRGB, straight alpha.
    pub fn finalize(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::Texture,
        output: &wgpu::TextureView,
        profiler: &mut GpuProfiler,
    ) -> PyResult<()> {
        let frame_view = frame.create_view(&Default::default());
        let resources = [wgpu::BindingResource::TextureView(&frame_view)];
        self.fullscreen.run(
            gpu,
            encoder,
            "finalize.wgsl",
            "fs_finalize",
            output,
            COLOR_FORMAT,
            &resources,
            Blend::Replace,
            true,
            profiler,
            "finalize",
        )
    }

    /// `_BloomProcessor.apply` (see shaders/bloom.wgsl).
    fn bloom(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        lit: &wgpu::Texture,
        bloom: &Bloom,
        profiler: &mut GpuProfiler,
    ) -> PyResult<wgpu::Texture> {
        let (width, height) = (lit.width(), lit.height());
        // The chain starts at half resolution; stop before mips get tiny.
        let smallest = (width.min(height) / 4).max(1);
        let levels = bloom.levels.min(smallest.ilog2()).max(1);
        let device = &gpu.device;
        let chain = self.textures.get(
            device,
            "bloom chain",
            ((width / 2).max(1), (height / 2).max(1)),
            HDR_FORMAT,
            levels,
            wgpu::TextureUsages::empty(),
        );
        let output = self.textures.get(
            device,
            "bloom output",
            (width, height),
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::empty(),
        );
        let mips: Vec<wgpu::TextureView> = (0..levels)
            .map(|level| {
                chain.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let mip_size = |level: u32| {
            (
                (chain.width() >> level).max(1),
                (chain.height() >> level).max(1),
            )
        };
        let lit_view = lit.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());

        let passes = &mut self.fullscreen;
        let params = &mut self.bloom_params;
        let mut step = |encoder: &mut wgpu::CommandEncoder,
                        entry: &'static str,
                        source: &wgpu::TextureView,
                        size: (u32, u32),
                        target: &wgpu::TextureView,
                        blend: Blend,
                        clear: bool| {
            let values = BloomParams {
                texel: [1.0 / size.0 as f32, 1.0 / size.1 as f32],
                threshold: bloom.threshold,
                knee: bloom.knee,
                intensity: bloom.intensity,
                radius: bloom.radius,
                _pad: [0.0; 2],
            };
            let linear_clamp = passes.linear_clamp.clone();
            let mut resources = vec![
                params.write(&gpu.queue, bytemuck::bytes_of(&values)),
                wgpu::BindingResource::TextureView(source),
                wgpu::BindingResource::Sampler(&linear_clamp),
            ];
            if entry == "fs_combine" {
                resources.push(wgpu::BindingResource::TextureView(&mips[0]));
            }
            passes.run(
                gpu,
                encoder,
                "bloom.wgsl",
                entry,
                target,
                HDR_FORMAT,
                &resources,
                blend,
                clear,
                profiler,
                "bloom",
            )
        };
        step(
            encoder,
            "fs_prefilter",
            &lit_view,
            (width, height),
            &mips[0],
            Blend::Replace,
            true,
        )?;
        for level in 0..levels - 1 {
            step(
                encoder,
                "fs_downsample",
                &mips[level as usize],
                mip_size(level),
                &mips[level as usize + 1],
                Blend::Replace,
                true,
            )?;
        }
        for level in (1..levels).rev() {
            // Add the upsampled smaller mip onto the next larger one.
            step(
                encoder,
                "fs_upsample",
                &mips[level as usize],
                mip_size(level),
                &mips[level as usize - 1],
                Blend::Additive,
                false,
            )?;
        }
        step(
            encoder,
            "fs_combine",
            &lit_view,
            (width, height),
            &output_view,
            Blend::Replace,
            true,
        )?;
        Ok(output)
    }

    /// `_ShaderProcessor.apply`: the user's WGSL in the effect template.
    fn shader_effect(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        index: usize,
        stage: &'static str,
        frame_uniform: &wgpu::Buffer,
    ) -> PyResult<wgpu::Texture> {
        let Effect::Shader(effect) = &self.effects[index] else {
            unreachable!()
        };
        // Alternate between two outputs per stage so effects can be chained.
        self.effect_uses += 1;
        let name: &'static str = match (stage, self.effect_uses % 2) {
            ("lit", 0) => "lit effect 0",
            ("lit", _) => "lit effect 1",
            (_, 0) => "frame effect 0",
            _ => "frame effect 1",
        };
        let output = self.textures.get(
            &gpu.device,
            name,
            (source.width(), source.height()),
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::empty(),
        );
        let mut values = [0.0f32; 16];
        values[..effect.values.len()].copy_from_slice(&effect.values);
        if !self.effect_pipelines.contains_key(&effect.code) {
            let code = SHADER_EFFECT_TEMPLATE.replace("{code}", &effect.code);
            let code = gpu.expand_includes(&code, &mut HashSet::new())?;
            let module = gpu.module("shader effect", &code);
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&self.effect_layout)],
                    immediate_size: 0,
                });
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("shader effect"),
                    layout: Some(&layout),
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
                        entry_point: Some("fs_effect"),
                        compilation_options: Default::default(),
                        targets: &[Some(HDR_FORMAT.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            self.effect_pipelines.insert(effect.code.clone(), pipeline);
        }
        let pipeline = &self.effect_pipelines[&effect.code];
        let source_view = source.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let values = self
            .effect_values
            .write(&gpu.queue, bytemuck::cast_slice(&values));
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shader effect"),
            layout: &self.effect_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.effect_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: values,
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shader effect"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        Ok(output)
    }
}
