//! `ShaderEffect`: the user's own WGSL, run once per pixel on the lit
//! layer (`stage="lit"`, linear HDR, before compositing) or on the whole
//! frame (`stage="frame"`, after every view and sub-frame).
//!
//! The user writes one function, `effect(uv, pixel) -> vec4<f32>`; it is
//! put into `TEMPLATE`, which gives it the `source` texture, the frame's
//! uniforms (`frame.time` among them) and up to 16 numbers (`value(i)`).
//! A pipeline is made once per piece of code.

use std::collections::{HashMap, HashSet};

use crate::error::Result;
use crate::gpu::device::{Gpu, HDR_FORMAT};
use crate::gpu::fullscreen::{TexturePool, UniformSlots};

/// The settings of a shader effect.
pub struct ShaderEffect {
    pub code: String,
    /// Whether it runs on the frame ("frame"), else on the lit layer ("lit").
    pub frame_stage: bool,
    pub values: Vec<f32>,
}

const TEMPLATE: &str = r#"
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

/// The GPU side of shader effects: one pipeline per piece of code, and
/// the bindings they share.
pub struct ShaderEffects {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    values: UniformSlots,
    pipelines: HashMap<String, wgpu::RenderPipeline>,
    /// Counts the effects run, to alternate between two output textures.
    uses: u64,
}

impl ShaderEffects {
    pub fn new(device: &wgpu::Device) -> ShaderEffects {
        let both = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX;
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        ShaderEffects {
            layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            values: UniformSlots::new(device, "effect values"),
            pipelines: HashMap::new(),
            uses: 0,
        }
    }

    /// `effect` run on `source` at `stage` ("lit" or "frame"); returns the
    /// texture the result is in.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        textures: &mut TexturePool,
        effect: &ShaderEffect,
        source: &wgpu::Texture,
        stage: &'static str,
        frame_uniform: &wgpu::Buffer,
    ) -> Result<wgpu::Texture> {
        // Alternate between two outputs per stage so effects can be chained.
        self.uses += 1;
        let name: &'static str = match (stage, self.uses % 2) {
            ("lit", 0) => "lit effect 0",
            ("lit", _) => "lit effect 1",
            (_, 0) => "frame effect 0",
            _ => "frame effect 1",
        };
        let output = textures.get(
            &gpu.device,
            name,
            (source.width(), source.height()),
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::empty(),
        );
        let mut values = [0.0f32; 16];
        values[..effect.values.len()].copy_from_slice(&effect.values);
        if !self.pipelines.contains_key(&effect.code) {
            let pipeline = self.pipeline(gpu, &effect.code)?;
            self.pipelines.insert(effect.code.clone(), pipeline);
        }
        let pipeline = &self.pipelines[&effect.code];
        let source_view = source.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let values = self.values.write(&gpu.queue, bytemuck::cast_slice(&values));
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shader effect"),
            layout: &self.layout,
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
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
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

    /// The pipeline of the user's `code` in the template.
    fn pipeline(&self, gpu: &Gpu, code: &str) -> Result<wgpu::RenderPipeline> {
        let code = TEMPLATE.replace("{code}", code);
        let code = gpu.expand_includes(&code, &mut HashSet::new())?;
        let module = gpu.module("shader effect", &code);
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&self.layout)],
                immediate_size: 0,
            });
        Ok(gpu
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
            }))
    }
}
