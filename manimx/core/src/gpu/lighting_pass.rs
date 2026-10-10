//! Deferred lighting: G-buffer and lights in, the linear HDR "lit" layer
//! out, one fullscreen triangle over every MSAA sample. Ports
//! `LightingPass` (manimx/renderer/webgpu/lighting_pass.py).

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use super::buffer::GrowableBuffer;
use super::context::{Gpu, HDR_FORMAT};
use super::shadow_pass::ShadowPass;

pub struct LightingPass {
    lights: GrowableBuffer,
    scope_parents: GrowableBuffer,
    variants: HashMap<u32, (wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    bind_group: Option<wgpu::BindGroup>,
    sample_count: u32,
    targets_generation: u64,
}

impl LightingPass {
    pub fn new(gpu: &Gpu) -> LightingPass {
        let storage = wgpu::BufferUsages::STORAGE;
        LightingPass {
            lights: GrowableBuffer::new(&gpu.device, "lights", storage),
            scope_parents: GrowableBuffer::new(&gpu.device, "scope parents", storage),
            variants: HashMap::new(),
            bind_group: None,
            sample_count: 0,
            targets_generation: u64::MAX,
        }
    }

    /// Upload lights and scope parents; `gbuffer` is (albedo, normal,
    /// emissive, scope) and `depth` the G-buffer's depth.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        lights: &[u8],
        scope_parents: &[i32],
        frame_uniform: &wgpu::Buffer,
        gbuffer: &[wgpu::TextureView],
        depth: &wgpu::TextureView,
        sample_count: u32,
        shadows: &ShadowPass,
        shadows_changed: bool,
        targets_generation: u64,
    ) -> pyo3::PyResult<()> {
        let mut grown = self.lights.upload(&gpu.device, &gpu.queue, lights);
        let parents: &[i32] = if scope_parents.is_empty() {
            &[-1]
        } else {
            scope_parents
        };
        grown |= self
            .scope_parents
            .upload(&gpu.device, &gpu.queue, bytemuck::cast_slice(parents));
        if grown
            || shadows_changed
            || self.bind_group.is_none()
            || targets_generation != self.targets_generation
            || sample_count != self.sample_count
        {
            self.targets_generation = targets_generation;
            self.sample_count = sample_count;
            let layout = self.variant(gpu, sample_count)?.0.clone();
            let view = |view| wgpu::BindingResource::TextureView(view);
            let entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lights.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: view(&gbuffer[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: view(&gbuffer[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: view(&gbuffer[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: view(&gbuffer[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: view(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.scope_parents.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: view(&shadows.maps_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&shadows.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: shadows.views.binding(),
                },
            ];
            self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting"),
                layout: &layout,
                entries: &entries,
            }));
        }
        Ok(())
    }

    pub fn draw(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass) -> pyo3::PyResult<()> {
        let sample_count = self.sample_count;
        pass.set_pipeline(&self.variant(gpu, sample_count)?.1);
        pass.set_bind_group(0, self.bind_group.as_ref(), &[]);
        pass.draw(0..3, 0..1); // one fullscreen triangle
        Ok(())
    }

    /// Layout and pipeline for multisampled or single-sampled G-buffers: a
    /// multisampled binding cannot have one sample, so without MSAA the
    /// shader reads plain textures.
    fn variant(
        &mut self,
        gpu: &Gpu,
        sample_count: u32,
    ) -> pyo3::PyResult<&(wgpu::BindGroupLayout, wgpu::RenderPipeline)> {
        if let Entry::Vacant(slot) = self.variants.entry(sample_count) {
            let multisampled = sample_count > 1;
            let replacements: &[(&str, &str)] = if multisampled {
                &[]
            } else {
                &[
                    ("texture_depth_multisampled_2d", "texture_depth_2d"),
                    ("texture_multisampled_2d<f32>", "texture_2d<f32>"),
                ]
            };
            let shader = gpu.shader("deferred_lighting.wgsl", replacements)?;
            let layout = create_layout(&gpu.device, multisampled);
            let pipeline_layout =
                gpu.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("deferred lighting"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_fullscreen"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_light"),
                        compilation_options: Default::default(),
                        targets: &[Some(HDR_FORMAT.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            slot.insert((layout, pipeline));
        }
        Ok(&self.variants[&sample_count])
    }
}

fn create_layout(device: &wgpu::Device, multisampled: bool) -> wgpu::BindGroupLayout {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty,
        count: None,
    };
    let buffer = |ty| wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let storage = buffer(wgpu::BufferBindingType::Storage { read_only: true });
    let gbuffer = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting"),
        entries: &[
            entry(0, buffer(wgpu::BufferBindingType::Uniform)),
            entry(1, storage),
            entry(2, gbuffer),
            entry(3, gbuffer),
            entry(4, gbuffer),
            entry(
                5,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled,
                },
            ),
            entry(
                6,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
            ),
            entry(
                7,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            ),
            entry(8, storage),
            entry(9, gbuffer),
            entry(10, storage),
        ],
    })
}
