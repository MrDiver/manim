//! Deferred lighting: G-buffer and lights in, the linear HDR "lit" layer
//! out: fullscreen triangles shade once per pixel where all MSAA samples
//! hold one surface, once per sample on edges, which a first triangle
//! marks in a stencil (shaders/deferred_lighting.wgsl). Lighting edges
//! per sample and the rest once is Andrew Lauritzen's,
//! [Deferred Rendering for Current and Future Rendering Pipelines](https://www.intel.com/content/www/us/en/developer/articles/technical/deferred-rendering-for-current-and-future-rendering-pipelines.html)
//! (SIGGRAPH 2010).

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::render::buffer::GrowableBuffer;
use crate::render::context::{Gpu, EDGE_FORMAT, HDR_FORMAT};
use crate::render::error::Result;
use crate::render::passes::shadow::ShadowPass;

/// The stencil value of a pixel on an edge.
const EDGE: u32 = 1;

pub struct LightingPass {
    lights: GrowableBuffer,
    scope_parents: GrowableBuffer,
    /// Per sample count: the layout and the pipelines (with MSAA: marking
    /// edges, per pixel, per sample on edges).
    variants: HashMap<u32, (wgpu::BindGroupLayout, Vec<wgpu::RenderPipeline>)>,
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
    ) -> Result<()> {
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

    /// Light into `pass`, whose stencil (with MSAA) is the edge stencil
    /// (`RenderTargets::edge_attachment`), cleared.
    pub fn draw(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass) -> Result<()> {
        let sample_count = self.sample_count;
        pass.set_bind_group(0, self.bind_group.as_ref(), &[]);
        pass.set_stencil_reference(EDGE);
        for pipeline in &self.variant(gpu, sample_count)?.1 {
            pass.set_pipeline(pipeline);
            pass.draw(0..3, 0..1); // one fullscreen triangle
        }
        Ok(())
    }

    /// Layout and pipelines for multisampled or single-sampled G-buffers: a
    /// multisampled binding cannot have one sample, so without MSAA the
    /// shader reads plain textures and shades every pixel once.
    fn variant(
        &mut self,
        gpu: &Gpu,
        sample_count: u32,
    ) -> Result<&(wgpu::BindGroupLayout, Vec<wgpu::RenderPipeline>)> {
        if let Entry::Vacant(slot) = self.variants.entry(sample_count) {
            let multisampled = sample_count > 1;
            let samples = format!("const GBUFFER_SAMPLES: i32 = {sample_count};");
            let mut replacements = vec![("const GBUFFER_SAMPLES: i32 = 4;", samples.as_str())];
            if !multisampled {
                replacements.push(("texture_depth_multisampled_2d", "texture_depth_2d"));
                replacements.push(("texture_multisampled_2d<f32>", "texture_2d<f32>"));
                replacements.push((
                    "const FIRST_SAMPLE: vec2<f32> = vec2<f32>(0.375, 0.125);",
                    "const FIRST_SAMPLE: vec2<f32> = vec2<f32>(0.5);",
                ));
            }
            let shader = gpu.shader("deferred_lighting.wgsl", &replacements)?;
            let layout = create_layout(&gpu.device, multisampled);
            let pipeline_layout =
                gpu.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            // `edges`: the stencil test (None: no stencil); `mark`: write
            // the edge value instead of color.
            let pipeline = |entry_point: &str, edges: Option<wgpu::CompareFunction>, mark: bool| {
                let write_mask = if mark {
                    wgpu::ColorWrites::empty()
                } else {
                    wgpu::ColorWrites::ALL
                };
                let depth_stencil = edges.map(|compare| {
                    let face = wgpu::StencilFaceState {
                        compare,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: if mark {
                            wgpu::StencilOperation::Replace
                        } else {
                            wgpu::StencilOperation::Keep
                        },
                    };
                    wgpu::DepthStencilState {
                        format: EDGE_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::Always),
                        stencil: wgpu::StencilState {
                            front: face,
                            back: face,
                            read_mask: 0xff,
                            write_mask: 0xff,
                        },
                        bias: Default::default(),
                    }
                });
                gpu.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(entry_point),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs_fullscreen"),
                            compilation_options: Default::default(),
                            buffers: &[],
                        },
                        primitive: Default::default(),
                        depth_stencil,
                        multisample: wgpu::MultisampleState {
                            count: sample_count,
                            ..Default::default()
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some(entry_point),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: HDR_FORMAT,
                                blend: None,
                                write_mask,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    })
            };
            let pipelines = if multisampled {
                use wgpu::CompareFunction::{Always, Equal, NotEqual};
                vec![
                    pipeline("fs_mark_edges", Some(Always), true),
                    pipeline("fs_light_pixel", Some(NotEqual), false),
                    pipeline("fs_light_sample", Some(Equal), false),
                ]
            } else {
                vec![pipeline("fs_light_pixel", None, false)]
            };
            slot.insert((layout, pipelines));
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
