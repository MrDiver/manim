//! GPU side of the vector pipeline: buffers, pipelines and draw encoding.
//!
//! Per frame, `prepare` uploads the records into persistent storage
//! buffers and one indirect-draw buffer laid out as
//!
//! ```text
//! [opaque draws][transparent draws][shadow draws per scope][mask 0 draws]...
//! ```
//!
//! Every list is then issued with a single `multi_draw_indirect`.

use std::collections::HashMap;

use crate::render::draw::passes::camera_views::CameraViewTexture;
use crate::render::draw::passes::images::ImageTexture;
use crate::render::error::Result;
use crate::render::gpu::buffer::{GrowableBuffer, Records};
use crate::render::gpu::device::{
    shadow_depth_stencil, Gpu, ShadowStencil, COLOR_FORMAT, DEPTH_FORMAT,
};
use crate::render::pack::flatten::ShadowList;
use crate::render::pack::records::Frame;

/// Bytes of one indirect draw (`DrawIndirectArgs`).
const DRAW_SIZE: u64 = 16;

/// The records of one frame's vector buffers and draw lists.
pub struct VectorData<'a> {
    /// objects, instances, fill vertices, fill colors, stroke points, stroke curves.
    pub buffers: [Records<'a>; 6],
    pub opaque: &'a [u8],
    pub transparent: &'a [u8],
    pub masks: &'a [Vec<u8>],
    /// Per light scope (NO_CLIP first), the shadow casters' draws sorted
    /// by clip scope, and their runs.
    pub shadows: &'a [ShadowList<u8>],
    /// The images fills show, by layer.
    pub images: &'a [std::sync::Arc<crate::robjects::Texture>],
}

/// What a pipeline variant is used for (`Role`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Role {
    /// Frame pass, no blending; overlap is resolved by the depth test.
    Opaque,
    /// Frame pass with blending, drawn back to front after the opaque list.
    Transparent,
    /// Clip mask pass: blending, no depth buffer.
    Mask,
    /// Shadow map pass: depth only, seen from a light.
    Shadow(ShadowStencil),
}

/// "Over" blending of premultiplied colors; in mask passes the alpha
/// channel accumulates the union of all mask shapes.
const ALPHA_BLEND: wgpu::BlendState = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;

pub struct VectorPass {
    buffers: Vec<GrowableBuffer>,
    indirect: GrowableBuffer,
    /// Shader and layout depend on whether the meshes' depth is multisampled.
    multisampled: Option<bool>,
    layout: Option<wgpu::BindGroupLayout>,
    shader: Option<wgpu::ShaderModule>,
    pipelines: HashMap<(Role, u32), wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    /// The render targets the bind group was made with.
    targets_generation: u64,
    opaque: (u64, u32),
    transparent: (u64, u32),
    masks: Vec<(u64, u32)>,
    /// (first, count) of each (light scope, clip scope) run of casters.
    shadows: HashMap<(i32, i32), (u64, u32)>,
    shadow_groups: HashMap<u64, wgpu::BindGroup>,
    /// The pictures of the frame's camera views.
    pub camera_views: CameraViewTexture,
    /// The camera view texture the bind group was made with.
    camera_views_generation: u64,
    /// The images fills show.
    images: ImageTexture,
    /// The image texture the bind group was made with.
    images_generation: u64,
}

impl VectorPass {
    pub fn new(gpu: &Gpu) -> VectorPass {
        let storage = wgpu::BufferUsages::STORAGE;
        let names = [
            "vector objects",
            "vector instances",
            "fill vertices",
            "fill colors",
            "stroke points",
            "stroke curves",
        ];
        VectorPass {
            buffers: names
                .iter()
                .map(|name| GrowableBuffer::new(&gpu.device, name, storage))
                .collect(),
            indirect: GrowableBuffer::new(
                &gpu.device,
                "vector draws",
                wgpu::BufferUsages::INDIRECT,
            ),
            multisampled: None,
            layout: None,
            shader: None,
            pipelines: HashMap::new(),
            bind_group: None,
            targets_generation: u64::MAX,
            opaque: (0, 0),
            transparent: (0, 0),
            masks: Vec::new(),
            shadows: HashMap::new(),
            shadow_groups: HashMap::new(),
            camera_views: CameraViewTexture::new(&gpu.device),
            camera_views_generation: u64::MAX,
            images: ImageTexture::new(&gpu.device),
            images_generation: u64::MAX,
        }
    }

    /// Upload the frame's records. `mask_layers` holds one layer per clip
    /// scope; `mesh_depth` is the meshes' depth that hides shapes behind
    /// meshes; `targets_generation` changes when they are new textures.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        data: &VectorData,
        frame_uniform: &wgpu::Buffer,
        mask_layers: &wgpu::TextureView,
        mesh_depth: &wgpu::TextureView,
        multisampled: bool,
        targets_generation: u64,
    ) -> Result<()> {
        let changed = self.use_variant(gpu, multisampled)?;
        self.images.ensure(&gpu.device, &gpu.queue, data.images);
        let mut grown = false;
        for (buffer, records) in self.buffers.iter_mut().zip(&data.buffers) {
            grown |= buffer.write_records(&gpu.device, &gpu.queue, records);
        }

        let mut lists: Vec<&[u8]> = vec![data.opaque, data.transparent];
        lists.extend(data.shadows.iter().map(|(_, draws, _)| draws.as_slice()));
        lists.extend(data.masks.iter().map(|draws| draws.as_slice()));
        self.indirect
            .upload(&gpu.device, &gpu.queue, &lists.concat());
        let mut ranges = Vec::with_capacity(lists.len());
        let mut first = 0u64;
        for list in &lists {
            let count = list.len() as u64 / DRAW_SIZE;
            ranges.push((first, count as u32));
            first += count;
        }
        self.opaque = ranges[0];
        self.transparent = ranges[1];
        let shadow_count = data.shadows.len();
        self.shadows = shadow_runs(
            data.shadows
                .iter()
                .zip(&ranges[2..2 + shadow_count])
                .map(|((scope, _, runs), range)| (*scope, *range, runs.as_slice())),
        );
        self.masks = ranges[2 + shadow_count..].to_vec();

        if grown
            || changed
            || self.bind_group.is_none()
            || targets_generation != self.targets_generation
            || self.camera_views.generation != self.camera_views_generation
            || self.images.generation != self.images_generation
        {
            self.targets_generation = targets_generation;
            self.camera_views_generation = self.camera_views.generation;
            self.images_generation = self.images.generation;
            self.bind_group =
                Some(self.create_bind_group(gpu, frame_uniform, 0, mask_layers, mesh_depth));
            self.shadow_groups.clear();
        }
        Ok(())
    }

    /// Draw the mask shapes of clip scope `scope`.
    pub fn draw_mask(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass,
        sample_count: u32,
        scope: usize,
    ) {
        pass.set_bind_group(0, self.bind_group.as_ref(), &[]);
        let draws = self.masks[scope];
        self.draw(gpu, pass, Role::Mask, sample_count, draws);
    }

    /// Draw all opaque, then all transparent shapes into the frame.
    pub fn draw_frame(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass, sample_count: u32) {
        pass.set_bind_group(0, self.bind_group.as_ref(), &[]);
        let (opaque, transparent) = (self.opaque, self.transparent);
        self.draw(gpu, pass, Role::Opaque, sample_count, opaque);
        self.draw(gpu, pass, Role::Transparent, sample_count, transparent);
    }

    /// The clip scopes of light scope `scope`'s shadow casters.
    pub fn shadow_clips(&self, scope: i32) -> Vec<i32> {
        self.shadows
            .keys()
            .filter(|(light_scope, _)| *light_scope == scope)
            .map(|(_, clip)| *clip)
            .collect()
    }

    /// Draw the depth of light scope `scope`'s casters in clip scope
    /// `clip`, seen from the frame at `offset` of `frames` (`clipped`: where
    /// the stencil equals the reference).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_shadow(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass,
        frames: &wgpu::Buffer,
        offset: u64,
        (scope, clip, clipped): (i32, i32, bool),
        mask_layers: &wgpu::TextureView,
        mesh_depth: &wgpu::TextureView,
    ) {
        self.use_shadow_frame(gpu, pass, frames, offset, mask_layers, mesh_depth);
        let stencil = if clipped {
            ShadowStencil::Clipped
        } else {
            ShadowStencil::None
        };
        let draws = self.shadows.get(&(scope, clip)).copied().unwrap_or((0, 0));
        self.draw(gpu, pass, Role::Shadow(stencil), 1, draws);
    }

    /// Draw clip scope `clip`'s mask into the stencil, seen from the frame
    /// at `offset`: the reference where it covers, or (`inner`) one more
    /// where it covers the reference.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_shadow_mask(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass,
        frames: &wgpu::Buffer,
        offset: u64,
        (clip, inner): (i32, bool),
        mask_layers: &wgpu::TextureView,
        mesh_depth: &wgpu::TextureView,
    ) {
        self.use_shadow_frame(gpu, pass, frames, offset, mask_layers, mesh_depth);
        let stencil = if inner {
            ShadowStencil::InnerMask
        } else {
            ShadowStencil::Mask
        };
        let draws = self.masks[clip as usize];
        self.draw(gpu, pass, Role::Shadow(stencil), 1, draws);
    }

    fn use_shadow_frame(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass,
        frames: &wgpu::Buffer,
        offset: u64,
        mask_layers: &wgpu::TextureView,
        mesh_depth: &wgpu::TextureView,
    ) {
        if !self.shadow_groups.contains_key(&offset) {
            let group = self.create_bind_group(gpu, frames, offset, mask_layers, mesh_depth);
            self.shadow_groups.insert(offset, group);
        }
        pass.set_bind_group(0, self.shadow_groups.get(&offset), &[]);
    }

    /// The shadow bind groups refer to the shadow frames buffer: forget
    /// them when it grows.
    pub fn forget_shadow_groups(&mut self) {
        self.shadow_groups.clear();
    }

    fn draw(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass,
        role: Role,
        sample_count: u32,
        (first, count): (u64, u32),
    ) {
        if count == 0 {
            return;
        }
        let pipeline = self.pipeline(gpu, role, sample_count);
        pass.set_pipeline(pipeline);
        pass.multi_draw_indirect(&self.indirect.buffer, first * DRAW_SIZE, count);
    }

    // -- setup ------------------------------------------------------------

    fn pipeline(&mut self, gpu: &Gpu, role: Role, sample_count: u32) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&(role, sample_count)) {
            let pipeline = self.create_pipeline(gpu, role, sample_count);
            self.pipelines.insert((role, sample_count), pipeline);
        }
        &self.pipelines[&(role, sample_count)]
    }

    fn create_pipeline(&self, gpu: &Gpu, role: Role, sample_count: u32) -> wgpu::RenderPipeline {
        let shader = self.shader.as_ref().unwrap();
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(self.layout.as_ref().unwrap())],
                immediate_size: 0,
            });
        let (depth_stencil, entry_point, targets) = match role {
            Role::Shadow(stencil) => (Some(shadow_depth_stencil(stencil)), "fs_shadow", vec![]),
            Role::Mask => (None, "fs_mask", vec![Some(color_target(true))]),
            // Transparent draws also write depth: with the strict "less"
            // test an object never blends over itself (where its stroke
            // ribbon overlaps a joint), only over earlier objects.
            Role::Opaque | Role::Transparent => (
                Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                "fs_main",
                vec![Some(color_target(role == Role::Transparent))],
            ),
        };
        gpu.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vector"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                // Fills and strokes are flat and visible from both sides.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// Build shader, layout and pipelines for the mesh depth's sample
    /// count; returns whether they changed.
    fn use_variant(&mut self, gpu: &Gpu, multisampled: bool) -> Result<bool> {
        if self.multisampled == Some(multisampled) {
            return Ok(false);
        }
        let replacements: &[(&str, &str)] = if multisampled {
            &[]
        } else {
            &[("texture_depth_multisampled_2d", "texture_depth_2d")]
        };
        self.multisampled = Some(multisampled);
        self.shader = Some(gpu.shader("vector.wgsl", replacements)?);
        self.layout = Some(self.create_bind_group_layout(gpu, multisampled));
        self.pipelines.clear();
        Ok(true)
    }

    fn create_bind_group_layout(&self, gpu: &Gpu, multisampled: bool) -> wgpu::BindGroupLayout {
        let vertex = wgpu::ShaderStages::VERTEX;
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            // The fragment shader reads the layer depth step from the frame.
            visibility: vertex | fragment,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for binding in 1..=self.buffers.len() as u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vertex,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let next = 1 + self.buffers.len() as u32;
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next,
            visibility: fragment,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next + 1,
            visibility: fragment,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next + 2,
            visibility: fragment,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next + 3,
            visibility: fragment,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next + 4,
            visibility: fragment,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next + 5,
            visibility: fragment,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next + 6,
            visibility: fragment,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        gpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vector"),
                entries: &entries,
            })
    }

    fn create_bind_group(
        &self,
        gpu: &Gpu,
        frames: &wgpu::Buffer,
        offset: u64,
        mask_layers: &wgpu::TextureView,
        mesh_depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        // Binding order matches vector.wgsl: frame, storage buffers, masks,
        // mesh depth, camera views and their sampler, images, their sampler
        // and regions.
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: frames,
                offset,
                size: wgpu::BufferSize::new(size_of::<Frame>() as u64),
            }),
        }];
        for (index, buffer) in self.buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + index as u32,
                resource: buffer.binding(),
            });
        }
        let next = 1 + self.buffers.len() as u32;
        entries.push(wgpu::BindGroupEntry {
            binding: next,
            resource: wgpu::BindingResource::TextureView(mask_layers),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 1,
            resource: wgpu::BindingResource::TextureView(mesh_depth),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 2,
            resource: wgpu::BindingResource::TextureView(&self.camera_views.array_view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 3,
            resource: wgpu::BindingResource::Sampler(&self.camera_views.sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 4,
            resource: wgpu::BindingResource::TextureView(&self.images.array_view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 5,
            resource: wgpu::BindingResource::Sampler(&self.images.sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 6,
            resource: self.images.regions.as_entire_binding(),
        });
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector"),
            layout: self.layout.as_ref().unwrap(),
            entries: &entries,
        })
    }
}

fn color_target(blend: bool) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: COLOR_FORMAT,
        blend: blend.then_some(ALPHA_BLEND),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// (first, count) per (light scope, clip scope) of shadow casters: each
/// light scope's range split into its runs of equal clip scopes
/// (`shadow_runs`).
pub fn shadow_runs<'a>(
    lists: impl Iterator<Item = (i32, (u64, u32), &'a [(i32, u32)])>,
) -> HashMap<(i32, i32), (u64, u32)> {
    let mut runs = HashMap::new();
    for (scope, (mut first, _), clips) in lists {
        for &(clip, count) in clips {
            runs.insert((scope, clip), (first, count));
            first += count as u64;
        }
    }
    runs
}
