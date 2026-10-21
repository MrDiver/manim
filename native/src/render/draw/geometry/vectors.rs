//! `VectorGeometry`: the GPU side of 2D shapes and 3D curves: buffers,
//! pipelines and draws.
//!
//! Per frame, `prepare` uploads the records of each page (pack/vectors.rs)
//! into that page's persistent storage buffers, bound by the page's own
//! bind group, and every draw list into one indirect-draw buffer laid out as
//!
//! ```text
//! [opaque draws][transparent draws][shadow draws per scope][mask 0 draws]...
//! ```
//!
//! Each list is issued as one `multi_draw_indirect` per run of draws on
//! one page (`Run`), switching the bind group between runs: with one page
//! that is one call per list, as before pages existed.

use std::collections::HashMap;

use crate::error::Result;
use crate::gpu::buffer::GrowableBuffer;
use crate::gpu::device::{shadow_depth_stencil, Gpu, ShadowStencil, COLOR_FORMAT, DEPTH_FORMAT};
use crate::render::draw::geometry::images::ImageTexture;
use crate::render::draw::{write_records, Pictures};
use crate::render::packet::records::Frame;
use crate::render::packet::{DrawList, Records, ShadowDraws};

/// Bytes of one indirect draw (`DrawIndirectArgs`).
const DRAW_SIZE: u64 = 16;
/// Storage buffers per page (pack/vectors.rs `Page::arenas`).
const BUFFERS: u32 = 6;

/// The records of one frame's vector buffers and draw lists.
pub struct VectorData<'a> {
    /// Per page: objects, instances, fill vertices, fill colors, stroke
    /// points, stroke curves.
    pub pages: Vec<[Records<'a>; 6]>,
    pub opaque: &'a DrawList,
    pub transparent: &'a DrawList,
    pub masks: &'a [DrawList],
    /// Per light scope (NO_CLIP first), its shadow casters.
    pub shadows: &'a [ShadowDraws],
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

/// Consecutive draws of a list on one page: `count` draws from `first`.
#[derive(Clone, Copy)]
struct Run {
    first: u64,
    count: u32,
    page: usize,
}

/// A shadow map frame to draw from: the shadow frames buffer and this
/// frame's offset in it, and the textures the bind group binds.
struct ShadowFrame<'a> {
    frames: &'a wgpu::Buffer,
    offset: u64,
    mask_layers: &'a wgpu::TextureView,
    mesh_depth: &'a wgpu::TextureView,
}

/// One page's buffers and the bind groups that bind them.
struct PageBuffers {
    buffers: Vec<GrowableBuffer>,
    bind_group: Option<wgpu::BindGroup>,
    /// Per shadow frame offset.
    shadow_groups: HashMap<u64, wgpu::BindGroup>,
}

impl PageBuffers {
    fn new(gpu: &Gpu, limit: u64) -> PageBuffers {
        let names = [
            "vector objects",
            "vector instances",
            "fill vertices",
            "fill colors",
            "stroke points",
            "stroke curves",
        ];
        let storage = wgpu::BufferUsages::STORAGE;
        PageBuffers {
            buffers: names
                .iter()
                .map(|name| GrowableBuffer::with_limit(&gpu.device, name, storage, limit))
                .collect(),
            bind_group: None,
            shadow_groups: HashMap::new(),
        }
    }
}

pub struct VectorGeometry {
    pages: Vec<PageBuffers>,
    /// The largest a page's buffer grows by doubling (the binding limit).
    limit: u64,
    indirect: GrowableBuffer,
    /// Shader and layout depend on whether the meshes' depth is multisampled.
    multisampled: Option<bool>,
    layout: Option<wgpu::BindGroupLayout>,
    shader: Option<wgpu::ShaderModule>,
    pipelines: HashMap<(Role, u32), wgpu::RenderPipeline>,
    /// The render targets the bind groups were made with.
    targets_generation: u64,
    opaque: Vec<Run>,
    transparent: Vec<Run>,
    masks: Vec<Vec<Run>>,
    /// The runs of each (light scope, clip scope)'s casters.
    shadows: HashMap<(i32, i32), Vec<Run>>,
    /// The pictures of the frame's camera views the bind groups were made
    /// with (handed in by `prepare`), and their generation.
    pictures: Option<(wgpu::TextureView, u64)>,
    pictures_sampler: wgpu::Sampler,
    /// The images fills show.
    images: ImageTexture,
    /// The image texture the bind group was made with.
    images_generation: u64,
}

impl VectorGeometry {
    /// Vector geometry whose page buffers grow by doubling up to `limit`
    /// bytes (the device's largest storage binding).
    pub fn new(gpu: &Gpu, limit: u64) -> VectorGeometry {
        VectorGeometry {
            pages: Vec::new(),
            limit,
            indirect: GrowableBuffer::new(
                &gpu.device,
                "vector draws",
                wgpu::BufferUsages::INDIRECT,
            ),
            multisampled: None,
            layout: None,
            shader: None,
            pipelines: HashMap::new(),
            targets_generation: u64::MAX,
            opaque: Vec::new(),
            transparent: Vec::new(),
            masks: Vec::new(),
            shadows: HashMap::new(),
            pictures: None,
            pictures_sampler: gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("camera views"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            images: ImageTexture::new(&gpu.device),
            images_generation: u64::MAX,
        }
    }

    /// Upload the frame's records. `mask_layers` holds one layer per clip
    /// scope; `mesh_depth` is the meshes' depth that hides shapes behind
    /// meshes; `targets_generation` changes when they are new textures;
    /// `pictures` are the camera views' pictures.
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
        pictures: Pictures,
    ) -> Result<()> {
        let changed = self.use_variant(gpu, multisampled)?;
        self.images.ensure(&gpu.device, &gpu.queue, data.images);
        let new_pictures = self
            .pictures
            .as_ref()
            .is_none_or(|(_, generation)| *generation != pictures.generation);
        if new_pictures {
            self.pictures = Some((pictures.array.clone(), pictures.generation));
        }
        // What every page's bind groups bind besides its own buffers.
        let everything = changed
            || targets_generation != self.targets_generation
            || new_pictures
            || self.images.generation != self.images_generation;
        while self.pages.len() < data.pages.len() {
            self.pages.push(PageBuffers::new(gpu, self.limit));
        }
        for (page, records) in self.pages.iter_mut().zip(&data.pages) {
            let mut grown = false;
            for (buffer, records) in page.buffers.iter_mut().zip(records) {
                grown |= write_records(buffer, &gpu.device, &gpu.queue, records);
            }
            if grown || everything {
                page.bind_group = None;
                page.shadow_groups.clear();
            }
        }

        let mut lists: Vec<&DrawList> = vec![data.opaque, data.transparent];
        lists.extend(data.shadows.iter().map(|(_, draws, _)| draws));
        lists.extend(data.masks.iter());
        let bytes: Vec<u8> = lists
            .iter()
            .flat_map(|list| list.bytes.iter().copied())
            .collect();
        self.indirect.upload(&gpu.device, &gpu.queue, &bytes);
        let mut first = 0u64;
        let mut ranges = lists.iter().map(|list| {
            let runs: Vec<Run> = list
                .runs
                .iter()
                .map(|&(page, count)| {
                    let run = Run { first, count, page };
                    first += count as u64;
                    run
                })
                .collect();
            runs
        });
        self.opaque = ranges.next().unwrap();
        self.transparent = ranges.next().unwrap();
        self.shadows = HashMap::new();
        for (scope, _, clips) in data.shadows {
            // The scope's page runs split by clip scope: they never cross one.
            let mut runs = ranges.next().unwrap().into_iter();
            for &(clip, count) in clips {
                let mut left = count;
                let mut chosen = Vec::new();
                while left > 0 {
                    let run = runs
                        .next()
                        .expect("page runs end where clip scope runs end");
                    left -= run.count;
                    chosen.push(run);
                }
                self.shadows.insert((*scope, clip), chosen);
            }
        }
        self.masks = ranges.collect();

        self.targets_generation = targets_generation;
        self.images_generation = self.images.generation;
        for index in 0..self.pages.len() {
            if self.pages[index].bind_group.is_none() {
                let group =
                    self.create_bind_group(gpu, index, frame_uniform, 0, mask_layers, mesh_depth);
                self.pages[index].bind_group = Some(group);
            }
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
        let runs = self.masks[scope].clone();
        self.draw(gpu, pass, Role::Mask, sample_count, &runs, None);
    }

    /// Draw all opaque, then all transparent shapes into the frame.
    pub fn draw_color(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass, sample_count: u32) {
        let (opaque, transparent) = (self.opaque.clone(), self.transparent.clone());
        self.draw(gpu, pass, Role::Opaque, sample_count, &opaque, None);
        self.draw(
            gpu,
            pass,
            Role::Transparent,
            sample_count,
            &transparent,
            None,
        );
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
        let stencil = if clipped {
            ShadowStencil::Clipped
        } else {
            ShadowStencil::None
        };
        let runs = self
            .shadows
            .get(&(scope, clip))
            .cloned()
            .unwrap_or_default();
        let frame = ShadowFrame {
            frames,
            offset,
            mask_layers,
            mesh_depth,
        };
        self.draw(gpu, pass, Role::Shadow(stencil), 1, &runs, Some(frame));
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
        let stencil = if inner {
            ShadowStencil::InnerMask
        } else {
            ShadowStencil::Mask
        };
        let runs = self.masks[clip as usize].clone();
        let frame = ShadowFrame {
            frames,
            offset,
            mask_layers,
            mesh_depth,
        };
        self.draw(gpu, pass, Role::Shadow(stencil), 1, &runs, Some(frame));
    }

    /// The shadow bind groups refer to the shadow frames buffer: forget
    /// them when it grows.
    pub fn forget_shadow_groups(&mut self) {
        for page in &mut self.pages {
            page.shadow_groups.clear();
        }
    }

    /// Issue `runs`, each with its page's bind group: the frame's, or
    /// (`shadow`) the one that sees through a shadow map's frame.
    fn draw(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass,
        role: Role,
        sample_count: u32,
        runs: &[Run],
        shadow: Option<ShadowFrame>,
    ) {
        if runs.iter().all(|run| run.count == 0) {
            return;
        }
        if let Some(frame) = &shadow {
            for run in runs {
                if !self.pages[run.page]
                    .shadow_groups
                    .contains_key(&frame.offset)
                {
                    let group = self.create_bind_group(
                        gpu,
                        run.page,
                        frame.frames,
                        frame.offset,
                        frame.mask_layers,
                        frame.mesh_depth,
                    );
                    self.pages[run.page]
                        .shadow_groups
                        .insert(frame.offset, group);
                }
            }
        }
        self.pipeline(gpu, role, sample_count);
        pass.set_pipeline(&self.pipelines[&(role, sample_count)]);
        let mut bound = None;
        for run in runs.iter().filter(|run| run.count > 0) {
            if bound != Some(run.page) {
                let page = &self.pages[run.page];
                let group = match &shadow {
                    Some(frame) => page.shadow_groups.get(&frame.offset),
                    None => page.bind_group.as_ref(),
                };
                pass.set_bind_group(0, group, &[]);
                bound = Some(run.page);
            }
            pass.multi_draw_indirect(&self.indirect.buffer, run.first * DRAW_SIZE, run.count);
        }
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
            // Transparent draws also write depth, so later objects blend
            // over them. The test is "less or equal": an object's pieces
            // share one depth, so where a stroke crosses itself its covered
            // inside draws over its own faded edge instead of leaving a seam
            // (stroke_coverage in vector.wgsl). The price: where a
            // translucent stroke crosses itself or its joints overlap, it
            // blends over itself and shows darker or brighter there
            // (native/docs/stroke_aa.md).
            Role::Opaque | Role::Transparent => (
                Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
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
        for binding in 1..=BUFFERS {
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
        let next = 1 + BUFFERS;
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

    /// The bind group of page `page`'s buffers, with the frame at `offset`
    /// of `frames`.
    fn create_bind_group(
        &self,
        gpu: &Gpu,
        page: usize,
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
        for (index, buffer) in self.pages[page].buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + index as u32,
                resource: buffer.binding(),
            });
        }
        let next = 1 + BUFFERS;
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
            resource: wgpu::BindingResource::TextureView(
                &self.pictures.as_ref().expect("handed in by prepare").0,
            ),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 3,
            resource: wgpu::BindingResource::Sampler(&self.pictures_sampler),
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
