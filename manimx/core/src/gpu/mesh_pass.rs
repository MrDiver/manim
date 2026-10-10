//! GPU side of the mesh geometry pass: buffers, material textures,
//! pipelines and draws into the G-buffer. Ports `MeshPass`
//! (manimx/renderer/webgpu/mesh_pass.py).
//!
//! Draws are grouped by the material's texture set and surface shader: one
//! bind group and one `multi_draw_indexed_indirect` per group, and one
//! pipeline per surface shader (mesh.wgsl with the shader's code in place
//! of surface_shader.wgsl's default). Textures are uploaded once per
//! (texture, color space) and kept while any frame uses them; their mip
//! levels come from Python (core_callbacks.texture_levels), made as the
//! reference makes them.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use super::buffer::{GrowableBuffer, Records};
use super::context::{Gpu, DEPTH_FORMAT, GBUFFER_FORMATS, SHADOW_FORMAT};
use crate::callbacks::bytes_of;
use crate::meshes::DrawGroup;
use crate::records::{DrawIndexed, Frame};

/// Color space of each texture slot, in `PbrMaterial.textures` order: base
/// color and emissive hold sRGB colors, the others linear data.
const SLOT_IS_SRGB: [bool; 5] = [true, false, false, false, true];
/// Neutral values for missing textures: white, except a flat normal map.
const SLOT_DEFAULT: [[u8; 4]; 5] = [[255; 4], [255; 4], [128, 128, 255, 255], [255; 4], [255; 4]];
const DRAW_SIZE: u64 = size_of::<DrawIndexed>() as u64;

/// The meshes of one frame, as the pass uploads them.
pub struct MeshData<'a> {
    pub objects: &'a [u8],
    pub materials: &'a [u8],
    pub vertices: Records<'a>,
    pub indices: Records<'a>,
    pub groups: &'a [DrawGroup],
    /// Per clip scope (NO_CLIP first), the shadow casters' draws.
    pub shadow_draws: &'a [(i32, Vec<DrawIndexed>)],
}

pub struct MeshPass {
    objects: GrowableBuffer,
    vertices: GrowableBuffer,
    materials: GrowableBuffer,
    indices: GrowableBuffer,
    indirect: GrowableBuffer,
    shader: wgpu::ShaderModule,
    scene_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    defaults: Vec<wgpu::TextureView>,
    /// (texture uid, sRGB) -> view.
    textures: HashMap<(u64, bool), wgpu::TextureView>,
    material_groups: HashMap<Vec<u64>, wgpu::BindGroup>,
    /// By sample count and surface shader uid (0: none).
    pipelines: HashMap<(u32, u64), wgpu::RenderPipeline>,
    /// mesh.wgsl with each surface shader's code, by shader uid.
    surface_shaders: HashMap<u64, wgpu::ShaderModule>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    scene_group: Option<wgpu::BindGroup>,
    targets_generation: u64,
    /// (material group key, surface shader uid, first draw, draw count) per group.
    draw_groups: Vec<(Vec<u64>, u64, u64, u32)>,
    shadows: HashMap<i32, (u64, u32)>,
    shadow_groups: HashMap<u64, wgpu::BindGroup>,
}

impl MeshPass {
    pub fn new(gpu: &Gpu) -> PyResult<MeshPass> {
        let device = &gpu.device;
        let storage = wgpu::BufferUsages::STORAGE;
        let (scene_layout, material_layout) = create_layouts(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let defaults = SLOT_DEFAULT
            .iter()
            .zip(SLOT_IS_SRGB)
            .map(|(rgba, srgb)| upload_texture(gpu, &[(1, 1, rgba.to_vec())], srgb))
            .collect();
        Ok(MeshPass {
            objects: GrowableBuffer::new(device, "mesh objects", storage),
            vertices: GrowableBuffer::new(device, "mesh vertices", storage),
            materials: GrowableBuffer::new(device, "materials", storage),
            indices: GrowableBuffer::new(device, "mesh indices", wgpu::BufferUsages::INDEX),
            indirect: GrowableBuffer::new(device, "mesh draws", wgpu::BufferUsages::INDIRECT),
            shader: gpu.shader("mesh.wgsl", &[])?,
            scene_layout,
            material_layout,
            sampler,
            defaults,
            textures: HashMap::new(),
            material_groups: HashMap::new(),
            pipelines: HashMap::new(),
            surface_shaders: HashMap::new(),
            shadow_pipeline: None,
            scene_group: None,
            targets_generation: u64::MAX,
            draw_groups: Vec::new(),
            shadows: HashMap::new(),
            shadow_groups: HashMap::new(),
        })
    }

    /// Upload the frame's meshes and their materials: vertices and indices
    /// in the ranges written since the last frame, the rest in full.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        py: Python<'_>,
        gpu: &Gpu,
        data: &MeshData,
        frame_uniform: &wgpu::Buffer,
        mask_layers: &wgpu::TextureView,
        targets_generation: u64,
        callbacks_module: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let mut grown = self.objects.upload(device, queue, data.objects);
        grown |= self.vertices.write_records(device, queue, &data.vertices);
        grown |= self.materials.upload(device, queue, data.materials);
        self.indices.write_records(device, queue, &data.indices);

        // Material groups' draws first, then the shadow casters per scope.
        let mut draws: Vec<DrawIndexed> = data
            .groups
            .iter()
            .flat_map(|group| group.draws.iter().copied())
            .collect();
        let mut first = draws.len() as u64;
        self.shadows.clear();
        for (scope, casters) in data.shadow_draws {
            self.shadows.insert(*scope, (first, casters.len() as u32));
            first += casters.len() as u64;
            draws.extend(casters);
        }
        self.indirect
            .upload(device, queue, bytemuck::cast_slice(&draws));
        if grown || self.scene_group.is_none() || targets_generation != self.targets_generation {
            self.targets_generation = targets_generation;
            self.scene_group = Some(self.create_scene_group(gpu, frame_uniform, 0, mask_layers));
            self.shadow_groups.clear();
        }

        let mut used_textures = HashSet::new();
        let mut used_groups = HashSet::new();
        let mut used_shaders = HashSet::new();
        self.draw_groups.clear();
        let mut first = 0u64;
        for group in data.groups {
            let uids = group
                .textures
                .iter()
                .map(|texture| texture_uid(py, texture))
                .collect::<PyResult<Vec<u64>>>()?;
            used_groups.insert(uids.clone());
            for ((uid, srgb), texture) in uids.iter().zip(SLOT_IS_SRGB).zip(&group.textures) {
                if *uid != 0 {
                    used_textures.insert((*uid, srgb));
                    if let Entry::Vacant(slot) = self.textures.entry((*uid, srgb)) {
                        let levels =
                            callbacks_module.call_method1("texture_levels", (texture.bind(py),))?;
                        let levels = levels
                            .try_iter()?
                            .map(|level| {
                                let level = level?;
                                Ok((
                                    level.get_item(0)?.extract()?,
                                    level.get_item(1)?.extract()?,
                                    bytes_of(&level.get_item(2)?)?,
                                ))
                            })
                            .collect::<PyResult<Vec<(u32, u32, Vec<u8>)>>>()?;
                        slot.insert(upload_texture(gpu, &levels, srgb));
                    }
                }
            }
            if !self.material_groups.contains_key(&uids) {
                let bind_group = self.create_material_group(gpu, &uids);
                self.material_groups.insert(uids.clone(), bind_group);
            }
            let shader = self.surface_shader(py, gpu, &group.surface_shader, callbacks_module)?;
            used_shaders.insert(shader);
            self.draw_groups
                .push((uids, shader, first, group.draws.len() as u32));
            first += group.draws.len() as u64;
        }
        // Forget textures and bind groups no longer used by any mesh.
        self.textures.retain(|key, _| used_textures.contains(key));
        self.material_groups
            .retain(|key, _| used_groups.contains(key));
        self.surface_shaders
            .retain(|uid, _| used_shaders.contains(uid));
        self.pipelines
            .retain(|(_, uid), _| *uid == 0 || used_shaders.contains(uid));
        Ok(())
    }

    /// The uid of a group's surface shader (0 for none), its shader module
    /// compiled on first use.
    fn surface_shader(
        &mut self,
        py: Python<'_>,
        gpu: &Gpu,
        shader: &Py<PyAny>,
        callbacks_module: &Bound<'_, PyAny>,
    ) -> PyResult<u64> {
        let shader = shader.bind(py);
        if shader.is_none() {
            return Ok(0);
        }
        let uid: u64 = shader.getattr("uid")?.extract()?;
        if let Entry::Vacant(slot) = self.surface_shaders.entry(uid) {
            let default: String = callbacks_module
                .getattr("DEFAULT_SURFACE_SHADER")?
                .extract()?;
            let code: String = shader.call_method0("wgsl")?.extract()?;
            slot.insert(gpu.checked_shader("mesh.wgsl", &[(&default, &code)])?);
        }
        Ok(uid)
    }

    pub fn draw(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass, sample_count: u32) {
        if self.draw_groups.is_empty() {
            return;
        }
        for (_, shader, _, _) in &self.draw_groups {
            if !self.pipelines.contains_key(&(sample_count, *shader)) {
                let module = self.surface_shaders.get(shader).unwrap_or(&self.shader);
                let pipeline = self.create_pipeline(gpu, module, sample_count);
                self.pipelines.insert((sample_count, *shader), pipeline);
            }
        }
        pass.set_bind_group(0, self.scene_group.as_ref(), &[]);
        pass.set_index_buffer(self.indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
        for (key, shader, first, count) in &self.draw_groups {
            pass.set_pipeline(&self.pipelines[&(sample_count, *shader)]);
            pass.set_bind_group(1, self.material_groups.get(key), &[]);
            pass.multi_draw_indexed_indirect(&self.indirect.buffer, first * DRAW_SIZE, *count);
        }
    }

    /// Draw the depth of `scope`'s shadow-casting meshes, seen from the
    /// frame at `offset` of `frames`.
    pub fn draw_shadow(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass,
        frames: &wgpu::Buffer,
        offset: u64,
        scope: i32,
        mask_layers: &wgpu::TextureView,
    ) {
        let Some(&(first, count)) = self.shadows.get(&scope) else {
            return;
        };
        if count == 0 {
            return;
        }
        if !self.shadow_groups.contains_key(&offset) {
            let group = self.create_scene_group(gpu, frames, offset, mask_layers);
            self.shadow_groups.insert(offset, group);
        }
        if self.shadow_pipeline.is_none() {
            self.shadow_pipeline = Some(self.create_shadow_pipeline(gpu));
        }
        pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.shadow_groups.get(&offset), &[]);
        pass.set_index_buffer(self.indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.multi_draw_indexed_indirect(&self.indirect.buffer, first * DRAW_SIZE, count);
    }

    pub fn forget_shadow_groups(&mut self) {
        self.shadow_groups.clear();
    }

    // -- setup ------------------------------------------------------------

    fn create_pipeline(
        &self,
        gpu: &Gpu,
        module: &wgpu::ShaderModule,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&self.scene_layout), Some(&self.material_layout)],
                immediate_size: 0,
            });
        let targets: Vec<Option<wgpu::ColorTargetState>> = GBUFFER_FORMATS
            .iter()
            .map(|format| Some((*format).into()))
            .collect();
        gpu.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mesh"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                // Double sided: the fragment shader flips back-face normals.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn create_shadow_pipeline(&self, gpu: &Gpu) -> wgpu::RenderPipeline {
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&self.scene_layout)],
                immediate_size: 0,
            });
        gpu.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mesh shadow"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 0,
                        slope_scale: 1.5,
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some("fs_shadow"),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn create_material_group(&self, gpu: &Gpu, uids: &[u64]) -> wgpu::BindGroup {
        let views: Vec<&wgpu::TextureView> = uids
            .iter()
            .zip(SLOT_IS_SRGB)
            .enumerate()
            .map(|(slot, (uid, srgb))| {
                if *uid == 0 {
                    &self.defaults[slot]
                } else {
                    &self.textures[&(*uid, srgb)]
                }
            })
            .collect();
        let mut entries: Vec<wgpu::BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(slot, view)| wgpu::BindGroupEntry {
                binding: slot as u32,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: SLOT_IS_SRGB.len() as u32,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        });
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.material_layout,
            entries: &entries,
        })
    }

    fn create_scene_group(
        &self,
        gpu: &Gpu,
        frames: &wgpu::Buffer,
        offset: u64,
        mask_layers: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let frame = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: frames,
            offset,
            size: wgpu::BufferSize::new(size_of::<Frame>() as u64),
        });
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh scene"),
            layout: &self.scene_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.objects.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.vertices.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.materials.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(mask_layers),
                },
            ],
        })
    }
}

/// `Texture.uid`, or 0 for a missing texture (None).
fn texture_uid(py: Python<'_>, texture: &Py<PyAny>) -> PyResult<u64> {
    let texture = texture.bind(py);
    if texture.is_none() {
        return Ok(0);
    }
    texture.getattr("uid")?.extract()
}

fn upload_texture(gpu: &Gpu, levels: &[(u32, u32, Vec<u8>)], srgb: bool) -> wgpu::TextureView {
    let (width, height, _) = levels[0];
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (w, h, data)) in levels.iter().enumerate() {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(*h),
            },
            wgpu::Extent3d {
                width: *w,
                height: *h,
                depth_or_array_layers: 1,
            },
        );
    }
    texture.create_view(&Default::default())
}

fn create_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let vertex = wgpu::ShaderStages::VERTEX;
    let fragment = wgpu::ShaderStages::FRAGMENT;
    let storage = |binding, visibility| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh scene"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vertex | fragment,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(1, vertex | fragment), // objects
            storage(2, vertex),            // vertices
            storage(3, fragment),          // materials
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: fragment,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..SLOT_IS_SRGB.len() as u32)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        })
        .collect();
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: SLOT_IS_SRGB.len() as u32,
        visibility: fragment,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    });
    let material = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material"),
        entries: &entries,
    });
    (scene, material)
}
