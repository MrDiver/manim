//! Draw: a view's packet in, the lit and vector layers out (`Drawer`).
//! The passes (passes/) run in order: mask, shadow, gbuffer, lighting,
//! vector, translucent; each draws the geometry (geometry/) it needs.
//! Draw decides nothing about the scene: what to draw and what changed is
//! in the packet and the packer's arenas.

pub mod geometry;
pub mod passes;
pub mod targets;

use crate::render::draw::geometry::meshes::{MeshData, MeshGeometry};
use crate::render::draw::geometry::vectors::{VectorData, VectorGeometry};
use crate::render::draw::passes::lighting::LightingPass;
use crate::render::draw::passes::shadow::ShadowPass;
use crate::render::draw::passes::{gbuffer, mask, translucent, vector};
use crate::render::draw::targets::RenderTargets;
use crate::render::error::Result;
use crate::render::gpu::buffer::Records;
use crate::render::gpu::device::Gpu;
use crate::render::gpu::inspect::TextureKind;
use crate::render::gpu::profiler::GpuProfiler;
use crate::render::pack::packer::{Packer, Packet};
use crate::render::pack::records::Frame;

/// The size and sampling of the targets a view is drawn into.
#[derive(Clone, Copy, PartialEq)]
pub struct TargetSettings {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub shadow_resolution: u32,
}

pub struct Drawer {
    frame_uniform: wgpu::Buffer,
    vectors: VectorGeometry,
    meshes: MeshGeometry,
    lighting_pass: LightingPass,
    shadow_pass: ShadowPass,
    targets: Option<RenderTargets>,
    /// Increases whenever the targets are made again (bind groups refer to them).
    targets_generation: u64,
    /// Whether the G-buffer and lit layer of these targets are known to be
    /// empty (the frames since they were cleared had no meshes).
    empty_3d: Option<u64>,
}

impl Drawer {
    /// Make the camera view texture hold `count` pictures of `size` pixels.
    pub fn ensure_camera_views(&mut self, gpu: &Gpu, count: usize, size: (u32, u32)) {
        self.vectors.camera_views.ensure(&gpu.device, count, size);
    }

    /// Layer `layer` of the camera view texture, to render a picture into.
    pub fn camera_view_layer(&self, layer: usize) -> wgpu::TextureView {
        self.vectors.camera_views.layer(layer).clone()
    }

    /// A texture of the last render to look at (`inspect.rs`): its view,
    /// how to read it and its size, by name.
    pub fn inspected(&self, name: &str) -> Option<(wgpu::TextureView, TextureKind, (u32, u32))> {
        let targets = self.targets.as_ref()?;
        let size = (targets.width, targets.height);
        let multisampled =
            |view: &wgpu::TextureView| Some((view.clone(), TextureKind::FloatMultisampled, size));
        match name {
            "lit" => Some((
                targets.lit.create_view(&Default::default()),
                TextureKind::Float,
                size,
            )),
            "vector" => Some((
                targets.vector.create_view(&Default::default()),
                TextureKind::Float,
                size,
            )),
            "albedo" => multisampled(&targets.gbuffer_views[0]),
            "normal" => multisampled(&targets.gbuffer_views[1]),
            "emissive" => multisampled(&targets.gbuffer_views[2]),
            "scope" => multisampled(&targets.gbuffer_views[3]),
            "depth" => Some((
                targets.depth_view.clone(),
                TextureKind::DepthMultisampled,
                size,
            )),
            "masks" => Some((
                targets.mask_layers_view.clone(),
                TextureKind::FloatArray,
                size,
            )),
            "shadow_maps" => {
                let side = self.shadow_pass.resolution();
                Some((
                    self.shadow_pass.maps_view.clone(),
                    TextureKind::DepthArray,
                    (side, side),
                ))
            }
            _ => None,
        }
    }

    /// How many shadow map layers the last render drew into.
    pub fn shadow_layers(&self) -> u32 {
        self.shadow_pass.layer_count()
    }

    pub fn new(gpu: &Gpu) -> Result<Drawer> {
        Ok(Drawer {
            frame_uniform: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame"),
                size: size_of::<Frame>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            vectors: VectorGeometry::new(gpu),
            meshes: MeshGeometry::new(gpu)?,
            lighting_pass: LightingPass::new(gpu),
            shadow_pass: ShadowPass::new(gpu),
            targets: None,
            targets_generation: 0,
            empty_3d: None,
        })
    }

    /// Camera, viewport and time of the latest render (frame.wgsl).
    pub fn frame_uniform(&self) -> &wgpu::Buffer {
        &self.frame_uniform
    }

    /// Render `packet` (with the records of `packer`'s arenas) and the
    /// camera's `frame` into the layers; the passes are recorded into
    /// `encoder`, which the caller submits; `profiler` times the passes.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        packer: &mut Packer,
        packet: &Packet,
        frame: &Frame,
        settings: TargetSettings,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut GpuProfiler,
    ) -> Result<&RenderTargets> {
        self.ensure_targets(gpu, settings, packet.scope_parents.len() as u32);
        let generation = self.targets_generation;
        let targets = self.targets.as_ref().unwrap();
        let samples = settings.samples;
        let has_meshes = packet.meshes.is_some();
        // A frame without meshes skips the 3D passes; its G-buffer (whose
        // depth the vector pass tests against) and lit layer only need to
        // be empty, and stay empty while the next frames have no meshes.
        let clear_3d = !has_meshes && self.empty_3d != Some(generation);
        gpu.queue
            .write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(frame));

        let vectors = &mut packer.vectors;
        let ranges: Vec<Option<Vec<(usize, usize)>>> = vectors
            .arenas_mut()
            .into_iter()
            .map(|(_, arena)| arena.ranges())
            .collect();
        let arenas = [
            &vectors.objects,
            &vectors.instances,
            &vectors.fill_vertices,
            &vectors.fill_colors,
            &vectors.stroke_points,
            &vectors.stroke_curves,
        ];
        let mut ranges = ranges.into_iter();
        let buffers = arenas.map(|arena| Records {
            bytes: arena.bytes(),
            record_size: arena.record_size,
            written: ranges.next().unwrap(),
        });
        let data = VectorData {
            buffers,
            opaque: &packet.draws.opaque,
            transparent: &packet.draws.transparent,
            masks: &packet.draws.masks,
            shadows: &packet.draws.shadows,
            images: &packet.draws.images,
        };
        let mesh_depth = &targets.depth_view;
        self.vectors.prepare(
            gpu,
            &data,
            &self.frame_uniform,
            &targets.mask_layers_view,
            mesh_depth,
            samples > 1,
            generation,
        )?;

        if let Some(meshes) = &packet.meshes {
            let mesh_arenas = &mut packer.meshes;
            let vertex_ranges = mesh_arenas.vertices.ranges();
            let index_ranges = mesh_arenas.indices.ranges();
            let data = MeshData {
                objects: bytemuck::cast_slice(&meshes.output.objects),
                materials: bytemuck::cast_slice(&meshes.output.materials),
                vertices: Records {
                    bytes: mesh_arenas.vertices.bytes(),
                    record_size: mesh_arenas.vertices.record_size,
                    written: vertex_ranges,
                },
                indices: Records {
                    bytes: mesh_arenas.indices.bytes(),
                    record_size: mesh_arenas.indices.record_size,
                    written: index_ranges,
                },
                groups: &meshes.output.groups,
                translucent: &meshes.output.translucent,
                translucent_shadow_draws: &meshes.output.translucent_shadow_draws,
                shadow_draws: &meshes.output.shadow_draws,
            };
            self.meshes.prepare(
                gpu,
                &data,
                &self.frame_uniform,
                &targets.mask_layers_view,
                generation,
            )?;
            let shadows_changed =
                self.shadow_pass
                    .prepare(gpu, &meshes.shadows, settings.shadow_resolution);
            if shadows_changed {
                self.meshes.forget_shadow_groups();
                self.vectors.forget_shadow_groups();
            }
            self.lighting_pass.prepare(
                gpu,
                bytemuck::cast_slice(&meshes.lights),
                &packet.scope_parents,
                &self.frame_uniform,
                &targets.gbuffer_views,
                &targets.depth_view,
                samples,
                &self.shadow_pass,
                shadows_changed,
                generation,
            )?;
            self.shadow_pass.draw(
                gpu,
                encoder,
                &mut self.meshes,
                &mut self.vectors,
                profiler,
                &targets.mask_layers_view,
                mesh_depth,
                &packet.scope_parents,
            );
        }

        mask::draw(
            gpu,
            encoder,
            targets,
            &mut self.vectors,
            packet.scope_parents.len(),
            profiler,
        );
        let translucent = has_meshes && self.meshes.has_translucent();
        if has_meshes || clear_3d {
            let meshes = has_meshes.then_some(&mut self.meshes);
            gbuffer::draw(gpu, encoder, targets, meshes, profiler);
            self.lighting_pass
                .draw(gpu, encoder, targets, has_meshes, translucent, profiler)?;
        }
        self.empty_3d = if has_meshes { None } else { Some(generation) };
        vector::draw(gpu, encoder, targets, &mut self.vectors, profiler);
        if translucent {
            translucent::draw(
                gpu,
                encoder,
                targets,
                &mut self.meshes,
                &self.lighting_pass,
                profiler,
            );
        }
        Ok(targets)
    }

    fn ensure_targets(&mut self, gpu: &Gpu, settings: TargetSettings, mask_count: u32) {
        let current = self.targets.as_ref();
        let fits = current.is_some_and(|t| {
            t.width == settings.width
                && t.height == settings.height
                && t.samples == settings.samples
                && t.mask_capacity >= mask_count
        });
        if !fits {
            // Grow in steps when the masks do not fit; a new size alone
            // keeps the capacity.
            let mut capacity = current.map_or(1, |t| t.mask_capacity);
            if capacity < mask_count {
                capacity = mask_count.max(2 * capacity);
            }
            self.targets = Some(RenderTargets::new(
                &gpu.device,
                settings.width,
                settings.height,
                settings.samples,
                capacity,
            ));
            self.targets_generation += 1;
        }
    }
}
