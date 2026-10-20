//! Draw: a view's packet in, the lit and vector layers out (`Drawer`).
//! The passes (passes/) run in order: mask, shadow, gbuffer, lighting,
//! vector, translucent; each draws the geometry (geometry/) it needs.
//! Draw decides nothing about the scene and never sees the packer: what
//! to draw and which records changed is all in the packet and its
//! uploads (packet/). It keeps only GPU buffers, pipelines and textures.

pub mod geometry;
pub mod passes;
pub mod targets;

use crate::error::Result;
use crate::gpu::buffer::GrowableBuffer;
use crate::gpu::device::Gpu;
use crate::gpu::profiler::GpuProfiler;
use crate::render::draw::geometry::meshes::{MeshData, MeshGeometry};
use crate::render::draw::geometry::vectors::{VectorData, VectorGeometry};
use crate::render::draw::passes::lighting::LightingPass;
use crate::render::draw::passes::shadow::ShadowPass;
use crate::render::draw::passes::{gbuffer, mask, translucent, vector};
use crate::render::draw::targets::RenderTargets;
use crate::render::packet::records::Frame;
use crate::render::packet::{Packet, Records, Uploads};

/// The pictures of a view's camera views, made before the view (by the
/// compositor): a 2D array texture, one layer per camera view, which the
/// camera views' shapes show. `generation` changes whenever it is a new
/// texture.
#[derive(Clone, Copy)]
pub struct Pictures<'a> {
    pub array: &'a wgpu::TextureView,
    pub generation: u64,
}

/// Upload the ranges of `records` written since the last upload into
/// `buffer` (all of them when the packer says so); returns whether the
/// buffer grew, so bind groups referring to it must be made again.
pub fn write_records(
    buffer: &mut GrowableBuffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    records: &Records,
) -> bool {
    buffer.write(
        device,
        queue,
        records.bytes,
        records.record_size,
        records.written.as_deref(),
    )
}

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
    /// A drawer whose vector page buffers grow by doubling up to
    /// `page_bytes` (the binding limit the packer's pages keep to).
    pub fn new(gpu: &Gpu, page_bytes: u64) -> Result<Drawer> {
        Ok(Drawer {
            frame_uniform: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame"),
                size: size_of::<Frame>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            vectors: VectorGeometry::new(gpu, page_bytes),
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

    /// Render `packet` (reading the records of `uploads`, and showing
    /// `pictures` on the camera views' shapes) and the camera's `frame`
    /// into the layers; the passes are recorded into `encoder`, which the
    /// caller submits; `profiler` times the passes.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        packet: &Packet,
        uploads: Uploads,
        pictures: Pictures,
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

        let Uploads {
            vector_pages,
            meshes: mesh_records,
        } = uploads;
        let data = VectorData {
            pages: vector_pages,
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
            pictures,
        )?;

        if let (Some(meshes), Some([vertices, indices])) = (&packet.meshes, mesh_records) {
            let data = MeshData {
                objects: bytemuck::cast_slice(&meshes.output.objects),
                materials: bytemuck::cast_slice(&meshes.output.materials),
                vertices,
                indices,
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
