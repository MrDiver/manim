//! Shadow map rendering: one depth layer per shadow view, casters drawn
//! with the mesh and vector pipelines seen from the light. Ports
//! `ShadowPass` (manimx/renderer/webgpu/shadow_pass.py).
//!
//! Layers whose views say they are unchanged keep their depth from the
//! previous render and get no pass, unless the maps were just made again.

use super::buffer::GrowableBuffer;
use super::context::{Gpu, SHADOW_FORMAT};
use super::mesh_pass::MeshPass;
use super::profiler::GpuProfiler;
use super::vector_pass::VectorPass;
use crate::records::{Frame, ShadowView};
use crate::shadows::ShadowViews;

/// Frames of the shadow views share one uniform buffer; bind group offsets
/// must be multiples of 256 bytes.
const FRAME_STRIDE: usize = 512;

pub struct ShadowPass {
    frames: GrowableBuffer,
    pub views: GrowableBuffer,
    pub sampler: wgpu::Sampler,
    maps: wgpu::Texture,
    pub maps_view: wgpu::TextureView,
    layers: Vec<wgpu::TextureView>,
    resolution: u32,
    scopes: Vec<i32>,
    draw_layers: Vec<usize>,
}

impl ShadowPass {
    pub fn new(gpu: &Gpu) -> ShadowPass {
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let (maps, maps_view, layers) = create_maps(&gpu.device, 1, 1);
        ShadowPass {
            frames: GrowableBuffer::new(&gpu.device, "shadow frames", wgpu::BufferUsages::UNIFORM),
            views: GrowableBuffer::new(&gpu.device, "shadow views", wgpu::BufferUsages::STORAGE),
            sampler,
            maps,
            maps_view,
            layers,
            resolution: 1,
            scopes: Vec::new(),
            draw_layers: Vec::new(),
        }
    }

    /// Upload the views. Returns whether bind groups that refer to the
    /// maps, the views or the frames must be made again.
    pub fn prepare(&mut self, gpu: &Gpu, shadows: &ShadowViews, resolution: u32) -> bool {
        let count = shadows.frames.len();
        self.scopes = shadows.scopes.clone();
        let mut frames = vec![0u8; FRAME_STRIDE * count.max(1)];
        for (index, frame) in shadows.frames.iter().enumerate() {
            frames[index * FRAME_STRIDE..index * FRAME_STRIDE + size_of::<Frame>()]
                .copy_from_slice(bytemuck::bytes_of(frame));
        }
        let mut changed = self.frames.upload(&gpu.device, &gpu.queue, &frames);
        let views: Vec<ShadowView> = if count > 0 {
            shadows.views.clone()
        } else {
            vec![bytemuck::Zeroable::zeroed()]
        };
        changed |= self
            .views
            .upload(&gpu.device, &gpu.queue, bytemuck::cast_slice(&views));
        if count == 0 {
            self.draw_layers.clear();
            return changed;
        }
        let recreated = self.ensure_maps(gpu, resolution, count as u32);
        self.draw_layers = (0..count)
            .filter(|&layer| recreated || shadows.changed.get(layer).copied().unwrap_or(true))
            .collect();
        changed || recreated
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        meshes: &mut MeshPass,
        vectors: &mut VectorPass,
        profiler: &mut GpuProfiler,
        mask_layers: &wgpu::TextureView,
        mesh_depth: &wgpu::TextureView,
    ) {
        for &layer in &self.draw_layers {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadows"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.layers[layer],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: profiler.pass_timing("shadows"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let offset = (layer * FRAME_STRIDE) as u64;
            let scope = self.scopes[layer];
            meshes.draw_shadow(
                gpu,
                &mut pass,
                &self.frames.buffer,
                offset,
                scope,
                mask_layers,
            );
            vectors.draw_shadow(
                gpu,
                &mut pass,
                &self.frames.buffer,
                offset,
                scope,
                mask_layers,
                mesh_depth,
            );
        }
    }

    fn ensure_maps(&mut self, gpu: &Gpu, resolution: u32, layers: u32) -> bool {
        let current = self.maps.depth_or_array_layers();
        if self.resolution == resolution && current >= layers {
            return false;
        }
        let layers = layers.max(current); // never fewer layers than before
        (self.maps, self.maps_view, self.layers) = create_maps(&gpu.device, resolution, layers);
        self.resolution = resolution;
        true
    }
}

fn create_maps(
    device: &wgpu::Device,
    resolution: u32,
    layers: u32,
) -> (wgpu::Texture, wgpu::TextureView, Vec<wgpu::TextureView>) {
    let maps = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow maps"),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SHADOW_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = maps.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let layer_views = (0..layers)
        .map(|layer| {
            maps.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    (maps, view, layer_views)
}
