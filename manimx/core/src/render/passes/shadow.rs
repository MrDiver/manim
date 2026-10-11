//! Shadow map rendering: one depth layer per shadow view, casters drawn
//! with the mesh and vector pipelines seen from the light.
//!
//! Layers whose views say they are unchanged keep their depth from the
//! previous render and get no pass, unless the maps were just made again.
//!
//! Casters inside clipping groups cast only the shadow of what their masks
//! let through, as seen from the light: before a clip scope's casters, the
//! masks between the light and them are drawn into the stencil, outermost
//! first. (A light limited to a clip scope is inside it, and so are the
//! casters it lights: that scope's mask, and those around it, do not apply.) The outermost writes a
//! reference value, each nested one counts it up where the masks around it
//! are; the casters then draw where the stencil holds the last value.
//! References only grow within a layer, so values left by earlier clip
//! scopes never match.

use crate::packing::records::{Frame, ShadowView};
use crate::packing::scene_graph::NO_CLIP;
use crate::packing::shadows::ShadowViews;
use crate::render::buffer::GrowableBuffer;
use crate::render::context::{Gpu, SHADOW_FORMAT};
use crate::render::passes::mesh::MeshPass;
use crate::render::passes::vector::VectorPass;
use crate::render::profiler::GpuProfiler;

/// Frames of the shadow views share one uniform buffer; bind group offsets
/// must be multiples of 256 bytes.
const FRAME_STRIDE: usize = 512;
/// The largest stencil value (8 bits).
const STENCIL_MAX: u32 = 255;

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

    /// Width and height of every shadow map.
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// How many layers the last render's shadow views use.
    pub fn layer_count(&self) -> u32 {
        self.scopes.len() as u32
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
        scope_parents: &[i32],
    ) {
        let frames = &self.frames.buffer;
        for &layer in &self.draw_layers {
            let mut pass = self.begin(encoder, layer, profiler, true);
            let offset = (layer * FRAME_STRIDE) as u64;
            let scope = self.scopes[layer];
            let mut clips = meshes.shadow_clips(scope);
            clips.extend(vectors.shadow_clips(scope));
            clips.sort();
            clips.dedup();
            let mut reference = 0u32;
            let around_light = mask_chain(scope, scope_parents).len();
            for clip in clips {
                // The light is inside its own clip scope and the ones around
                // it: only the masks between it and the casters cut them.
                let chain = mask_chain(clip, scope_parents).split_off(around_light);
                if !chain.is_empty() {
                    if reference + chain.len() as u32 > STENCIL_MAX {
                        // Out of stencil values: start the layer's stencil over.
                        drop(pass);
                        pass = self.begin(encoder, layer, profiler, false);
                        reference = 0;
                    }
                    reference += 1;
                    pass.set_stencil_reference(reference);
                    vectors.draw_shadow_mask(
                        gpu,
                        &mut pass,
                        frames,
                        offset,
                        (chain[0], false),
                        mask_layers,
                        mesh_depth,
                    );
                    for &inner in &chain[1..] {
                        pass.set_stencil_reference(reference);
                        vectors.draw_shadow_mask(
                            gpu,
                            &mut pass,
                            frames,
                            offset,
                            (inner, true),
                            mask_layers,
                            mesh_depth,
                        );
                        reference += 1;
                    }
                    pass.set_stencil_reference(reference);
                }
                let clipped = !chain.is_empty();
                meshes.draw_shadow(
                    gpu,
                    &mut pass,
                    frames,
                    offset,
                    (scope, clip, clipped),
                    mask_layers,
                );
                vectors.draw_shadow(
                    gpu,
                    &mut pass,
                    frames,
                    offset,
                    (scope, clip, clipped),
                    mask_layers,
                    mesh_depth,
                );
            }
        }
    }

    fn begin<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        layer: usize,
        profiler: &mut GpuProfiler,
        clear_depth: bool,
    ) -> wgpu::RenderPass<'e> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadows"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.layers[layer],
                depth_ops: Some(wgpu::Operations {
                    load: if clear_depth {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Discard,
                }),
            }),
            timestamp_writes: profiler.pass_timing("shadows"),
            occlusion_query_set: None,
            multiview_mask: None,
        })
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
    // The lighting pass samples the depth; the stencil only serves the passes.
    let view = maps.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::DepthOnly,
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

/// The clip scopes `clip` is inside, outermost first (with `clip`).
fn mask_chain(mut clip: i32, scope_parents: &[i32]) -> Vec<i32> {
    let mut chain = Vec::new();
    while clip != NO_CLIP {
        chain.push(clip);
        clip = scope_parents[clip as usize];
    }
    chain.reverse();
    chain
}
