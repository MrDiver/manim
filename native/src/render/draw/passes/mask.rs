//! The mask pass: every clip scope's mask shapes into its layer of the
//! mask texture, parents first; what is drawn inside a scope is
//! multiplied by its layer.

use crate::gpu::device::Gpu;
use crate::render::draw::geometry::vectors::VectorGeometry;
use crate::render::draw::targets::RenderTargets;

pub fn draw(
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    targets: &RenderTargets,
    vectors: &mut VectorGeometry,
    scope_count: usize,
) {
    for scope in 0..scope_count {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mask"),
                color_attachments: &[Some(targets.mask_attachment())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            vectors.draw_mask(gpu, &mut pass, targets.samples, scope);
        }
        targets.copy_mask_to_layer(encoder, scope as u32);
    }
}
