//! The vector pass: 2D shapes and 3D curves into the vector layer, the
//! opaque ones first, then the transparent ones back to front, with their
//! own depth and tested against the opaque meshes' depth.

use crate::render::draw::geometry::vectors::VectorGeometry;
use crate::render::draw::targets::RenderTargets;
use crate::render::gpu::device::Gpu;
use crate::render::gpu::profiler::GpuProfiler;

pub fn draw(
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    targets: &RenderTargets,
    vectors: &mut VectorGeometry,
    profiler: &mut GpuProfiler,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("vectors"),
        color_attachments: &[Some(targets.vector_attachment())],
        depth_stencil_attachment: Some(targets.vector_depth_attachment()),
        timestamp_writes: profiler.pass_timing("vectors"),
        occlusion_query_set: None,
        multiview_mask: None,
    });
    vectors.draw_color(gpu, &mut pass, targets.samples);
}
