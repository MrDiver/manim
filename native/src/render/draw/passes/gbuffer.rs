//! The G-buffer pass: opaque meshes into the multisampled G-buffer (base
//! color, normal, roughness and metallic, emission, clip scope) and its
//! depth, per sample. No light is computed yet (lighting.rs). Without
//! meshes it only clears them.

use crate::gpu::device::Gpu;
use crate::gpu::profiler::GpuProfiler;
use crate::render::draw::geometry::meshes::MeshGeometry;
use crate::render::draw::targets::RenderTargets;

pub fn draw(
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    targets: &RenderTargets,
    meshes: Option<&mut MeshGeometry>,
    profiler: &mut GpuProfiler,
) {
    let attachments = targets.gbuffer_attachments();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("geometry"),
        color_attachments: &attachments,
        depth_stencil_attachment: Some(targets.depth_attachment()),
        timestamp_writes: profiler.pass_timing("geometry"),
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some(meshes) = meshes {
        meshes.draw_gbuffer(gpu, &mut pass, targets.samples);
    }
}
