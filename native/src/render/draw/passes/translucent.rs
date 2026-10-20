//! The translucent pass: translucent meshes lit forward (mesh.wgsl's
//! fs_forward) and blended over the lit layer back to front, tested
//! against the G-buffer's depth and writing their own into it. It runs
//! after the vector pass, so 2D shapes are tested against the opaque
//! meshes only.

use crate::gpu::device::Gpu;
use crate::gpu::profiler::GpuProfiler;
use crate::render::draw::geometry::meshes::MeshGeometry;
use crate::render::draw::passes::lighting::LightingPass;
use crate::render::draw::targets::RenderTargets;

pub fn draw(
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    targets: &RenderTargets,
    meshes: &mut MeshGeometry,
    lighting: &LightingPass,
    profiler: &mut GpuProfiler,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("translucent"),
        color_attachments: &[Some(targets.lit_over_attachment())],
        depth_stencil_attachment: Some(targets.depth_test_attachment()),
        timestamp_writes: profiler.pass_timing("translucent"),
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some(group) = &lighting.forward_group {
        meshes.draw_translucent(
            gpu,
            &mut pass,
            targets.samples,
            &lighting.forward_layout,
            group,
        );
    }
}
