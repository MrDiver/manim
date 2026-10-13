//! Depth of field: a view blurred by how far each pixel is from the
//! camera's focus, like a real lens with an open aperture (the camera's
//! `focus_distance` and `aperture`).
//!
//! Three fullscreen passes per view, after the view is composited sharp:
//!
//! ```text
//! view_distance.wgsl         each pixel's distance from the camera,
//!                            from the multisampled depth buffers
//! depth_of_field_tiles.wgsl  the largest blur in every tile of pixels
//! depth_of_field.wgsl        each pixel gathers the pixels whose blur
//!                            reaches it, as far as its tiles say
//! ```
//!
//! The blur of a pixel is a thin lens's circle of confusion
//! ([Potmesil and Chakravarty 1981](https://dl.acm.org/doi/10.1145/965161.806818)).
//! Gathering what reaches a pixel instead of spreading each pixel out
//! ("scatter as gather") and the tiles of largest blur follow Jorge
//! Jimenez,
//! [Next Generation Post Processing in Call of Duty: Advanced
//! Warfare](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/)
//! (SIGGRAPH 2014). The taps lie on a golden-angle spiral
//! ([Vogel 1979](https://doi.org/10.1016/0025-5564(79)90080-4)), turned
//! per pixel by interleaved gradient noise (also from Jimenez 2014).

use bytemuck::{Pod, Zeroable};

use crate::render::gpu::device::{Gpu, HDR_FORMAT};
use crate::render::error::Result;
use crate::render::gpu::fullscreen::{Blend, Fullscreen, FullscreenPass, TexturePool, UniformSlots};
use crate::render::pack::records::Frame;
use crate::render::gpu::profiler::GpuProfiler;

/// `DepthOfFieldParams` (lens.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthOfFieldParams {
    focus_distance: f32,
    aperture: f32,
    weight: f32,
    /// Pixels per side of a blur tile (`blur_tile`).
    tile: f32,
}

/// The largest blur radius, as a fraction of the image height
/// (`MAX_RADIUS` in lens.wgsl).
const MAX_BLUR: f32 = 0.05;

/// Pixels per side of a blur tile: at least the largest blur, so what
/// reaches a pixel lies in its tile or the eight around.
fn blur_tile(height: u32) -> u32 {
    ((MAX_BLUR * height as f32).ceil() as u32).max(1)
}

/// What the depth of field of one view needs besides its image: the
/// camera's lens, the view's depth buffers and its Frame uniform.
pub struct DepthOfField<'a> {
    pub focus_distance: f32,
    pub aperture: f32,
    pub mesh_depth: &'a wgpu::TextureView,
    pub vector_depth: &'a wgpu::TextureView,
    /// Samples per pixel of the depth buffers.
    pub samples: u32,
    pub frame_uniform: &'a wgpu::Buffer,
    pub size: (u32, u32),
}

/// Where the blurred view goes.
pub struct BlurTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub weight: f32,
    pub blend: Blend,
    pub clear: bool,
}

/// The GPU side of depth of field: the uniforms of its passes.
pub struct DepthOfFieldPasses {
    params: UniformSlots,
}

impl DepthOfFieldPasses {
    pub fn new(device: &wgpu::Device) -> DepthOfFieldPasses {
        DepthOfFieldPasses {
            params: UniformSlots::new(device, "depth of field params"),
        }
    }

    /// Blur `image` (the view composited sharp) through `lens` into `target`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        fullscreen: &mut Fullscreen,
        textures: &mut TexturePool,
        image: &wgpu::TextureView,
        lens: &DepthOfField,
        target: BlurTarget,
        profiler: &mut GpuProfiler,
    ) -> Result<()> {
        let frame_uniform = || {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: lens.frame_uniform,
                offset: 0,
                size: wgpu::BufferSize::new(size_of::<Frame>() as u64),
            })
        };

        // 1. Each pixel's distance from the camera.
        let distances = textures.get(
            &gpu.device,
            "view distances",
            lens.size,
            wgpu::TextureFormat::R32Float,
            1,
            wgpu::TextureUsages::empty(),
        );
        let distances_view = distances.create_view(&Default::default());
        let pass = FullscreenPass {
            shader: "view_distance.wgsl",
            entry_point: "fs_view_distance",
            depth_samples: Some(lens.samples),
        };
        let resources = [
            frame_uniform(),
            wgpu::BindingResource::TextureView(lens.mesh_depth),
            wgpu::BindingResource::TextureView(lens.vector_depth),
        ];
        fullscreen.run_pass(
            gpu,
            encoder,
            pass,
            &distances_view,
            wgpu::TextureFormat::R32Float,
            &resources,
            Blend::Replace,
            true,
            profiler,
            "depth of field",
        )?;

        // 2. The largest blur in every tile.
        let tile = blur_tile(lens.size.1);
        let params = DepthOfFieldParams {
            focus_distance: lens.focus_distance,
            aperture: lens.aperture,
            weight: target.weight,
            tile: tile as f32,
        };
        let params = self.params.write(&gpu.queue, bytemuck::bytes_of(&params));
        let tiles = textures.get(
            &gpu.device,
            "blur tiles",
            (lens.size.0.div_ceil(tile), lens.size.1.div_ceil(tile)),
            wgpu::TextureFormat::R32Float,
            1,
            wgpu::TextureUsages::empty(),
        );
        let tiles_view = tiles.create_view(&Default::default());
        let resources = [
            frame_uniform(),
            params.clone(),
            wgpu::BindingResource::TextureView(&distances_view),
        ];
        fullscreen.run(
            gpu,
            encoder,
            "depth_of_field_tiles.wgsl",
            "fs_blur_tiles",
            &tiles_view,
            wgpu::TextureFormat::R32Float,
            &resources,
            Blend::Replace,
            true,
            profiler,
            "depth of field",
        )?;

        // 3. The blur.
        let resources = [
            frame_uniform(),
            params,
            wgpu::BindingResource::TextureView(image),
            wgpu::BindingResource::TextureView(&distances_view),
            wgpu::BindingResource::TextureView(&tiles_view),
        ];
        fullscreen.run(
            gpu,
            encoder,
            "depth_of_field.wgsl",
            "fs_depth_of_field",
            target.view,
            HDR_FORMAT,
            &resources,
            target.blend,
            target.clear,
            profiler,
            "depth of field",
        )
    }
}
