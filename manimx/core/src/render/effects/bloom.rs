//! Bloom: light brighter than a threshold spreads into a soft glow.
//!
//! The bright part of the lit layer is drawn into half resolution, then
//! halved again level by level (a mip chain), and added back up level by
//! level: every level blurs a little, and together they make a wide,
//! smooth glow at the cost of a few small passes. This is the
//! downsample/upsample chain of Jorge Jimenez,
//! [Next Generation Post Processing in Call of Duty: Advanced
//! Warfare](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/)
//! (SIGGRAPH 2014), with a soft threshold ("knee") so the glow fades in
//! instead of starting at a hard edge. The filters are in
//! shaders/bloom.wgsl.

use bytemuck::{Pod, Zeroable};

use crate::render::context::{Gpu, HDR_FORMAT};
use crate::render::error::Result;
use crate::render::fullscreen::{Blend, Fullscreen, TexturePool, UniformSlots};
use crate::render::profiler::GpuProfiler;

/// The settings of `Bloom` (manimx/renderer/effects.py).
#[derive(Clone, Copy)]
pub struct Bloom {
    pub threshold: f32,
    pub knee: f32,
    pub intensity: f32,
    pub radius: f32,
    pub levels: u32,
}

/// `BloomParams` (bloom.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomParams {
    texel: [f32; 2],
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    _pad: [f32; 2],
}

/// The GPU side of bloom: the uniforms of its passes.
pub struct BloomPasses {
    params: UniformSlots,
}

impl BloomPasses {
    pub fn new(device: &wgpu::Device) -> BloomPasses {
        BloomPasses {
            params: UniformSlots::new(device, "bloom params"),
        }
    }

    /// `lit` with bloom; returns the texture it is in.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        fullscreen: &mut Fullscreen,
        textures: &mut TexturePool,
        lit: &wgpu::Texture,
        bloom: &Bloom,
        profiler: &mut GpuProfiler,
    ) -> Result<wgpu::Texture> {
        let (width, height) = (lit.width(), lit.height());
        // The chain starts at half resolution; stop before mips get tiny.
        let smallest = (width.min(height) / 4).max(1);
        let levels = bloom.levels.min(smallest.ilog2()).max(1);
        let device = &gpu.device;
        let chain = textures.get(
            device,
            "bloom chain",
            ((width / 2).max(1), (height / 2).max(1)),
            HDR_FORMAT,
            levels,
            wgpu::TextureUsages::empty(),
        );
        let output = textures.get(
            device,
            "bloom output",
            (width, height),
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::empty(),
        );
        let mips: Vec<wgpu::TextureView> = (0..levels)
            .map(|level| {
                chain.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let mip_size = |level: u32| {
            (
                (chain.width() >> level).max(1),
                (chain.height() >> level).max(1),
            )
        };
        let lit_view = lit.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());

        let params = &mut self.params;
        let mut step = |encoder: &mut wgpu::CommandEncoder,
                        entry: &'static str,
                        source: &wgpu::TextureView,
                        size: (u32, u32),
                        target: &wgpu::TextureView,
                        blend: Blend,
                        clear: bool| {
            let values = BloomParams {
                texel: [1.0 / size.0 as f32, 1.0 / size.1 as f32],
                threshold: bloom.threshold,
                knee: bloom.knee,
                intensity: bloom.intensity,
                radius: bloom.radius,
                _pad: [0.0; 2],
            };
            let linear_clamp = fullscreen.linear_clamp.clone();
            let mut resources = vec![
                params.write(&gpu.queue, bytemuck::bytes_of(&values)),
                wgpu::BindingResource::TextureView(source),
                wgpu::BindingResource::Sampler(&linear_clamp),
            ];
            if entry == "fs_combine" {
                resources.push(wgpu::BindingResource::TextureView(&mips[0]));
            }
            fullscreen.run(
                gpu,
                encoder,
                "bloom.wgsl",
                entry,
                target,
                HDR_FORMAT,
                &resources,
                blend,
                clear,
                profiler,
                "bloom",
            )
        };
        // The bright part, at half resolution.
        step(
            encoder,
            "fs_prefilter",
            &lit_view,
            (width, height),
            &mips[0],
            Blend::Replace,
            true,
        )?;
        // Down the chain: each level half the size of the one before.
        for level in 0..levels - 1 {
            step(
                encoder,
                "fs_downsample",
                &mips[level as usize],
                mip_size(level),
                &mips[level as usize + 1],
                Blend::Replace,
                true,
            )?;
        }
        // Up the chain: each smaller level, blurred, added onto the next larger one.
        for level in (1..levels).rev() {
            step(
                encoder,
                "fs_upsample",
                &mips[level as usize],
                mip_size(level),
                &mips[level as usize - 1],
                Blend::Additive,
                false,
            )?;
        }
        // The glow over the lit layer.
        step(
            encoder,
            "fs_combine",
            &lit_view,
            (width, height),
            &output_view,
            Blend::Replace,
            true,
        )?;
        Ok(output)
    }
}
