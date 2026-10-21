//! The compositor: frames made of the layers a renderer renders. It
//! drives the renderer (render/) and decides everything above one
//! render (frames.rs):
//!
//! ```text
//! for each sub-frame (one, or the RenderSamples renders):
//!     for each view (one per active clip, lowest first):
//!         for each of its camera views:
//!             picture = the view's scene rendered from that camera, composited
//!         layers = the view rendered (its camera views showing the pictures)
//!         lit = lit-stage effects (Bloom, custom shaders) on layers.lit
//!         image = composite(tone_map(lit), layers.vector) over image
//!                 (blurred by depth of field when the camera's lens is open)
//!     accumulator += weight * image
//! frame = frame-stage effects on the accumulator
//! output = encode to sRGB, read back (output/)
//! ```
//!
//! Which sub-frames and views there are is decided by the caller. The
//! renderer is never told about frames, sub-frames or effects; it renders
//! one scene with one view's settings at a time. The effects themselves
//! are in effects/.

pub mod effects;
pub mod frames;
pub mod output;
pub mod pictures;

use bytemuck::{Pod, Zeroable};

use crate::composite::effects::bloom::BloomPasses;
use crate::composite::effects::depth_of_field::{BlurTarget, DepthOfField, DepthOfFieldPasses};
use crate::composite::effects::shader_effect::ShaderEffects;
use crate::composite::effects::Effect;
use crate::error::Result;
use crate::gpu::device::{Gpu, HDR_FORMAT};
use crate::gpu::fullscreen::{Blend, Fullscreen, TexturePool, UniformSlots};

/// `CompositeParams` (composite.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParams {
    background: [f32; 4],
    tone_map: u32,
    weight: f32,
    _pad: [f32; 2],
}

/// The texture to render the composited frame of `views` into, and how.
pub struct CompositeTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub weight: f32,
    pub clear: bool,
    pub background: bool,
    pub blend: Blend,
}

pub struct Compositor {
    pub effects: Vec<Effect>,
    /// `ToneMapOperator` value (color.wgsl) of the view being composited
    /// (`View::tone_map`).
    pub tone_map: u32,
    pub textures: TexturePool,
    fullscreen: Fullscreen,
    composite_params: UniformSlots,
    bloom: BloomPasses,
    shader_effects: ShaderEffects,
    depth_of_field: DepthOfFieldPasses,
}

impl Compositor {
    pub fn new(gpu: &Gpu, effects: Vec<Effect>) -> Compositor {
        let device = &gpu.device;
        Compositor {
            effects,
            tone_map: 0,
            textures: TexturePool::default(),
            fullscreen: Fullscreen::new(device),
            composite_params: UniformSlots::new(device, "composite params"),
            bloom: BloomPasses::new(device),
            shader_effects: ShaderEffects::new(device),
            depth_of_field: DepthOfFieldPasses::new(device),
        }
    }

    pub fn has_shader_effects(&self) -> bool {
        self.effects
            .iter()
            .any(|effect| matches!(effect, Effect::Shader(_)))
    }

    /// The lit-stage effects (Bloom, lit shaders) on `lit`, in chain order.
    pub fn lit_effects(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        lit: &wgpu::Texture,
        frame_uniform: &wgpu::Buffer,
    ) -> Result<wgpu::Texture> {
        let mut lit = lit.clone();
        for effect in &self.effects {
            lit = match effect {
                Effect::Bloom(bloom) => self.bloom.apply(
                    gpu,
                    encoder,
                    &mut self.fullscreen,
                    &mut self.textures,
                    &lit,
                    bloom,
                )?,
                Effect::Shader(effect) if !effect.frame_stage => self.shader_effects.apply(
                    gpu,
                    encoder,
                    &mut self.textures,
                    effect,
                    &lit,
                    "lit",
                    frame_uniform,
                )?,
                Effect::Shader(_) => lit,
            };
        }
        Ok(lit)
    }

    /// The frame-stage shader effects on the accumulated frame.
    pub fn frame_effects(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::Texture,
        frame_uniform: &wgpu::Buffer,
    ) -> Result<wgpu::Texture> {
        let mut frame = frame.clone();
        for effect in &self.effects {
            if let Effect::Shader(effect) = effect {
                if effect.frame_stage {
                    frame = self.shader_effects.apply(
                        gpu,
                        encoder,
                        &mut self.textures,
                        effect,
                        &frame,
                        "frame",
                        frame_uniform,
                    )?;
                }
            }
        }
        Ok(frame)
    }

    /// `fs_composite`: tone mapped lit layer and vector layer over the
    /// background, `weight` times into `target`; through `depth_of_field`
    /// when given.
    #[allow(clippy::too_many_arguments)]
    pub fn composite(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        lit: &wgpu::Texture,
        vector: &wgpu::Texture,
        target: CompositeTarget,
        background: [f32; 4],
        depth_of_field: Option<DepthOfField>,
    ) -> Result<()> {
        let Some(lens) = depth_of_field else {
            return self.composite_layers(gpu, encoder, lit, vector, target, background);
        };
        // Composite the view sharp first, then blur it into the target.
        let image = self.textures.get(
            &gpu.device,
            "depth of field",
            lens.size,
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::empty(),
        );
        let image_view = image.create_view(&Default::default());
        let sharp = CompositeTarget {
            view: &image_view,
            weight: 1.0,
            clear: true,
            background: target.background,
            blend: Blend::Replace,
        };
        self.composite_layers(gpu, encoder, lit, vector, sharp, background)?;
        let blurred = BlurTarget {
            view: target.view,
            weight: target.weight,
            blend: target.blend,
            clear: target.clear,
        };
        self.depth_of_field.apply(
            gpu,
            encoder,
            &mut self.fullscreen,
            &mut self.textures,
            &image_view,
            &lens,
            blurred,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn composite_layers(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        lit: &wgpu::Texture,
        vector: &wgpu::Texture,
        target: CompositeTarget,
        background: [f32; 4],
    ) -> Result<()> {
        let params = CompositeParams {
            background: if target.background {
                background
            } else {
                [0.0; 4]
            },
            tone_map: self.tone_map,
            weight: target.weight,
            _pad: [0.0; 2],
        };
        let lit_view = lit.create_view(&Default::default());
        let vector_view = vector.create_view(&Default::default());
        let params = self
            .composite_params
            .write(&gpu.queue, bytemuck::bytes_of(&params));
        let resources = [
            params,
            wgpu::BindingResource::TextureView(&lit_view),
            wgpu::BindingResource::TextureView(&vector_view),
        ];
        self.fullscreen.run(
            gpu,
            encoder,
            "composite.wgsl",
            "fs_composite",
            target.view,
            HDR_FORMAT,
            &resources,
            target.blend,
            target.clear,
        )
    }

    /// `fs_accumulate`: `weight` times a stack of views into the accumulator.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        stack: &wgpu::Texture,
        accumulator: &wgpu::TextureView,
        weight: f32,
        clear: bool,
    ) -> Result<()> {
        let params = CompositeParams {
            background: [0.0; 4],
            tone_map: self.tone_map,
            weight,
            _pad: [0.0; 2],
        };
        let stack_view = stack.create_view(&Default::default());
        let params = self
            .composite_params
            .write(&gpu.queue, bytemuck::bytes_of(&params));
        let resources = [params, wgpu::BindingResource::TextureView(&stack_view)];
        self.fullscreen.run(
            gpu,
            encoder,
            "composite.wgsl",
            "fs_accumulate",
            accumulator,
            HDR_FORMAT,
            &resources,
            Blend::Additive,
            clear,
        )
    }
}
