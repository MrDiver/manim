//! Every texture one frame renders into, made again when the size changes.
//! Ports `RenderTargets` (manimx/renderer/webgpu/targets.py):
//!
//!     masks       one RGBA layer per clip scope (+ a scratch target to draw into)
//!     G-buffer    4 color targets + depth, multisampled (see gbuffer.wgsl)
//!     lit         multisampled HDR target of the lighting pass, resolved
//!     vector      multisampled premultiplied RGBA of 2D shapes and curves,
//!                 depth-tested against the G-buffer depth, resolved
//!
//! The resolved `lit` and `vector` textures are the renderer's output; the
//! composer combines them into an image.

use super::context::{COLOR_FORMAT, DEPTH_FORMAT, GBUFFER_FORMATS, HDR_FORMAT};

pub struct RenderTargets {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub mask_capacity: u32,
    pub gbuffer_views: Vec<wgpu::TextureView>,
    pub depth_view: wgpu::TextureView,
    vector_depth_view: wgpu::TextureView,
    pub lit: wgpu::Texture,
    pub vector: wgpu::Texture,
    lit_view: wgpu::TextureView,
    vector_view: wgpu::TextureView,
    lit_msaa: Option<wgpu::TextureView>,
    vector_msaa: Option<wgpu::TextureView>,
    mask_layers: wgpu::Texture,
    pub mask_layers_view: wgpu::TextureView,
    mask_scratch: wgpu::Texture,
    mask_scratch_view: wgpu::TextureView,
    mask_msaa: Option<wgpu::TextureView>,
}

const ATTACHMENT: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT;
const SAMPLED: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING;

impl RenderTargets {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        samples: u32,
        mask_capacity: u32,
    ) -> RenderTargets {
        let texture = |label: &str, format, sample_count, usage, layers| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let msaa = |label: &str, format| {
            (samples > 1).then(|| view(&texture(label, format, samples, ATTACHMENT, 1)))
        };

        // G-buffer: sampled per sample by the lighting pass.
        let gbuffer_views = GBUFFER_FORMATS
            .iter()
            .enumerate()
            .map(|(index, format)| {
                view(&texture(
                    &format!("gbuffer {index}"),
                    *format,
                    samples,
                    ATTACHMENT | SAMPLED,
                    1,
                ))
            })
            .collect();
        let depth_view = view(&texture(
            "depth",
            DEPTH_FORMAT,
            samples,
            ATTACHMENT | SAMPLED,
            1,
        ));
        // The vector pass orders shapes with biased depth in its own buffer
        // and tests against the meshes' depth (above) explicitly.
        let vector_depth_view = view(&texture(
            "vector depth",
            DEPTH_FORMAT,
            samples,
            ATTACHMENT,
            1,
        ));

        // Resolved outputs, read by the composer.
        let output = ATTACHMENT | SAMPLED | wgpu::TextureUsages::COPY_SRC;
        let lit = texture("lit", HDR_FORMAT, 1, output, 1);
        let vector = texture("vector", COLOR_FORMAT, 1, output, 1);

        // One layer per clip scope, sampled by clipped content. Mask passes
        // cannot render into it while it is bound for sampling, so they
        // render into a scratch target that is copied into the layer.
        let mask_layers = texture(
            "mask layers",
            COLOR_FORMAT,
            1,
            SAMPLED | wgpu::TextureUsages::COPY_DST,
            mask_capacity,
        );
        let mask_layers_view = mask_layers.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let mask_scratch = texture(
            "mask scratch",
            COLOR_FORMAT,
            1,
            ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            1,
        );

        RenderTargets {
            width,
            height,
            samples,
            mask_capacity,
            gbuffer_views,
            depth_view,
            vector_depth_view,
            lit_view: view(&lit),
            vector_view: view(&vector),
            lit,
            vector,
            lit_msaa: msaa("lit msaa", HDR_FORMAT),
            vector_msaa: msaa("vector msaa", COLOR_FORMAT),
            mask_scratch_view: view(&mask_scratch),
            mask_layers,
            mask_layers_view,
            mask_scratch,
            mask_msaa: msaa("mask msaa", COLOR_FORMAT),
        }
    }

    /// Transparent scratch target; mask shapes accumulate alpha.
    pub fn mask_attachment(&self) -> wgpu::RenderPassColorAttachment<'_> {
        color(&self.mask_scratch_view, self.mask_msaa.as_ref())
    }

    pub fn copy_mask_to_layer(&self, encoder: &mut wgpu::CommandEncoder, layer: u32) {
        encoder.copy_texture_to_texture(
            self.mask_scratch.as_image_copy(),
            wgpu::TexelCopyTextureInfo {
                texture: &self.mask_layers,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn gbuffer_attachments(&self) -> Vec<Option<wgpu::RenderPassColorAttachment<'_>>> {
        self.gbuffer_views
            .iter()
            .map(|view| Some(clear_color(view, None)))
            .collect()
    }

    pub fn lit_attachment(&self) -> wgpu::RenderPassColorAttachment<'_> {
        color(&self.lit_view, self.lit_msaa.as_ref())
    }

    pub fn vector_attachment(&self) -> wgpu::RenderPassColorAttachment<'_> {
        color(&self.vector_view, self.vector_msaa.as_ref())
    }

    /// The meshes' depth buffer, written by the geometry pass.
    pub fn depth_attachment(&self) -> wgpu::RenderPassDepthStencilAttachment<'_> {
        depth(&self.depth_view)
    }

    /// The vector pass's own depth buffer (painter's-order layers).
    pub fn vector_depth_attachment(&self) -> wgpu::RenderPassDepthStencilAttachment<'_> {
        depth(&self.vector_depth_view)
    }
}

fn clear_color<'a>(
    view: &'a wgpu::TextureView,
    resolve_target: Option<&'a wgpu::TextureView>,
) -> wgpu::RenderPassColorAttachment<'a> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    }
}

/// A color attachment that clears to transparent and ends up in `output`
/// (rendered into `msaa` and resolved, when multisampled).
fn color<'a>(
    output: &'a wgpu::TextureView,
    msaa: Option<&'a wgpu::TextureView>,
) -> wgpu::RenderPassColorAttachment<'a> {
    match msaa {
        None => clear_color(output, None),
        Some(msaa) => wgpu::RenderPassColorAttachment {
            view: msaa,
            depth_slice: None,
            resolve_target: Some(output),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Discard,
            },
        },
    }
}

fn depth(view: &wgpu::TextureView) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    }
}
