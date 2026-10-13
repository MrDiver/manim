//! Reading the renderer's textures back, to look at how a frame is made
//! (scripts/render_internals.py draws the pictures of the renderer's
//! README with it). Not used to render frames.
//!
//! Multisampled and depth textures cannot be copied to the CPU, so a
//! fullscreen pass draws any texture into a plain `rgba32float` image
//! first: for each output pixel it loads the texel (sample 0 of a
//! multisampled one, `layer` of an array) unchanged, and the image is
//! copied out. Turning values into colors is left to the caller.

use crate::render::error::{Error, Result};
use crate::render::gpu::device::Gpu;

/// How a texture is read in WGSL.
#[derive(Clone, Copy)]
pub enum TextureKind {
    Float,
    FloatMultisampled,
    FloatArray,
    DepthMultisampled,
    DepthArray,
}

impl TextureKind {
    /// The WGSL type of the texture and the expression loading `value`
    /// (a vec4) at `pixel`.
    fn wgsl(self) -> (&'static str, &'static str) {
        match self {
            TextureKind::Float => ("texture_2d<f32>", "textureLoad(source, pixel, 0)"),
            TextureKind::FloatMultisampled => (
                "texture_multisampled_2d<f32>",
                "textureLoad(source, pixel, 0)",
            ),
            TextureKind::FloatArray => (
                "texture_2d_array<f32>",
                "textureLoad(source, pixel, layer, 0)",
            ),
            TextureKind::DepthMultisampled => (
                "texture_depth_multisampled_2d",
                "vec4<f32>(textureLoad(source, pixel, 0))",
            ),
            TextureKind::DepthArray => (
                "texture_depth_2d_array",
                "vec4<f32>(textureLoad(source, pixel, layer, 0))",
            ),
        }
    }
}

const SHADER: &str = r#"
@group(0) @binding(0) var source: TEXTURE;

struct Out { @builtin(position) position: vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Out {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return Out(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0));
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(floor(in.position.xy));
    let layer = LAYER;
    return LOAD;
}
"#;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The texels of `view` (`size` of them, `layer` of an array) as RGBA
/// floats, row by row.
pub fn read(
    gpu: &Gpu,
    view: &wgpu::TextureView,
    kind: TextureKind,
    size: (u32, u32),
    layer: u32,
) -> Result<Vec<f32>> {
    let device = &gpu.device;
    let (texture_type, load) = kind.wgsl();
    let code = SHADER
        .replace("TEXTURE", texture_type)
        .replace("LOAD", load)
        .replace("LAYER", &format!("{layer}i"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("inspect"),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("inspect"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(FORMAT.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("inspect"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(view),
        }],
    });
    let extent = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("inspect"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let row_bytes = (16 * size.0).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("inspect"),
        size: (row_bytes * size.1) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let target_view = target.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("inspect"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(size.1),
            },
        },
        extent,
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice
        .get_mapped_range()
        .map_err(|error| Error::Gpu(format!("reading the texture back failed: {error}")))?;
    let mut texels = Vec::with_capacity((4 * size.0 * size.1) as usize);
    for row in data.chunks(row_bytes as usize) {
        texels.extend_from_slice(bytemuck::cast_slice(&row[..(16 * size.0) as usize]));
    }
    Ok(texels)
}
