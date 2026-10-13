//! The pictures of a frame's camera views: one layer each of a 2D array
//! texture, rendered before the frame and sampled by the vector pass.
//! See manimx/animation/nodes/camera_view.py.

use crate::render::gpu::device::HDR_FORMAT;

pub struct CameraViewTexture {
    texture: wgpu::Texture,
    /// All layers, as the vector pass samples them.
    pub array_view: wgpu::TextureView,
    /// One view per layer, to render a picture into.
    layers: Vec<wgpu::TextureView>,
    pub sampler: wgpu::Sampler,
    /// Increases whenever the texture is made again (bind groups refer to it).
    pub generation: u64,
}

impl CameraViewTexture {
    pub fn new(device: &wgpu::Device) -> CameraViewTexture {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("camera views"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (texture, array_view, layers) = create(device, 1, (1, 1));
        CameraViewTexture {
            texture,
            array_view,
            layers,
            sampler,
            generation: 0,
        }
    }

    /// Make the texture hold at least `count` layers of `size` pixels.
    pub fn ensure(&mut self, device: &wgpu::Device, count: usize, size: (u32, u32)) {
        let current = self.texture.size();
        if current.depth_or_array_layers as usize >= count
            && (current.width, current.height) == size
        {
            return;
        }
        (self.texture, self.array_view, self.layers) = create(device, count.max(1), size);
        self.generation += 1;
    }

    /// Layer `layer`, to render a picture into.
    pub fn layer(&self, layer: usize) -> &wgpu::TextureView {
        &self.layers[layer]
    }
}

fn create(
    device: &wgpu::Device,
    count: usize,
    size: (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView, Vec<wgpu::TextureView>) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("camera views"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: count as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let layers = (0..count as u32)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    (texture, array_view, layers)
}
