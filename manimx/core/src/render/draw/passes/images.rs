//! The images that 2D fills show (`Fill.image`, `Shape2D.fill_image`): one
//! layer each of a 2D array texture with mip levels, sampled by the vector
//! pass (vector.wgsl's `tinted_image`).
//!
//! The layers of an array share one size, the largest width and height of
//! the frame's images (at most `MAX_SIZE`; larger images are shrunk to fit).
//! Each image sits in the top left corner of its layer, and its last column
//! and row are repeated into the rest, so the lower mip levels blend its
//! edge with itself, as clamping would, never with the padding. A storage
//! buffer holds each layer's region: the center and size of the image in
//! texture coordinates.
//!
//! Pixels are stored as their sRGB values (vector colors are sRGB values
//! too) premultiplied by alpha, so filtering does not darken transparent
//! edges. The texture is made again only when the frame's list of images
//! changes.

use wgpu::util::DeviceExt;

use crate::render::draw::passes::textures::{mip_levels, premultiply, shrink};
use crate::robjects::Texture;

/// The largest layer width and height.
const MAX_SIZE: u32 = 4096;

pub struct ImageTexture {
    pub array_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// `[center u, center v, width, height]` per layer.
    pub regions: wgpu::Buffer,
    /// Increases whenever the texture is made again (bind groups refer to it).
    pub generation: u64,
    /// The texture uids of the layers.
    uids: Vec<u64>,
}

impl ImageTexture {
    pub fn new(device: &wgpu::Device) -> ImageTexture {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("images"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let empty = Texture::new(1, 1, vec![0; 4]);
        let (array_view, regions) = create(device, None, &[&empty]);
        ImageTexture {
            array_view,
            sampler,
            regions,
            generation: 0,
            uids: Vec::new(),
        }
    }

    /// Make the layers hold `images` in this order (nothing to do when
    /// they already do).
    pub fn ensure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: &[std::sync::Arc<Texture>],
    ) {
        let uids: Vec<u64> = images.iter().map(|image| image.uid).collect();
        if uids == self.uids || images.is_empty() {
            return;
        }
        let images: Vec<&Texture> = images.iter().map(|image| image.as_ref()).collect();
        (self.array_view, self.regions) = create(device, Some(queue), &images);
        self.uids = uids;
        self.generation += 1;
    }
}

/// The array view and region buffer of `images` (uploaded with `queue`;
/// None leaves the texels unwritten, for the empty start).
fn create(
    device: &wgpu::Device,
    queue: Option<&wgpu::Queue>,
    images: &[&Texture],
) -> (wgpu::TextureView, wgpu::Buffer) {
    let fitted: Vec<(u32, u32, Vec<u8>)> = images.iter().map(|image| fit(image)).collect();
    let width = fitted.iter().map(|image| image.0).max().unwrap_or(1);
    let height = fitted.iter().map(|image| image.1).max().unwrap_or(1);
    let padded: Vec<Vec<(u32, u32, Vec<u8>)>> = fitted
        .iter()
        .map(|(w, h, pixels)| {
            let layer = Texture::new(width, height, pad(pixels, (*w, *h), (width, height)));
            mip_levels(&layer)
                .into_iter()
                .map(|(w, h, pixels)| {
                    (w, h, pixels.chunks_exact(4).flat_map(premultiply).collect())
                })
                .collect()
        })
        .collect();
    let levels = padded[0].len() as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("images"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: images.len() as u32,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if let Some(queue) = queue {
        for (layer, image) in padded.iter().enumerate() {
            for (level, (w, h, pixels)) in image.iter().enumerate() {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * w),
                        rows_per_image: Some(*h),
                    },
                    wgpu::Extent3d {
                        width: *w,
                        height: *h,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
    let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let regions: Vec<[f32; 4]> = fitted
        .iter()
        .map(|(w, h, _)| {
            let (u, v) = (*w as f32 / width as f32, *h as f32 / height as f32);
            [u / 2.0, v / 2.0, u, v]
        })
        .collect();
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("image regions"),
        contents: bytemuck::cast_slice(&regions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    (array_view, buffer)
}

/// `image` shrunk to fit `MAX_SIZE` (keeping its aspect), as
/// `(width, height, RGBA bytes)`.
fn fit(image: &Texture) -> (u32, u32, Vec<u8>) {
    let (width, height) = (image.width, image.height);
    let factor = (MAX_SIZE as f64 / width.max(height) as f64).min(1.0);
    if factor >= 1.0 {
        return (width, height, image.pixels.clone());
    }
    let size = (
        ((width as f64 * factor).round() as u32).max(1),
        ((height as f64 * factor).round() as u32).max(1),
    );
    (size.0, size.1, shrink(&image.pixels, (width, height), size))
}

/// `pixels` (`from` pixels) in the top left of a `to` image, the last
/// column and row repeated into the rest.
fn pad(pixels: &[u8], from: (u32, u32), to: (u32, u32)) -> Vec<u8> {
    let (width, height) = (from.0 as usize, from.1 as usize);
    let mut padded = Vec::with_capacity(4 * (to.0 * to.1) as usize);
    for y in 0..to.1 as usize {
        let row = &pixels[4 * width * y.min(height - 1)..][..4 * width];
        padded.extend_from_slice(row);
        let last = &row[4 * (width - 1)..];
        for _ in width..to.0 as usize {
            padded.extend_from_slice(last);
        }
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_repeats_the_last_column_and_row() {
        // 2 x 1 pixels: red, green.
        let pixels = [255, 0, 0, 255, 0, 255, 0, 255];
        let padded = pad(&pixels, (2, 1), (3, 2));
        let green = [0, 255, 0, 255];
        assert_eq!(&padded[..8], &pixels);
        assert_eq!(&padded[8..12], &green);
        assert_eq!(&padded[12..24], &padded[..12]);
    }

    #[test]
    fn large_images_shrink_to_fit_keeping_their_aspect() {
        let image = Texture::new(MAX_SIZE * 2, 2, vec![255; (4 * MAX_SIZE * 2 * 2) as usize]);
        let (width, height, pixels) = fit(&image);
        assert_eq!((width, height), (MAX_SIZE, 1));
        assert_eq!(pixels.len(), (4 * width * height) as usize);
    }
}
