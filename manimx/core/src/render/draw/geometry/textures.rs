//! The mip levels of material textures: the image and successive
//! half-size versions down to 1x1, which the GPU samples between (the mip
//! map of Lance Williams, "Pyramidal Parametrics", SIGGRAPH 1983,
//! https://en.wikipedia.org/wiki/Mipmap).
//!
//! Each level is the one before shrunk with a box filter, computed as
//! Pillow's `Image.resize(size, Resampling.BOX)` computes it
//! (https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c),
//! so the levels are the same bytes they were when Pillow made them:
//! colors premultiplied by alpha while filtering, rows then columns,
//! weights in 22-bit fixed point.

use crate::robjects::Texture;

/// `(width, height, RGBA bytes)` of every level, the image first.
pub fn mip_levels(texture: &Texture) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = vec![(texture.width, texture.height, texture.pixels.clone())];
    loop {
        let (width, height, pixels) = levels.last().unwrap();
        if *width <= 1 && *height <= 1 {
            return levels;
        }
        let size = ((width / 2).max(1), (height / 2).max(1));
        let smaller = shrink(pixels, (*width, *height), size);
        levels.push((size.0, size.1, smaller));
    }
}

/// Fixed-point precision of the filter weights.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// The RGBA image `pixels` of `from` pixels resized to `to` with a box
/// filter.
pub fn shrink(pixels: &[u8], from: (u32, u32), to: (u32, u32)) -> Vec<u8> {
    let (width, height) = (from.0 as usize, from.1 as usize);
    let mut image: Vec<u8> = pixels.chunks_exact(4).flat_map(premultiply).collect();
    if to.0 != from.0 {
        let weights = box_weights(width, to.0 as usize);
        let mut rows = Vec::with_capacity(4 * to.0 as usize * height);
        for row in image.chunks_exact(4 * width) {
            for (first, taps) in &weights {
                rows.extend(filter(taps, |tap, band| row[4 * (first + tap) + band]));
            }
        }
        image = rows;
    }
    if to.1 != from.1 {
        let width = to.0 as usize;
        let weights = box_weights(height, to.1 as usize);
        let mut columns = Vec::with_capacity(4 * width * to.1 as usize);
        for (first, taps) in &weights {
            for x in 0..width {
                columns.extend(filter(taps, |tap, band| {
                    image[4 * ((first + tap) * width + x) + band]
                }));
            }
        }
        image = columns;
    }
    image.chunks_exact(4).flat_map(unpremultiply).collect()
}

/// One output pixel: the weighted sum of the input pixels `read(tap,
/// band)`, rounded, per band.
fn filter(taps: &[i64], read: impl Fn(usize, usize) -> u8) -> [u8; 4] {
    std::array::from_fn(|band| {
        let mut sum: i64 = 1 << (PRECISION_BITS - 1);
        for (tap, weight) in taps.iter().enumerate() {
            sum += read(tap, band) as i64 * weight;
        }
        (sum >> PRECISION_BITS).clamp(0, 255) as u8
    })
}

/// Per output pixel, the first input pixel it reads and the fixed-point
/// weights of the inputs from there (`precompute_coeffs` with the box
/// filter, `normalize_coeffs_8bpc`).
fn box_weights(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 0.5 * filter_scale;
    (0..output)
        .map(|x| {
            let center = (x as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let last = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let weights: Vec<f64> = (first..last)
                .map(|i| {
                    let at = (i as f64 - center + 0.5) / filter_scale;
                    if at > -0.5 && at <= 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let fixed = weights
                .iter()
                .map(|w| {
                    let w = if total != 0.0 { w / total } else { *w };
                    (0.5 + w * (1u64 << PRECISION_BITS) as f64) as i64
                })
                .collect();
            (first, fixed)
        })
        .collect()
}

/// RGBA to RGBa: colors times alpha, rounded (Pillow's `MULDIV255`).
pub fn premultiply(pixel: &[u8]) -> [u8; 4] {
    let alpha = pixel[3] as u32;
    let times = |value: u8| {
        let product = value as u32 * alpha + 128;
        (((product >> 8) + product) >> 8) as u8
    };
    [times(pixel[0]), times(pixel[1]), times(pixel[2]), pixel[3]]
}

/// RGBa back to RGBA.
fn unpremultiply(pixel: &[u8]) -> [u8; 4] {
    let alpha = pixel[3] as u32;
    if alpha == 0 || alpha == 255 {
        return [pixel[0], pixel[1], pixel[2], pixel[3]];
    }
    let divide = |value: u8| (255 * value as u32 / alpha).min(255) as u8;
    [
        divide(pixel[0]),
        divide(pixel[1]),
        divide(pixel[2]),
        pixel[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_halve_down_to_one_pixel() {
        let texture = Texture::new(5, 2, vec![200; 5 * 2 * 4]);
        let sizes: Vec<(u32, u32)> = mip_levels(&texture)
            .iter()
            .map(|(w, h, _)| (*w, *h))
            .collect();
        assert_eq!(sizes, [(5, 2), (2, 1), (1, 1)]);
    }

    #[test]
    fn levels_are_the_bytes_pillow_makes() {
        // A 5x3 image with transparent and partly transparent pixels, and
        // its levels as `Image.resize(size, Resampling.BOX)` makes them.
        let pixels = vec![
            139, 74, 229, 0, 169, 65, 6, 160, 149, 106, 38, 175, 188, 205, 175, 229, 98, 249, 10,
            148, 95, 86, 147, 198, 66, 39, 106, 213, 171, 45, 167, 255, 69, 81, 55, 14, 153, 178,
            215, 76, 52, 39, 250, 72, 215, 50, 161, 223, 112, 172, 161, 233, 38, 17, 89, 1, 221, 6,
            242, 17,
        ];
        let levels = mip_levels(&Texture::new(5, 3, pixels));
        assert_eq!(
            levels[1],
            (2, 1, vec![135, 78, 126, 170, 154, 204, 129, 81])
        );
        assert_eq!(levels[2], (1, 1, vec![141, 119, 127, 126]));
    }

    #[test]
    fn an_even_halving_averages_two_by_two() {
        let pixels = [
            [0, 0, 0, 255],
            [100, 100, 100, 255],
            [50, 50, 50, 255],
            [51, 51, 51, 255],
        ]
        .concat();
        let smaller = shrink(&pixels, (2, 2), (1, 1));
        // Rows first (50, 50.5 rounds to 51), then the column (50.5 to 51).
        assert_eq!(smaller, [51, 51, 51, 255]);
    }
}
