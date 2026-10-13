//! Colors: render objects hold sRGB colors, as they are written; lighting
//! and blending happen in linear light.

/// One sRGB channel in linear light: the sRGB transfer function undone
/// (IEC 61966-2-1, https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22)).
pub fn linear(srgb: f64) -> f64 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// An sRGB color in linear light.
pub fn linear_rgb(srgb: [f64; 3]) -> [f64; 3] {
    srgb.map(linear)
}
