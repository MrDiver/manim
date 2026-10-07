// Color transforms from linear lighting values to the frame's sRGB values.

/// Khronos PBR Neutral tone mapping: leaves base colors untouched up to
/// ~0.8 and compresses brighter values smoothly, so lit Manim colors stay
/// recognizable. https://github.com/KhronosGroup/ToneMapping
fn tone_map_pbr_neutral(color: vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    let lowest = min(color.r, min(color.g, color.b));
    let offset = select(0.04, lowest - 6.25 * lowest * lowest, lowest < 0.08);
    var result = color - offset;
    let peak = max(result.r, max(result.g, result.b));
    if peak < start_compression {
        return result;
    }
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    result *= new_peak / peak;
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(result, vec3<f32>(new_peak), g);
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let c = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}
