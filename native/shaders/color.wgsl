// Color transforms between linear light and the frame's sRGB values.

const TONE_MAP_NONE: u32 = 0u;
const TONE_MAP_PBR_NEUTRAL: u32 = 1u;
const TONE_MAP_ACES: u32 = 2u;
const TONE_MAP_AGX: u32 = 3u;
const TONE_MAP_REINHARD: u32 = 4u;

/// Map linear HDR radiance into [0, 1] with the operator selected by
/// `curve` (one of the TONE_MAP_* values).
fn tone_map(color: vec3<f32>, curve: u32) -> vec3<f32> {
    switch curve {
        case TONE_MAP_PBR_NEUTRAL: { return tone_map_pbr_neutral(color); }
        case TONE_MAP_ACES: { return tone_map_aces(color); }
        case TONE_MAP_AGX: { return tone_map_agx(color); }
        // Reinhard et al. 2002, "Photographic tone reproduction for digital
        // images", https://doi.org/10.1145/566654.566575 (the simple global curve).
        case TONE_MAP_REINHARD: { return color / (1.0 + color); }
        default: { return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)); }
    }
}

/// Khronos PBR Neutral: leaves base colors untouched up to ~0.8 and
/// compresses brighter values smoothly. https://github.com/KhronosGroup/ToneMapping
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

/// Krzysztof Narkowicz's fit of the ACES filmic curve: punchy contrast,
/// desaturated highlights.
/// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn tone_map_aces(color: vec3<f32>) -> vec3<f32> {
    let x = color * 0.6;  // the fit expects pre-exposed input
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Troy Sobotka's AgX (Blender 4's default view transform), in the minimal
/// polynomial fit by Benjamin Wrensch.
/// https://iolite-engine.com/blog_posts/minimal_agx_implementation
fn tone_map_agx(color: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104,
    );
    let outset = mat3x3<f32>(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116,
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var x = inset * color;
    x = clamp(log2(max(x, vec3<f32>(1e-10))), vec3<f32>(min_ev), vec3<f32>(max_ev));
    x = (x - min_ev) / (max_ev - min_ev);
    let x2 = x * x;
    let x4 = x2 * x2;
    x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
    x = outset * x;
    return clamp(pow(max(x, vec3<f32>(0.0)), vec3<f32>(2.2)), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// The sRGB transfer function (IEC 61966-2-1),
/// https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22).
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let c = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}
