// Turn the accumulated linear frame into the output image: straight alpha,
// sRGB values (fs_finalize), or the NV12 planes a video encoder takes
// (fs_luma, fs_chroma).

#include "fullscreen.wgsl"
#include "color.wgsl"

@group(0) @binding(0) var accumulated: texture_2d<f32>;

/// The output pixel at `pixel`: sRGB values, straight alpha.
fn output_pixel(pixel: vec2<i32>) -> vec4<f32> {
    let color = textureLoad(accumulated, pixel, 0);
    if color.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(linear_to_srgb(color.rgb / color.a), min(color.a, 1.0));
}

@fragment
fn fs_finalize(in: FullscreenVertex) -> @location(0) vec4<f32> {
    return output_pixel(vec2<i32>(floor(in.position.xy)));
}

// NV12 is Y'CbCr 4:2:0: one luma value per pixel, then one (Cb, Cr) pair
// per 2x2 pixels. The values are those of BT.709 in limited range (luma
// 16..235, chroma 16..240), what HD video players assume
// (ITU-R BT.709-6, https://www.itu.int/rec/R-REC-BT.709). Alpha is
// dropped, as a video has none: the colors are the sRGB values the RGBA
// output holds.

const LUMA_WEIGHTS = vec3<f32>(0.2126, 0.7152, 0.0722);

@fragment
fn fs_luma(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let rgb = output_pixel(vec2<i32>(floor(in.position.xy))).rgb;
    return vec4<f32>((16.0 + 219.0 * dot(rgb, LUMA_WEIGHTS)) / 255.0, 0.0, 0.0, 1.0);
}

/// One (Cb, Cr) pair from the mean color of the 2x2 pixels it covers.
@fragment
fn fs_chroma(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let corner = 2 * vec2<i32>(floor(in.position.xy));
    let rgb = 0.25 * (output_pixel(corner).rgb + output_pixel(corner + vec2<i32>(1, 0)).rgb
        + output_pixel(corner + vec2<i32>(0, 1)).rgb + output_pixel(corner + vec2<i32>(1, 1)).rgb);
    let luma = dot(rgb, LUMA_WEIGHTS);
    let blue = (rgb.b - luma) / (2.0 * (1.0 - LUMA_WEIGHTS.b));
    let red = (rgb.r - luma) / (2.0 * (1.0 - LUMA_WEIGHTS.r));
    return vec4<f32>((128.0 + 224.0 * vec2<f32>(blue, red)) / 255.0, 0.0, 1.0);
}
