// Physically based bloom on the lit HDR layer, after Jorge Jimenez, "Next
// Generation Post Processing in Call of Duty: Advanced Warfare" (2014):
//
//   prefilter   full res -> mip 0: 13-tap downsample, keep only the light
//               above `threshold` (soft knee)
//   downsample  mip i -> mip i + 1: 13-tap filter, repeated
//   upsample    mip i + 1 -> mip i: 9-tap tent filter, added on top,
//               from the smallest mip back up to mip 0
//   combine     lit + intensity * mip 0
//
// The wide, smooth glow comes from summing all mip levels; the 13-tap and
// tent filters keep it free of blocky or flickering artifacts.
// Slides: https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/

#include "fullscreen.wgsl"

struct BloomParams {
    texel: vec2<f32>,           // 1 / size of the texture being read
    threshold: f32,             // radiance where bloom starts
    knee: f32,                  // soft transition width around the threshold
    intensity: f32,
    radius: f32,                // upsample filter radius in source texels
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: BloomParams;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var linear_clamp: sampler;
@group(0) @binding(3) var bloom: texture_2d<f32>;    // combine only

fn tap(uv: vec2<f32>, x: f32, y: f32) -> vec3<f32> {
    return textureSampleLevel(source, linear_clamp, uv + vec2<f32>(x, y) * params.texel, 0.0).rgb;
}

/// 13 bilinear taps in a 4x4-texel footprint, weighted as five overlapping
/// boxes (the CoD:AW downsample).
fn downsample13(uv: vec2<f32>) -> vec3<f32> {
    let a = tap(uv, -2.0, -2.0); let b = tap(uv, 0.0, -2.0); let c = tap(uv, 2.0, -2.0);
    let d = tap(uv, -1.0, -1.0); let e = tap(uv, 1.0, -1.0);
    let f = tap(uv, -2.0, 0.0);  let g = tap(uv, 0.0, 0.0);  let h = tap(uv, 2.0, 0.0);
    let i = tap(uv, -1.0, 1.0);  let j = tap(uv, 1.0, 1.0);
    let k = tap(uv, -2.0, 2.0);  let l = tap(uv, 0.0, 2.0);  let m = tap(uv, 2.0, 2.0);
    return (d + e + i + j) * 0.125
        + (a + b + g + f) * 0.03125 + (b + c + h + g) * 0.03125
        + (f + g + l + k) * 0.03125 + (g + h + m + l) * 0.03125;
}

@fragment
fn fs_prefilter(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let color = downsample13(in.uv);
    // Soft threshold: quadratic ramp within the knee, linear above it.
    let brightness = max(color.r, max(color.g, color.b));
    let soft = clamp(brightness - params.threshold + params.knee, 0.0, 2.0 * params.knee);
    let ramp = soft * soft / (4.0 * params.knee + 1e-5);
    let contribution = max(ramp, brightness - params.threshold) / max(brightness, 1e-5);
    return vec4<f32>(color * contribution, 1.0);
}

@fragment
fn fs_downsample(in: FullscreenVertex) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample13(in.uv), 1.0);
}

@fragment
fn fs_upsample(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let r = params.radius;
    let sum = tap(in.uv, -r, -r) + 2.0 * tap(in.uv, 0.0, -r) + tap(in.uv, r, -r)
        + 2.0 * tap(in.uv, -r, 0.0) + 4.0 * tap(in.uv, 0.0, 0.0) + 2.0 * tap(in.uv, r, 0.0)
        + tap(in.uv, -r, r) + 2.0 * tap(in.uv, 0.0, r) + tap(in.uv, r, r);
    return vec4<f32>(sum / 16.0, 1.0);  // added onto the larger mip by blending
}

@fragment
fn fs_combine(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let lit = textureLoad(source, vec2<i32>(floor(in.position.xy)), 0);
    let glow = textureSampleLevel(bloom, linear_clamp, in.uv, 0.0).rgb * params.intensity;
    // Glow outside meshes adds light without adding coverage.
    return vec4<f32>(lit.rgb + glow, lit.a);
}
