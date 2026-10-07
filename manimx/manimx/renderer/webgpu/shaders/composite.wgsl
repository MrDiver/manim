// Composite one rendered sub-frame and add it to the frame accumulator.
//
//   1. tone map the lit 3D layer (linear HDR, premultiplied by coverage)
//      and put it over the background, in linear light
//   2. encode to sRGB and put the vector layer over it in sRGB values,
//      which is how Manim blends, so 2D colors come out exactly
//   3. decode back to linear and add `weight` times the result to the
//      accumulator (premultiplied), so motion-blur sub-frames average in
//      linear light; finalize.wgsl encodes the sum at the end

#include "fullscreen.wgsl"
#include "color.wgsl"

struct CompositeParams {
    background: vec4<f32>,      // sRGB values, straight alpha
    tone_map: u32,              // TONE_MAP_* operator
    weight: f32,                // this sub-frame's share of the frame
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: CompositeParams;
@group(0) @binding(1) var lit_layer: texture_2d<f32>;
@group(0) @binding(2) var vector_layer: texture_2d<f32>;

@fragment
fn fs_composite(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(floor(in.position.xy));
    let lit = textureLoad(lit_layer, pixel, 0);
    let vector = textureLoad(vector_layer, pixel, 0);

    // 3D over background, linear light, premultiplied.
    let background = params.background;
    let under_alpha = lit.a + background.a * (1.0 - lit.a);
    let under_linear = tone_map(lit.rgb, params.tone_map)
        + srgb_to_linear(background.rgb) * background.a * (1.0 - lit.a);

    // 2D over 3D, sRGB values, premultiplied.
    var under_srgb = vec3<f32>(0.0);
    if under_alpha > 0.0 {
        under_srgb = linear_to_srgb(under_linear / under_alpha) * under_alpha;
    }
    let alpha = vector.a + (1.0 - vector.a) * under_alpha;
    let srgb = vector.rgb + (1.0 - vector.a) * under_srgb;

    var linear = vec3<f32>(0.0);
    if alpha > 0.0 {
        linear = srgb_to_linear(srgb / alpha) * alpha;
    }
    return vec4<f32>(linear, alpha) * params.weight;
}
