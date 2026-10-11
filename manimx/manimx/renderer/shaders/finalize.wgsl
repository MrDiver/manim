// Turn the accumulated linear frame into the output image: straight alpha,
// sRGB values.

#include "fullscreen.wgsl"
#include "color.wgsl"

@group(0) @binding(0) var accumulated: texture_2d<f32>;

@fragment
fn fs_finalize(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let color = textureLoad(accumulated, vec2<i32>(floor(in.position.xy)), 0);
    if color.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(linear_to_srgb(color.rgb / color.a), min(color.a, 1.0));
}
