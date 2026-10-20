// Depth of field, step 2 of 3: the largest blur radius in each tile of
// `params.tile` pixels squared, so the blur (depth_of_field.wgsl) knows how
// far it must gather. A tile is at least as large as the largest blur, so
// whatever reaches a pixel lies in its tile or the eight around.

#include "fullscreen.wgsl"
#include "frame.wgsl"
#include "lens.wgsl"

@group(0) @binding(1) var<uniform> params: DepthOfFieldParams;
@group(0) @binding(2) var distances: texture_2d<f32>;

@fragment
fn fs_blur_tiles(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let tile = i32(params.tile);
    let size = vec2<i32>(textureDimensions(distances));
    let first = vec2<i32>(floor(in.position.xy)) * tile;
    let last = min(first + tile, size);
    var largest = 0.0;
    for (var y = first.y; y < last.y; y++) {
        for (var x = first.x; x < last.x; x++) {
            largest = max(largest, blur_radius(textureLoad(distances, vec2<i32>(x, y), 0).r));
        }
    }
    return vec4<f32>(largest);
}
