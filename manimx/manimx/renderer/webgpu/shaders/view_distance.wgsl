// Distance from the camera along its view axis of every pixel, for depth
// of field (depth_of_field.wgsl): the nearest of the meshes' and the
// shapes' depth samples (so a pixel an edge only partly covers is as far
// as that edge); FAR for the background. Resolving the multisampled depth
// buffers once here saves the blur reading them again for every tap.

#include "fullscreen.wgsl"
#include "frame.wgsl"

@group(0) @binding(1) var mesh_depth: texture_depth_multisampled_2d;
@group(0) @binding(2) var vector_depth: texture_depth_multisampled_2d;

/// Samples per pixel of the depth buffers (the frame's MSAA; the renderer
/// replaces this line, and the texture type when there is only one).
const DEPTH_SAMPLES: i32 = 4;
const FAR: f32 = 1e9;

@fragment
fn fs_view_distance(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(floor(in.position.xy));
    var depth = 1.0;
    for (var index = 0; index < DEPTH_SAMPLES; index++) {
        depth = min(depth, min(textureLoad(mesh_depth, pixel, index), textureLoad(vector_depth, pixel, index)));
    }
    if depth >= 1.0 {
        return vec4<f32>(FAR);
    }
    let position = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let ndc = vec4<f32>(position.x * 2.0 - 1.0, 1.0 - position.y * 2.0, depth, 1.0);
    let view = frame.inverse_projection * ndc;
    return vec4<f32>(-view.z / view.w);
}
