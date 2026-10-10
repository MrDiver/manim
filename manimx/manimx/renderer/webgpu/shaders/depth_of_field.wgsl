// Depth of field: one view's composited image blurred by how far each
// pixel is from the camera's focus (Camera.focus_distance, .aperture).
//
// The blur radius of a pixel (its circle of confusion) follows a thin
// lens: aperture * |distance - focus| / distance, as a fraction of the
// image height; so `aperture` is the blur of something infinitely far
// away. Distances come from the meshes' and the shapes' depth buffers
// (the nearest of all their samples, so a pixel an edge only partly
// covers is as far as that edge); the background is infinitely far.
//
// The blur gathers SAMPLES pixels in a disk (golden-angle spiral) as large
// as the largest blur in the image, the spiral turned by a different angle
// at every pixel (interleaved gradient noise), so too few samples for a
// large blur give fine grain instead of streaks. A sample counts where its own blur
// reaches this pixel ("scatter as gather"); one behind this pixel counts
// only as far as this pixel's blur reaches, so a sharp object never gets
// the blurred background spread over it, while a blurred one in front
// spreads over sharp things behind it.

#include "fullscreen.wgsl"
#include "frame.wgsl"

struct DepthOfFieldParams {
    focus_distance: f32,        // world units along the view axis
    aperture: f32,              // blur at infinity, fraction of the image height
    weight: f32,                // this view's share of the target
    _pad: f32,
};

@group(0) @binding(1) var<uniform> params: DepthOfFieldParams;
@group(0) @binding(2) var image: texture_2d<f32>;
@group(0) @binding(3) var mesh_depth: texture_depth_multisampled_2d;
@group(0) @binding(4) var vector_depth: texture_depth_multisampled_2d;

const SAMPLES: u32 = 64u;
/// Samples per pixel of the depth buffers (the frame's MSAA; the renderer
/// replaces this line, and the texture type when there is only one).
const DEPTH_SAMPLES: i32 = 4;
/// The largest blur radius, as a fraction of the image height.
const MAX_RADIUS: f32 = 0.05;
const GOLDEN_ANGLE: f32 = 2.39996323;
const FAR: f32 = 1e9;

/// Distance from the camera along its view axis at `pixel`.
fn distance_at(pixel: vec2<i32>) -> f32 {
    var depth = 1.0;
    for (var index = 0; index < DEPTH_SAMPLES; index++) {
        depth = min(depth, min(textureLoad(mesh_depth, pixel, index), textureLoad(vector_depth, pixel, index)));
    }
    if depth >= 1.0 {
        return FAR;
    }
    let position = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let ndc = vec4<f32>(position.x * 2.0 - 1.0, 1.0 - position.y * 2.0, depth, 1.0);
    let view = frame.inverse_projection * ndc;
    return -view.z / view.w;
}

/// The blur radius in pixels of something at `distance`.
fn blur_radius(distance: f32) -> f32 {
    let defocus = abs(distance - params.focus_distance) / max(distance, 1e-6);
    return min(params.aperture * defocus, MAX_RADIUS) * frame.viewport.y;
}

@fragment
fn fs_depth_of_field(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let size = vec2<i32>(frame.viewport.xy);
    let pixel = vec2<i32>(floor(in.position.xy));
    let distance = distance_at(pixel);
    let radius = blur_radius(distance);
    let largest = min(params.aperture, MAX_RADIUS) * frame.viewport.y;

    // Jimenez's interleaved gradient noise: a fixed angle per pixel.
    let turn = 6.2831853 * fract(52.9829189 * fract(dot(vec2<f32>(pixel), vec2<f32>(0.06711056, 0.00583715))));
    var sum = textureLoad(image, pixel, 0);
    var total = 1.0;
    for (var index = 0u; index < SAMPLES; index++) {
        let offset_length = largest * sqrt((f32(index) + 0.5) / f32(SAMPLES));
        let angle = f32(index) * GOLDEN_ANGLE + turn;
        let offset = vec2<f32>(cos(angle), sin(angle)) * offset_length;
        let sample = clamp(pixel + vec2<i32>(round(offset)), vec2<i32>(0), size - 1);
        let sample_distance = distance_at(sample);
        var reach = blur_radius(sample_distance);
        if sample_distance > distance {
            reach = min(reach, radius);  // behind this pixel: only as far as its own blur
        }
        let weight = clamp(reach - offset_length + 1.0, 0.0, 1.0);
        sum += textureLoad(image, sample, 0) * weight;
        total += weight;
    }
    return sum / total * params.weight;
}
