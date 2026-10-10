// Depth of field: one view's composited image blurred by how far each
// pixel is from the camera's focus (Camera.focus_distance, .aperture).
//
// The blur radius of a pixel (its circle of confusion) follows a thin
// lens: aperture * |distance - focus| / distance, as a fraction of the
// image height; so `aperture` is the blur of something infinitely far
// away. Distances come from view_distance.wgsl, rendered before.
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
@group(0) @binding(3) var distances: texture_2d<f32>;

const SAMPLES: u32 = 64u;
/// The largest blur radius, as a fraction of the image height.
const MAX_RADIUS: f32 = 0.05;
const GOLDEN_ANGLE: f32 = 2.39996323;

/// Distance from the camera along its view axis at `pixel`.
fn distance_at(pixel: vec2<i32>) -> f32 {
    return textureLoad(distances, pixel, 0).r;
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
