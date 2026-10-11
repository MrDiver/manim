// Depth of field: one view's composited image blurred by how far each
// pixel is from the camera's focus (Camera.focus_distance, .aperture).
//
// The blur radius of a pixel (its circle of confusion) follows a thin
// lens: aperture * |distance - focus| / distance, as a fraction of the
// image height; so `aperture` is the blur of something infinitely far
// away (Potmesil and Chakravarty 1981, "A lens and aperture camera model
// for synthetic image generation"; derived in
// https://pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models#TheThinLensModelandDepthofField).
// Distances come from view_distance.wgsl, rendered before.
//
// The blur gathers pixels in a disk (golden-angle spiral) as large as the
// largest blur that can reach the pixel: the largest in its tile and the
// tiles around (depth_of_field_tiles.wgsl), so sharp regions far from
// blurred ones gather little. The taps are as dense as SAMPLES over the
// largest blur in a frame, the spiral turned by a different angle
// at every pixel (interleaved gradient noise), so too few samples for a
// large blur give fine grain instead of streaks. A sample counts where its own blur
// reaches this pixel ("scatter as gather"); one behind this pixel counts
// only as far as this pixel's blur reaches, so a sharp object never gets
// the blurred background spread over it, while a blurred one in front
// spreads over sharp things behind it.
//
// Scatter as gather, the tiles and interleaved gradient noise follow Jorge
// Jimenez, "Next Generation Post Processing in Call of Duty: Advanced
// Warfare" (SIGGRAPH 2014),
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
// The spiral is Vogel's (1979), https://doi.org/10.1016/0025-5564(79)90080-4

#include "fullscreen.wgsl"
#include "frame.wgsl"
#include "lens.wgsl"

@group(0) @binding(1) var<uniform> params: DepthOfFieldParams;
@group(0) @binding(2) var image: texture_2d<f32>;
@group(0) @binding(3) var distances: texture_2d<f32>;
@group(0) @binding(4) var tiles: texture_2d<f32>;

const SAMPLES: u32 = 64u;
const GOLDEN_ANGLE: f32 = 2.39996323;

/// Distance from the camera along its view axis at `pixel`.
fn distance_at(pixel: vec2<i32>) -> f32 {
    return textureLoad(distances, pixel, 0).r;
}

/// The largest blur that can reach `pixel`: the largest of its tile and
/// the eight around.
fn reach_around(pixel: vec2<i32>) -> f32 {
    let tile = pixel / i32(params.tile);
    let last = vec2<i32>(textureDimensions(tiles)) - 1;
    var largest = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            largest = max(largest, textureLoad(tiles, clamp(tile + vec2<i32>(x, y), vec2<i32>(0), last), 0).r);
        }
    }
    return largest;
}

@fragment
fn fs_depth_of_field(in: FullscreenVertex) -> @location(0) vec4<f32> {
    let size = vec2<i32>(frame.viewport.xy);
    let pixel = vec2<i32>(floor(in.position.xy));
    let distance = distance_at(pixel);
    let radius = blur_radius(distance);
    let largest = min(params.aperture, MAX_RADIUS) * frame.viewport.y;
    let gather = reach_around(pixel);
    var sum = textureLoad(image, pixel, 0);
    if gather < 0.5 {
        return sum * params.weight;  // nothing blurred reaches this pixel
    }
    // As many taps as keep them as dense as SAMPLES over the largest blur.
    let ratio = gather / max(largest, 1e-6);
    let samples = clamp(u32(ceil(f32(SAMPLES) * ratio * ratio)), 1u, SAMPLES);

    // Jimenez's interleaved gradient noise: a fixed angle per pixel.
    let turn = 6.2831853 * fract(52.9829189 * fract(dot(vec2<f32>(pixel), vec2<f32>(0.06711056, 0.00583715))));
    var total = 1.0;
    for (var index = 0u; index < samples; index++) {
        let offset_length = gather * sqrt((f32(index) + 0.5) / f32(samples));
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
