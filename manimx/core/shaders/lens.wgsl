// The lens of depth of field: what depth_of_field.wgsl and
// depth_of_field_tiles.wgsl share. Needs `frame` and `params` declared by
// the including shader.

struct DepthOfFieldParams {
    focus_distance: f32,        // world units along the view axis
    aperture: f32,              // blur at infinity, fraction of the image height
    weight: f32,                // this view's share of the target
    tile: f32,                  // pixels per side of a tile of the blur tiles
};

/// The largest blur radius, as a fraction of the image height (the
/// renderer sizes the tiles by it).
const MAX_RADIUS: f32 = 0.05;

/// The blur radius in pixels of something at `distance`.
fn blur_radius(distance: f32) -> f32 {
    let defocus = abs(distance - params.focus_distance) / max(distance, 1e-6);
    return min(params.aperture * defocus, MAX_RADIUS) * frame.viewport.y;
}
