// Shadow lookups with percentage-closer soft shadows (PCSS, Randima
// Fernando 2005), so shadows are sharp where an object touches the surface
// and grow softer with distance, the way a light of finite size behaves.
//
// For one light and one surface point:
//
//   1. project the point into the light's shadow view (for point lights,
//      the cube face it lies in)
//   2. blocker search: average the depth of the casters found around it
//   3. penumbra width from similar triangles:
//        perspective   size * (receiver - blocker) / blocker
//        directional   angle * (receiver - blocker)
//   4. filter: compare depths at `samples` points spread over the
//      penumbra (hardware 2x2 comparison per tap) and average
//
// Taps lie on a Vogel (sunflower) disk rotated per point, which spreads the
// remaining noise evenly instead of producing banding.
//
// Needs `shadow_maps`, `shadow_sampler` and `shadow_views` declared by the
// including shader (deferred_lighting.wgsl).

// Mirrors SHADOW_VIEW_DTYPE in shadow_views.py.
struct ShadowView {
    view_projection: mat4x4<f32>,
    params: vec4<f32>,          // near, far, frustum width, perspective (1) / orthographic (0)
};

const MAX_SHADOW_SAMPLES: u32 = 64u;
const MAX_SEARCH_UV: f32 = 0.05;
const GOLDEN_ANGLE: f32 = 2.39996323;

/// 1 where `surface` sees `light` fully, 0 in full shadow.
fn shadow_visibility(light: Light, surface: Surface) -> f32 {
    if light.shadow_view < 0 || frame.shadow.w == 0.0 {
        return 1.0;
    }
    var layer = light.shadow_view;
    if light.kind == LIGHT_POINT {
        layer += i32(cube_face(surface.position - light.position));
    }
    let view = shadow_views[layer];
    let perspective = view.params.w > 0.5;
    let resolution = frame.shadow.z;

    // Normal offset: look the shadow up about one texel off the surface, so
    // the surface does not shadow itself ("shadow acne").
    let center = project(view, surface.position);
    let receiver = linear_depth(view, center.z);
    let texel_world = select(view.params.z, receiver * view.params.z, perspective) / resolution;
    let probe = project(view, surface.position + surface.normal * texel_world * 1.5);
    if any(probe.xy < vec2<f32>(0.0)) || any(probe.xy > vec2<f32>(1.0)) || probe.z > 1.0 {
        return 1.0;  // outside the shadow view: lit
    }

    let size = light.size * frame.shadow.x;
    let samples = clamp(u32(frame.shadow.y), 1u, MAX_SHADOW_SAMPLES);
    let rotation = 6.2831853 * hash(surface.position);
    let texel_uv = 1.0 / resolution;
    if size <= 0.0 {
        return filtered(layer, probe, texel_uv, samples, rotation);  // hard shadow
    }

    // Blocker search region: how far a penumbra could reach for blockers
    // halfway between the light and the receiver.
    let near = view.params.x;
    var search = size * (receiver - near) / view.params.z;
    if perspective {
        search = size / (receiver * view.params.z);
    }
    search = clamp(search, texel_uv, MAX_SEARCH_UV);

    var blocker_sum = 0.0;
    var blockers = 0.0;
    for (var i = 0u; i < samples; i++) {
        let uv = probe.xy + vogel(i, samples, rotation) * search;
        let texel = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (resolution - 1.0));
        let depth = linear_depth(view, textureLoad(shadow_maps, texel, layer, 0));
        if depth < receiver - texel_world {
            blocker_sum += depth;
            blockers += 1.0;
        }
    }
    if blockers == 0.0 {
        return 1.0;
    }
    let blocker = blocker_sum / blockers;

    var penumbra = size * (receiver - blocker) / view.params.z;  // directional
    if perspective {
        penumbra = size * (receiver - blocker) / blocker / (receiver * view.params.z);
    }
    return filtered(layer, probe, max(penumbra, texel_uv), samples, rotation);
}

/// Fraction of `samples` comparison taps within `radius` (uv) that are lit.
fn filtered(layer: i32, probe: vec3<f32>, radius: f32, samples: u32, rotation: f32) -> f32 {
    var lit = 0.0;
    for (var i = 0u; i < samples; i++) {
        let uv = probe.xy + vogel(i, samples, rotation) * radius;
        lit += textureSampleCompareLevel(shadow_maps, shadow_sampler, uv, layer, probe.z);
    }
    return lit / f32(samples);
}

/// Shadow-map uv (xy, top left origin) and depth (z) of a world point.
fn project(view: ShadowView, world: vec3<f32>) -> vec3<f32> {
    let clip = view.view_projection * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

/// Distance along the view direction for a depth-buffer value.
fn linear_depth(view: ShadowView, depth: f32) -> f32 {
    let near = view.params.x;
    let far = view.params.y;
    if view.params.w > 0.5 {
        return near * far / (far - depth * (far - near));
    }
    return near + depth * (far - near);
}

/// Point `index` of `count` on a unit Vogel disk, rotated by `rotation`.
fn vogel(index: u32, count: u32, rotation: f32) -> vec2<f32> {
    let radius = sqrt((f32(index) + 0.5) / f32(count));
    let angle = f32(index) * GOLDEN_ANGLE + rotation;
    return radius * vec2<f32>(cos(angle), sin(angle));
}

/// Cube face of a direction, in the order of shadow_views.py's _CUBE_FACES.
fn cube_face(direction: vec3<f32>) -> u32 {
    let a = abs(direction);
    if a.x >= a.y && a.x >= a.z {
        return select(1u, 0u, direction.x > 0.0);
    }
    if a.y >= a.z {
        return select(3u, 2u, direction.y > 0.0);
    }
    return select(5u, 4u, direction.z > 0.0);
}

fn hash(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}
