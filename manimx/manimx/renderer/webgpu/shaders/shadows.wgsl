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
/// PCF (previews): the radius of the soft edge in shadow-map texels, and
/// the most taps it takes.
const PCF_RADIUS: f32 = 1.5;
const PCF_TAPS: u32 = 8u;

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

    // The receiver's plane in the shadow map: its depth at a tap is the
    // probe's plus the slope times the tap's offset. A surface seen at a
    // grazing angle from the light would otherwise shadow itself, its own
    // depth at the taps being nearer than at the probe.
    let slope = receiver_slope(view, surface, probe, texel_world);

    let size = light.size * frame.shadow.x;
    // Taps per lookup: fewer when the frame averages several renders, each
    // with its pattern turned on (RenderConfig.shadow_filter).
    let samples = clamp(u32(frame.shadow_filter.y), 1u, MAX_SHADOW_SAMPLES);
    let rotation = 6.2831853 * (hash(surface.position) + frame.shadow_filter.z);
    let texel_uv = 1.0 / resolution;
    if frame.shadow_filter.x < 0.5 {
        // PCF: a fixed soft edge PCF_RADIUS texels wide (times the softness).
        return filtered(layer, probe, slope, texel_uv * PCF_RADIUS * frame.shadow.x, min(samples, PCF_TAPS), rotation);
    }
    if size <= 0.0 {
        return filtered(layer, probe, slope, texel_uv, samples, rotation);  // hard shadow
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
        let offset = vogel(i, samples, rotation) * search;
        let uv = probe.xy + offset;
        let texel = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (resolution - 1.0));
        let depth = linear_depth(view, textureLoad(shadow_maps, texel, layer, 0));
        let receiver_here = linear_depth(view, probe.z + dot(slope, offset));
        if depth < receiver_here - texel_world {
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
    return filtered(layer, probe, slope, max(penumbra, texel_uv), samples, rotation);
}

/// How the depth of the plane through `probe` parallel to the surface
/// changes per shadow-map uv (receiver plane depth bias): from the
/// projections of two steps along the surface, a texel long.
fn receiver_slope(view: ShadowView, surface: Surface, probe: vec3<f32>, step: f32) -> vec2<f32> {
    let n = surface.normal;
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) < 0.9);
    let t1 = normalize(cross(n, helper));
    let t2 = cross(n, t1);
    let origin = surface.position + n * step * 1.5;  // the probe's world point
    let d1 = project(view, origin + t1 * step) - probe;
    let d2 = project(view, origin + t2 * step) - probe;
    let det = d1.x * d2.y - d1.y * d2.x;
    if abs(det) < 1e-12 {
        return vec2<f32>(0.0);  // seen edge-on: no plane to follow
    }
    return vec2<f32>(d1.z * d2.y - d1.y * d2.z, d1.x * d2.z - d1.z * d2.x) / det;
}

/// Fraction of `samples` comparison taps within `radius` (uv) that are lit.
fn filtered(layer: i32, probe: vec3<f32>, slope: vec2<f32>, radius: f32, samples: u32, rotation: f32) -> f32 {
    var lit = 0.0;
    for (var i = 0u; i < samples; i++) {
        let offset = vogel(i, samples, rotation) * radius;
        let uv = probe.xy + offset;
        lit += textureSampleCompareLevel(shadow_maps, shadow_sampler, uv, layer, probe.z + dot(slope, offset));
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
