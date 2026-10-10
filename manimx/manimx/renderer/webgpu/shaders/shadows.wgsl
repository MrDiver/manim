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
// Taps lie on a Vogel (sunflower) disk in the receiving surface's plane,
// turned per pixel and sample by white noise (a PCG hash), which leaves no
// pattern once the samples and renders of a frame are averaged. The lighting pass
// sets `shadow_pixel` before shading.
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
/// PCF (previews): the radius of the soft edge in shadow-map texels, and
/// the most taps it takes.
const PCF_RADIUS: f32 = 1.5;
const PCF_TAPS: u32 = 8u;

/// The pixel being lit and its MSAA sample (in z), set by the lighting
/// pass: their noise turns the tap pattern.
var<private> shadow_pixel: vec3<u32>;

/// Where a world point is in the shadow maps: the layer (a point light's
/// cube face it falls in), that layer's view, its uv and depth there, and
/// whether the view covers it.
struct ShadowTap {
    layer: i32,
    view: ShadowView,
    at: vec3<f32>,
    inside: bool,
};

fn shadow_tap(light: Light, world: vec3<f32>) -> ShadowTap {
    var layer = light.shadow_view;
    if light.kind == LIGHT_POINT {
        layer += i32(cube_face(world - light.position));
    }
    let view = shadow_views[layer];
    let at = project(view, world);
    let inside = all(at.xy >= vec2<f32>(0.0)) && all(at.xy <= vec2<f32>(1.0)) && at.z <= 1.0;
    return ShadowTap(layer, view, at, inside);
}

/// 1 where `surface` sees `light` fully, 0 in full shadow.
///
/// Every tap is a point on a disk in the surface's plane, lifted 1.5
/// texels off it ("normal offset"), looked up in the shadow view it falls
/// in (across a cube seam, the next face) and compared with its own
/// depth. So a surface never shadows itself, whatever its angle to the
/// light, and soft edges continue across the faces of a point light.
fn shadow_visibility(light: Light, surface: Surface) -> f32 {
    if light.shadow_view < 0 || frame.shadow.w == 0.0 {
        return 1.0;
    }
    let center = shadow_tap(light, surface.position);
    let view = center.view;
    let perspective = view.params.w > 0.5;
    let resolution = frame.shadow.z;
    let receiver = linear_depth(view, center.at.z);
    // World size of one shadow-map texel at the surface.
    let texel_world = select(view.params.z, receiver * view.params.z, perspective) / resolution;
    let world_per_uv = texel_world * resolution;

    let normal = surface.normal;
    let origin = surface.position + normal * texel_world * 1.5;
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.x) < 0.9);
    let u = normalize(cross(normal, helper));
    let v = cross(normal, u);

    let size = light.size * frame.shadow.x;
    // Taps per lookup: fewer when the frame averages several renders, each
    // with its pattern turned on (RenderConfig.shadow_filter).
    let samples = clamp(u32(frame.shadow_filter.y), 1u, MAX_SHADOW_SAMPLES);
    let rotation = 6.2831853 * (pixel_noise(shadow_pixel) + frame.shadow_filter.z);
    if frame.shadow_filter.x < 0.5 {
        // PCF: a fixed soft edge PCF_RADIUS texels wide (times the softness).
        return filtered(light, origin, u, v, texel_world * PCF_RADIUS * frame.shadow.x, min(samples, PCF_TAPS), rotation);
    }
    if size <= 0.0 {
        return filtered(light, origin, u, v, texel_world, samples, rotation);  // hard shadow
    }

    // Blocker search region: how far a penumbra could reach for blockers
    // halfway between the light and the receiver (in uv, then on the plane).
    let near = view.params.x;
    var search = size * (receiver - near) / view.params.z;
    if perspective {
        search = size / (receiver * view.params.z);
    }
    search = clamp(search, 1.0 / resolution, MAX_SEARCH_UV) * world_per_uv;

    var blocker_sum = 0.0;
    var blockers = 0.0;
    for (var i = 0u; i < samples; i++) {
        let offset = vogel(i, samples, rotation) * search;
        let tap = shadow_tap(light, origin + u * offset.x + v * offset.y);
        if !tap.inside {
            continue;
        }
        let texel = vec2<i32>(tap.at.xy * (resolution - 1.0));
        let depth = linear_depth(tap.view, textureLoad(shadow_maps, texel, tap.layer, 0));
        if depth < linear_depth(tap.view, tap.at.z) - texel_world {
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
    return filtered(light, origin, u, v, max(penumbra, 1.0 / resolution) * world_per_uv, samples, rotation);
}

/// Fraction of `samples` comparison taps on the disk of `radius` (world
/// units) around `origin` in the plane of `u` and `v` that are lit.
fn filtered(light: Light, origin: vec3<f32>, u: vec3<f32>, v: vec3<f32>, radius: f32, samples: u32, rotation: f32) -> f32 {
    var lit = 0.0;
    for (var i = 0u; i < samples; i++) {
        let offset = vogel(i, samples, rotation) * radius;
        let tap = shadow_tap(light, origin + u * offset.x + v * offset.y);
        if !tap.inside {
            lit += 1.0;  // outside the shadow view: lit
            continue;
        }
        lit += textureSampleCompareLevel(shadow_maps, shadow_sampler, tap.at.xy, tap.layer, tap.at.z);
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

/// A turn in [0, 1) for a pixel and sample: the PCG hash (Jarzynski and
/// Olano 2020) of their coordinates, without any spatial pattern.
fn pixel_noise(pixel: vec3<u32>) -> f32 {
    var state = pixel.x * 1664525u + pixel.y * 22695477u + pixel.z * 1013904223u;
    state = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967296.0;
}
