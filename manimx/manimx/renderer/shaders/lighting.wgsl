// Lighting: Surface + lights -> linear radiance towards the viewer.
//
// The BRDF is the one from the glTF 2.0 specification (appendix B,
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation):
// Lambert diffuse plus GGX / Smith-correlated visibility / Schlick Fresnel
// specular. Light kinds and falloffs follow KHR_lights_punctual,
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual

const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_POINT: u32 = 1u;
const LIGHT_SPOT: u32 = 2u;

// Mirrors `Light` in core/src/packing/records.rs.
struct Light {
    position: vec3<f32>,        // world (point, spot)
    kind: u32,
    direction: vec3<f32>,       // world, unit, the way the light travels
    range: f32,                 // 0: unlimited
    color: vec3<f32>,           // linear RGB times intensity
    inner_cone_cos: f32,        // spot only
    outer_cone_cos: f32,        // spot only
    shadow_view: i32,           // first shadow map layer, or -1 without shadows
    size: f32,                  // radius, or angular diameter for directional lights
    scope: i32,                 // clip scope the light is limited to, or -1
};

#include "shadows.wgsl"

const PI_LIGHTING: f32 = 3.14159265358979;

/// Radiance leaving `surface` towards `view` (unit vector to the viewer).
/// `scope` is the surface's clip scope: scoped lights only reach surfaces
/// inside their scope (needs `scope_parents` from the including shader).
fn shade_surface(surface: Surface, view: vec3<f32>, scope: i32) -> vec3<f32> {
    var color = surface.emissive;
    // Uniform ambient light, darkened by ambient occlusion. Without an
    // environment map, metals would only show their highlights and look
    // black, so the ambient light is also reflected with the specular color
    // f0 (the reflectance at normal incidence). Image-based lighting would
    // replace this term.
    let diffuse_color = surface.base_color * (1.0 - surface.metallic);
    let f0 = mix(vec3<f32>(0.04), surface.base_color, surface.metallic);
    color += frame.ambient.rgb * (diffuse_color + f0) * surface.occlusion;
    for (var i = 0u; i < frame.light_count; i++) {
        if light_reaches(lights[i].scope, scope) {
            color += light_contribution(lights[i], surface, view);
        }
    }
    return color;
}

/// Whether a light limited to `light_scope` reaches a surface in `scope`:
/// the surface's scope must be the light's or nested inside it.
fn light_reaches(light_scope: i32, scope: i32) -> bool {
    if light_scope < 0 {
        return true;  // unscoped lights reach everything
    }
    var current = scope;
    for (var depth = 0; depth < 64 && current >= 0; depth++) {
        if current == light_scope {
            return true;
        }
        current = scope_parents[current];
    }
    return false;
}

fn light_contribution(light: Light, surface: Surface, view: vec3<f32>) -> vec3<f32> {
    var to_light: vec3<f32>;
    var attenuation = 1.0;
    if light.kind == LIGHT_DIRECTIONAL {
        to_light = -light.direction;
    } else {
        let offset = light.position - surface.position;
        let distance = length(offset);
        to_light = offset / max(distance, 1e-6);
        attenuation = range_falloff(distance, light.range) / max(distance * distance, 1e-4);
        if light.kind == LIGHT_SPOT {
            let cosine = dot(light.direction, -to_light);
            attenuation *= smoothstep(light.outer_cone_cos, light.inner_cone_cos, cosine);
        }
    }
    let n_dot_l = dot(surface.normal, to_light);
    if n_dot_l <= 0.0 || attenuation <= 0.0 {
        return vec3<f32>(0.0);
    }
    let visibility = shadow_visibility(light, surface);
    if visibility <= 0.0 {
        return vec3<f32>(0.0);
    }
    return brdf(surface, to_light, view) * light.color * attenuation * n_dot_l * visibility;
}

/// KHR_lights_punctual's recommended smooth cut-off at `range`.
fn range_falloff(distance: f32, range: f32) -> f32 {
    if range <= 0.0 {
        return 1.0;
    }
    let ratio = distance / range;
    return clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
}

fn brdf(surface: Surface, to_light: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let n = surface.normal;
    let h = normalize(to_light + view);
    let n_dot_l = clamp(dot(n, to_light), 0.0, 1.0);
    let n_dot_v = clamp(abs(dot(n, view)), 1e-4, 1.0);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(view, h), 0.0, 1.0);

    let alpha = surface.roughness * surface.roughness;
    let f0 = mix(vec3<f32>(0.04), surface.base_color, surface.metallic);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);

    let alpha2 = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let distribution = alpha2 / (PI_LIGHTING * denominator * denominator);
    let visibility = 0.5 / max(
        n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2)
            + n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2),
        1e-6,
    );

    let diffuse = (1.0 - fresnel) * (1.0 - surface.metallic) * surface.base_color / PI_LIGHTING;
    let specular = fresnel * distribution * visibility;
    return diffuse + specular;
}
