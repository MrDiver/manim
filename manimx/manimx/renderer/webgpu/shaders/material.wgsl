// Material evaluation: glTF metallic-roughness material + textures -> Surface.
//
// This file knows nothing about lights or G-buffers: the G-buffer pass
// (mesh.wgsl) evaluates the Surface here and stores it with gbuffer.wgsl;
// the lighting pass decodes it and shades it with lighting.wgsl.

#include "surface.wgsl"

// Mirrors MATERIAL_DTYPE in mesh_batch.py.
struct Material {
    base_color: vec4<f32>,      // linear RGBA factor
    emissive: vec4<f32>,        // linear RGB factor (strength applied), w unused
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    occlusion_strength: f32,
};


// Group 1: the textures of one material. Slots without a texture get a
// neutral 1x1 default (white, or a flat normal), so no branching is needed.
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;           // sRGB
@group(1) @binding(1) var metallic_roughness_texture: texture_2d<f32>;   // G rough, B metal
@group(1) @binding(2) var normal_texture: texture_2d<f32>;
@group(1) @binding(3) var occlusion_texture: texture_2d<f32>;            // R
@group(1) @binding(4) var emissive_texture: texture_2d<f32>;             // sRGB
@group(1) @binding(5) var material_sampler: sampler;

struct SurfaceInput {
    position: vec3<f32>,
    normal: vec3<f32>,          // interpolated, not normalized
    tangent: vec4<f32>,         // xyz world tangent, w handedness
    uv: vec2<f32>,
    front_facing: bool,
};

fn evaluate_surface(material: Material, in: SurfaceInput) -> Surface {
    let base = textureSample(base_color_texture, material_sampler, in.uv) * material.base_color;
    let metal_rough = textureSample(metallic_roughness_texture, material_sampler, in.uv);
    let occlusion = textureSample(occlusion_texture, material_sampler, in.uv).r;
    let emissive = textureSample(emissive_texture, material_sampler, in.uv).rgb * material.emissive.rgb;

    var surface: Surface;
    surface.position = in.position;
    surface.normal = shading_normal(material, in);
    surface.base_color = base.rgb;
    surface.alpha = base.a;
    surface.metallic = clamp(material.metallic * metal_rough.b, 0.0, 1.0);
    surface.roughness = clamp(material.roughness * metal_rough.g, 0.03, 1.0);
    surface.occlusion = mix(1.0, occlusion, material.occlusion_strength);
    surface.emissive = emissive;
    return surface;
}

/// Geometric normal perturbed by the normal map, flipped for back faces so
/// that every mesh is effectively double sided.
fn shading_normal(material: Material, in: SurfaceInput) -> vec3<f32> {
    let side = select(-1.0, 1.0, in.front_facing);
    let normal = normalize(in.normal) * side;
    let sampled = textureSample(normal_texture, material_sampler, in.uv).xyz * 2.0 - 1.0;
    let tangent_length = length(in.tangent.xyz);
    if tangent_length < 1e-6 {
        return normal;  // no tangents: the normal map cannot be applied
    }
    let tangent = normalize(in.tangent.xyz - normal * dot(normal, in.tangent.xyz));
    let bitangent = cross(normal, tangent) * in.tangent.w;
    let local = vec3<f32>(sampled.xy * material.normal_scale, sampled.z);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}
