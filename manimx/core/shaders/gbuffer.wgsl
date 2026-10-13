// G-buffer layout: how a Surface is stored between the geometry pass
// (mesh.wgsl) and the lighting pass (deferred_lighting.wgsl).
//
//   target 0  rgba8unorm-srgb  base color (linear in the shader), occlusion
//   target 1  rgba16float      world normal xyz, perceptual roughness
//   target 2  rgba16float      emissive radiance rgb, metallic
//   target 3  r16float         clip scope of the object (-1: none), which
//                              decides which scoped lights reach it
//   depth     depth32float     reconstructs the world position
//
// All targets are multisampled like the frame, and the lighting pass
// shades every sample, so edges keep their MSAA antialiasing.

#include "surface.wgsl"

struct GBuffer {
    @location(0) albedo_occlusion: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
    @location(2) emissive_metallic: vec4<f32>,
    @location(3) scope: f32,
};

fn encode_gbuffer(surface: Surface, scope: i32) -> GBuffer {
    var out: GBuffer;
    out.scope = f32(scope);  // small integers are exact in a half float
    out.albedo_occlusion = vec4<f32>(surface.base_color, surface.occlusion);
    out.normal_roughness = vec4<f32>(surface.normal, surface.roughness);
    out.emissive_metallic = vec4<f32>(surface.emissive, surface.metallic);
    return out;
}

fn decode_gbuffer(
    albedo_occlusion: vec4<f32>,
    normal_roughness: vec4<f32>,
    emissive_metallic: vec4<f32>,
    position: vec3<f32>,
) -> Surface {
    var surface: Surface;
    surface.position = position;
    surface.normal = normalize(normal_roughness.xyz);
    surface.base_color = albedo_occlusion.rgb;
    surface.alpha = 1.0;
    surface.metallic = emissive_metallic.a;
    surface.roughness = normal_roughness.a;
    surface.occlusion = albedo_occlusion.a;
    surface.emissive = emissive_metallic.rgb;
    return surface;
}
