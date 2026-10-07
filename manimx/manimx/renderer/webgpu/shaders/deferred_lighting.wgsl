// Deferred lighting pass: lights every G-buffer sample.
//
// A fullscreen triangle covers the frame. Reading `sample_index` makes the
// fragment shader run once per MSAA sample, and each invocation writes its
// own sample of the multisampled HDR target, so lit edges are antialiased
// exactly like with forward shading.
//
// Output: linear HDR radiance with alpha = 1 where a mesh covers the
// sample and 0 elsewhere. After the MSAA resolve this becomes a
// premultiplied "lit layer" whose alpha is the mesh coverage.

#include "frame.wgsl"
#include "fullscreen.wgsl"
#include "gbuffer.wgsl"
#include "lighting.wgsl"

@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var albedo_occlusion_samples: texture_multisampled_2d<f32>;
@group(0) @binding(3) var normal_roughness_samples: texture_multisampled_2d<f32>;
@group(0) @binding(4) var emissive_metallic_samples: texture_multisampled_2d<f32>;
@group(0) @binding(5) var depth_samples: texture_depth_multisampled_2d;
@group(0) @binding(6) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(7) var shadow_sampler: sampler_comparison;
@group(0) @binding(8) var<storage, read> shadow_views: array<ShadowView>;
@group(0) @binding(9) var scope_samples: texture_multisampled_2d<f32>;
@group(0) @binding(10) var<storage, read> scope_parents: array<i32>;


@fragment
fn fs_light(@builtin(position) position: vec4<f32>, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(floor(position.xy));
    let depth = textureLoad(depth_samples, pixel, i32(sample));
    if depth >= 1.0 {
        return vec4<f32>(0.0);  // no mesh here: transparent
    }
    let world = world_position(position.xy, depth);
    let surface = decode_gbuffer(
        textureLoad(albedo_occlusion_samples, pixel, i32(sample)),
        textureLoad(normal_roughness_samples, pixel, i32(sample)),
        textureLoad(emissive_metallic_samples, pixel, i32(sample)),
        world,
    );
    let scope = i32(round(textureLoad(scope_samples, pixel, i32(sample)).r));
    let radiance = shade_surface(surface, view_direction(world), scope) * frame.exposure;
    return vec4<f32>(radiance, 1.0);
}

/// World position of a fragment from its window coordinates and depth.
fn world_position(window: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(window.x * frame.viewport.z * 2.0 - 1.0, 1.0 - window.y * frame.viewport.w * 2.0);
    let view = frame.inverse_projection * vec4<f32>(ndc, depth, 1.0);
    return (frame.inverse_view * vec4<f32>(view.xyz / view.w, 1.0)).xyz;
}

/// Unit vector from a world point towards the viewer.
fn view_direction(point: vec3<f32>) -> vec3<f32> {
    if frame.projection[3][3] == 1.0 {  // orthographic
        return normalize((frame.inverse_view * vec4<f32>(0.0, 0.0, 1.0, 0.0)).xyz);
    }
    return normalize(frame.camera_position.xyz - point);
}
