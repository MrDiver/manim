// Deferred lighting pass: lights the G-buffer.
//
// Most pixels hold one surface in all their MSAA samples (the geometry
// pass ran once for them and wrote the same G-buffer values to every
// sample); only the pixels on an edge differ. With MSAA, three fullscreen
// triangles light the frame:
//
//   1. `fs_mark_edges` (per pixel) marks the edge pixels in the stencil
//      buffer; it writes no color.
//   2. `fs_light_pixel` (per pixel, where the stencil is unmarked) shades
//      a pixel once, where its first sample lies, and writes all its
//      samples.
//   3. `fs_light_sample` (per sample, where the stencil is marked) reads
//      `sample_index`, so it runs once per sample of an edge pixel and
//      writes each its own: lit edges keep their MSAA antialiasing.
//
// Shading edge pixels per sample and the rest once follows Andrew
// Lauritzen, "Deferred Rendering for Current and Future Rendering
// Pipelines" (SIGGRAPH 2010),
// https://www.intel.com/content/www/us/en/developer/articles/technical/deferred-rendering-for-current-and-future-rendering-pipelines.html
//
// Without MSAA, `fs_light_pixel` alone shades every pixel.
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


/// Samples per pixel of the G-buffer (the renderer replaces this line).
const GBUFFER_SAMPLES: i32 = 4;
/// Where in its pixel the first sample lies: WebGPU's fixed pattern for
/// 4 samples (the renderer replaces this line without MSAA: the center).
const FIRST_SAMPLE: vec2<f32> = vec2<f32>(0.375, 0.125);
/// Two samples are the same surface if their distances from the camera
/// differ by less than this fraction, their normals (and roughness) by
/// less than SAME_SURFACE_NORMAL and their colors by less than
/// SAME_SURFACE_VALUE: a smooth mesh's triangles meet with slightly
/// different interpolated values, and shading those once looks the same.
const SAME_SURFACE_DEPTH: f32 = 0.01;
const SAME_SURFACE_NORMAL: f32 = 0.02;
const SAME_SURFACE_VALUE: f32 = 0.004;

@fragment
fn fs_mark_edges(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    if one_surface(vec2<i32>(floor(position.xy))) {
        discard;  // no edge: the stencil stays unmarked
    }
    return vec4<f32>(0.0);  // the pipeline writes no color
}

@fragment
fn fs_light_pixel(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Shaded where its first sample lies, so the point is on the surface.
    let pixel = vec2<i32>(floor(position.xy));
    let window = vec2<f32>(pixel) + FIRST_SAMPLE;
    return shade(pixel, window, textureLoad(depth_samples, pixel, 0), 0);
}

@fragment
fn fs_light_sample(@builtin(position) position: vec4<f32>, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(floor(position.xy));
    return shade(pixel, position.xy, textureLoad(depth_samples, pixel, i32(sample)), i32(sample));
}

/// Radiance and coverage of G-buffer sample `sample` of `pixel`, seen at
/// window position `window` with `depth`.
fn shade(pixel: vec2<i32>, window: vec2<f32>, depth: f32, sample: i32) -> vec4<f32> {
    if depth >= 1.0 {
        return vec4<f32>(0.0);  // no mesh here: transparent
    }
    shadow_pixel = vec3<u32>(vec2<u32>(pixel), u32(sample));  // each sample its own noise
    let world = world_position(window, depth);
    let surface = decode_gbuffer(
        textureLoad(albedo_occlusion_samples, pixel, sample),
        textureLoad(normal_roughness_samples, pixel, sample),
        textureLoad(emissive_metallic_samples, pixel, sample),
        world,
    );
    let scope = i32(round(textureLoad(scope_samples, pixel, sample).r));
    let radiance = shade_surface(surface, view_direction(world), scope) * frame.exposure;
    return vec4<f32>(radiance, 1.0);
}

/// Whether all samples of `pixel` hold the same surface, or all none.
fn one_surface(pixel: vec2<i32>) -> bool {
    let depth = textureLoad(depth_samples, pixel, 0);
    let albedo = textureLoad(albedo_occlusion_samples, pixel, 0);
    let normal = textureLoad(normal_roughness_samples, pixel, 0);
    let emissive = textureLoad(emissive_metallic_samples, pixel, 0);
    let scope = textureLoad(scope_samples, pixel, 0).r;
    let distance = view_distance(depth);
    for (var index = 1; index < GBUFFER_SAMPLES; index++) {
        let other = textureLoad(depth_samples, pixel, index);
        if (other >= 1.0) != (depth >= 1.0) {
            return false;  // partly covered
        }
        if depth < 1.0 && abs(view_distance(other) - distance) > SAME_SURFACE_DEPTH * distance {
            return false;
        }
        if !close(textureLoad(albedo_occlusion_samples, pixel, index), albedo, SAME_SURFACE_VALUE)
            || !close(textureLoad(normal_roughness_samples, pixel, index), normal, SAME_SURFACE_NORMAL)
            || !close(textureLoad(emissive_metallic_samples, pixel, index), emissive, SAME_SURFACE_VALUE)
            || textureLoad(scope_samples, pixel, index).r != scope {
            return false;
        }
    }
    return true;
}

/// Whether every component of `a` and `b` differs by at most `tolerance`.
fn close(a: vec4<f32>, b: vec4<f32>, tolerance: f32) -> bool {
    return all(abs(a - b) <= vec4<f32>(tolerance));
}

/// Distance from the camera along its view axis for a depth-buffer value.
fn view_distance(depth: f32) -> f32 {
    let view = frame.inverse_projection * vec4<f32>(0.0, 0.0, depth, 1.0);
    return abs(view.z / view.w);
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
