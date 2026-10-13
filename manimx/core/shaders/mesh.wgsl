// Mesh passes: the geometry pass writes opaque Shape3D surfaces into the
// G-buffer (fs_main); the translucent pass lights translucent ones and
// blends them over the lit layer (fs_forward).
//
// Like the vector pipeline, all meshes share storage buffers (vertex
// pulling) and are drawn with multi_draw_indexed_indirect; `first_instance`
// selects the object. Draws are grouped by material texture set, since the
// textures live in a separate bind group (group 1), and by surface shader,
// which makes a pipeline of its own.
//
// Deferred shading: this pass only evaluates materials (material.wgsl),
// lets the mesh's surface shader change the result (surface_shader.wgsl)
// and stores the resulting Surface (gbuffer.wgsl). Lighting happens afterwards,
// once per MSAA sample, in deferred_lighting.wgsl, so the number of lights
// does not multiply the cost of overdraw.
//
// Translucent meshes (opacity or alpha below 1) have no place in a G-buffer
// of one surface per sample, so they are shaded forward: after the
// lighting pass, sorted back to front, each surface is lit here with the
// same material, surface shader and lighting code and blended over the
// lit layer (premultiplied), depth-tested against the opaque meshes
// without writing depth. Each is drawn twice, back faces then front
// faces, so a closed mesh's far side is behind its near side.

#include "frame.wgsl"
#include "material.wgsl"
#include "gbuffer.wgsl"
#include "surface_shader.wgsl"
#include "lighting.wgsl"

// Mirrors `MeshObject` in core/src/packing/records.rs.
struct MeshObject {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>, // inverse transpose of the model's 3x3 part
    material: u32,
    clip: i32,                  // layer in `mask_layers`, or -1
    _pad0: u32,
    _pad1: u32,
};

// Interleaved as `MESH_VERTEX_SIZE` in core/src/packing/records.rs says (built by core_callbacks.py).
struct MeshVertex {
    position: vec3<f32>,
    u: f32,
    normal: vec3<f32>,
    v: f32,
    tangent: vec4<f32>,         // zero when the mesh has no tangents
};

@group(0) @binding(1) var<storage, read> objects: array<MeshObject>;
@group(0) @binding(2) var<storage, read> vertices: array<MeshVertex>;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var mask_layers: texture_2d_array<f32>;

// Group 2, only for fs_forward: what lighting.wgsl and shadows.wgsl read.
@group(2) @binding(0) var<storage, read> lights: array<Light>;
@group(2) @binding(1) var shadow_maps: texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(3) var<storage, read> shadow_views: array<ShadowView>;
@group(2) @binding(4) var<storage, read> scope_parents: array<i32>;
@group(2) @binding(5) var shadow_transmittance: texture_2d_array<f32>;
@group(2) @binding(6) var shadow_translucent_depth: texture_2d_array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) object: u32,
    @location(5) local_position: vec3<f32>,
};

@vertex
fn vs_main(
    // For indexed draws this already includes the draw's base_vertex.
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) object_index: u32,
) -> VertexOutput {
    let object = objects[object_index];
    let vertex = vertices[vertex_index];
    let world = object.model * vec4<f32>(vertex.position, 1.0);

    var out: VertexOutput;
    out.clip_position = frame.view_projection * world;
    out.world_position = world.xyz;
    out.normal = (object.normal_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    out.tangent = vec4<f32>((object.model * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz, vertex.tangent.w);
    out.uv = vec2<f32>(vertex.u, vertex.v);
    out.object = object_index;
    out.local_position = vertex.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> GBuffer {
    let object = objects[in.object];
    // Meshes are opaque, so clipping is a hard cut at half mask coverage.
    if object.clip >= 0 {
        let pixel = vec2<i32>(floor(in.clip_position.xy));
        if textureLoad(mask_layers, pixel, object.clip, 0).a < 0.5 {
            discard;
        }
    }

    var input: SurfaceInput;
    input.position = in.world_position;
    input.local_position = in.local_position;
    input.normal = in.normal;
    input.tangent = in.tangent;
    input.uv = in.uv;
    input.front_facing = front_facing;
    let surface = shade_surface(evaluate_surface(materials[object.material], input), input);
    return encode_gbuffer(surface, object.clip);
}

/// A translucent mesh, lit: premultiplied radiance and its alpha (the
/// material's opacity times the node's alpha, times a clip mask's).
@fragment
fn fs_forward(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let object = objects[in.object];
    let pixel = vec2<i32>(floor(in.clip_position.xy));
    var coverage = 1.0;
    if object.clip >= 0 {
        coverage = textureLoad(mask_layers, pixel, object.clip, 0).a;
    }

    var input: SurfaceInput;
    input.position = in.world_position;
    input.local_position = in.local_position;
    input.normal = in.normal;
    input.tangent = in.tangent;
    input.uv = in.uv;
    input.front_facing = front_facing;
    let surface = shade_surface(evaluate_surface(materials[object.material], input), input);
    shadow_pixel = vec3<u32>(vec2<u32>(pixel), 0u);
    let radiance = light_surface(surface, view_direction(in.world_position), object.clip) * frame.exposure;
    let alpha = clamp(surface.alpha * coverage, 0.0, 1.0);
    return vec4<f32>(radiance * alpha, alpha);
}

/// Unit vector from a world point towards the viewer.
fn view_direction(point: vec3<f32>) -> vec3<f32> {
    if frame.projection[3][3] == 1.0 {  // orthographic
        return normalize((frame.inverse_view * vec4<f32>(0.0, 0.0, 1.0, 0.0)).xyz);
    }
    return normalize(frame.camera_position.xyz - point);
}

/// Shadow maps: depth only, seen from a light (frame.view_projection is
/// the light's view). Opaque meshes block light completely.
@fragment
fn fs_shadow() {}

/// What a translucent caster does to light, seen from a light (after the
/// opaque casters, which it is depth-tested against): the share of light
/// it lets through, multiplied into the transmittance, and its distance
/// along the light's view, of which the nearest is kept (shadow.rs).
struct TranslucentShadow {
    @location(0) transmittance: vec4<f32>,
    @location(1) depth: vec4<f32>,
};

@fragment
fn fs_shadow_translucent(in: VertexOutput) -> TranslucentShadow {
    let alpha = clamp(materials[objects[in.object].material].base_color.a, 0.0, 1.0);
    let distance = -(frame.view * vec4<f32>(in.world_position, 1.0)).z;
    return TranslucentShadow(vec4<f32>(1.0 - alpha), vec4<f32>(distance));
}
