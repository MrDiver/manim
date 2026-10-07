// Mesh geometry pass: writes Shape3D surfaces into the G-buffer.
//
// Like the vector pipeline, all meshes share storage buffers (vertex
// pulling) and are drawn with multi_draw_indexed_indirect; `first_instance`
// selects the object. Draws are grouped by material texture set, since the
// textures live in a separate bind group (group 1).
//
// Deferred shading: this pass only evaluates materials (material.wgsl) and
// stores the resulting Surface (gbuffer.wgsl). Lighting happens afterwards,
// once per MSAA sample, in deferred_lighting.wgsl, so the number of lights
// does not multiply the cost of overdraw.

#include "frame.wgsl"
#include "material.wgsl"
#include "gbuffer.wgsl"

// Mirrors MESH_OBJECT_DTYPE in mesh_batch.py.
struct MeshObject {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>, // inverse transpose of the model's 3x3 part
    material: u32,
    clip: i32,                  // layer in `mask_layers`, or -1
    _pad0: u32,
    _pad1: u32,
};

// Mirrors MESH_VERTEX_DTYPE in mesh_batch.py.
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) object: u32,
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
    input.normal = in.normal;
    input.tangent = in.tangent;
    input.uv = in.uv;
    input.front_facing = front_facing;
    return encode_gbuffer(evaluate_surface(materials[object.material], input), object.clip);
}

/// Shadow maps: depth only, seen from a light (frame.view_projection is
/// the light's view). Meshes always block light completely.
@fragment
fn fs_shadow() {}
