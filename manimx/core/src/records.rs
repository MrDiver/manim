//! GPU records, field for field as the WGSL structs in
//! manimx/renderer/shaders/ declare them. Matrices are column-major f32,
//! as WGSL reads them: `glam::Mat4::to_cols_array()`.

use bytemuck::{Pod, Zeroable};
use glam::DMat4;

/// A world matrix as a shader reads it.
pub fn gpu_matrix(m: &DMat4) -> [f32; 16] {
    m.as_mat4().to_cols_array()
}

// -- vector records (vector.wgsl) -----------------------------------------------

/// `OBJECT_DTYPE`: one path's placement.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Object {
    pub model: [f32; 16],
    pub layer: f32,
    pub clip: i32,
    pub flags: u32,
    pub fill_colors: u32,
    pub alpha: f32,
    /// Layer of the camera view texture the fill shows, or `NO_VIEW`.
    pub view: i32,
    /// `1 / (width, height)` of a camera view's plane.
    pub view_scale: [f32; 2],
}

/// `Object.view` of every shape that is not a camera view.
pub const NO_VIEW: i32 = -1;

/// `INSTANCE_DTYPE`: what one draw instance generates.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub kind: u32,
    pub object: u32,
    pub curve: u32,
    pub _pad: u32,
}

/// `STROKE_CURVE_DTYPE`: one cubic of a stroke, linked to its neighbors.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StrokeCurve {
    pub first_point: u32,
    pub prev_point: i32,
    pub next_point: i32,
    pub start_joint: u32,
    pub end_joint: u32,
    pub start_cap: u32,
    pub end_cap: u32,
    pub _pad: u32,
}

/// WebGPU's `DrawIndirectArgs`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Bytes per record of the vector buffers whose records Python builds.
pub const FILL_VERTEX_SIZE: usize = 32;
pub const FILL_COLOR_SIZE: usize = 16;
pub const STROKE_POINT_SIZE: usize = 32;

pub const KIND_FILL: u32 = 0;
pub const KIND_STROKE: u32 = 1;
pub const NO_NEIGHBOR: i32 = -1;
pub const FLAG_CAMERA_FACING: u32 = 1;
pub const FLAG_FRONT_ONLY: u32 = 2;
/// `STROKE_VERTEX_COUNT` (keep in sync with vector.wgsl).
pub const STROKE_VERTEX_COUNT: u32 = 24 * 6 + 2 * 8 * 3;

// -- mesh records (mesh.wgsl, lighting.wgsl) --------------------------------------

/// `MESH_OBJECT_DTYPE`: one mesh (or particle) placement.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshObject {
    pub model: [f32; 16],
    pub normal_matrix: [f32; 16],
    pub material: u32,
    pub clip: i32,
    pub _pad: [u32; 2],
}

/// `MATERIAL_DTYPE`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Material {
    pub base_color: [f32; 4],
    pub emissive: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
}

/// `LIGHT_DTYPE`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Light {
    pub position: [f32; 3],
    pub kind: u32,
    pub direction: [f32; 3],
    pub range: f32,
    pub color: [f32; 3],
    pub inner_cone_cos: f32,
    pub outer_cone_cos: f32,
    pub shadow_view: i32,
    pub size: f32,
    pub scope: i32,
}

pub const LIGHT_DIRECTIONAL: u32 = 0;
pub const LIGHT_POINT: u32 = 1;
pub const LIGHT_SPOT: u32 = 2;

/// WebGPU's `DrawIndexedIndirectArgs`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawIndexed {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// Bytes per interleaved mesh vertex (`MESH_VERTEX_DTYPE`, built by Python).
pub const MESH_VERTEX_SIZE: usize = 48;

// -- frames and shadows (frame.wgsl, shadows.wgsl) ----------------------------

/// `FRAME_DTYPE`: a point of view (the camera, or a light for its shadow map).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Frame {
    pub view: [f32; 16],
    pub projection: [f32; 16],
    pub view_projection: [f32; 16],
    pub inverse_view: [f32; 16],
    pub inverse_projection: [f32; 16],
    pub camera_position: [f32; 4],
    pub viewport: [f32; 4],
    pub time: f32,
    pub layer_depth_step: f32,
    pub light_count: u32,
    pub exposure: f32,
    pub ambient: [f32; 4],
    pub shadow: [f32; 4],
    pub shadow_filter: [f32; 4],
}

/// `SHADOW_VIEW_DTYPE`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShadowView {
    pub view_projection: [f32; 16],
    /// near, far, frustum width, 1 = perspective / 0 = orthographic.
    pub params: [f32; 4],
}

const _: () = {
    assert!(size_of::<Object>() == 96);
    assert!(size_of::<Instance>() == 16);
    assert!(size_of::<StrokeCurve>() == 32);
    assert!(size_of::<DrawIndirect>() == 16);
    assert!(size_of::<MeshObject>() == 144);
    assert!(size_of::<Material>() == 48);
    assert!(size_of::<Light>() == 64);
    assert!(size_of::<DrawIndexed>() == 20);
    assert!(size_of::<Frame>() == 416);
    assert!(size_of::<ShadowView>() == 80);
};
