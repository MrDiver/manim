// Per-frame values shared by every pipeline. Mirrors FRAME_DTYPE in
// renderer.py; always bound at group 0, binding 0.

struct Frame {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,        // width, height, 1 / width, 1 / height
    time: f32,                  // seconds, from RenderConfig.time
    layer_depth_step: f32,      // see layered_depth() in vector.wgsl
    light_count: u32,           // entries used in the lights buffer
    exposure: f32,              // multiplies lit colors before tone mapping
    ambient: vec4<f32>,         // linear RGB ambient light, w unused
};

@group(0) @binding(0) var<uniform> frame: Frame;
