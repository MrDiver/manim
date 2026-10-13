// A single triangle covering the whole viewport, for passes that process
// textures pixel by pixel (lighting, compositing, post effects).

struct FullscreenVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,   // (0, 0) top left, (1, 1) bottom right
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenVertex {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FullscreenVertex;
    out.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}
