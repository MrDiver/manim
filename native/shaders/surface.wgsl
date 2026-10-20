// The interface between materials and lighting.

/// Everything lighting needs to know about a point on a surface.
struct Surface {
    position: vec3<f32>,        // world space
    normal: vec3<f32>,          // world space, unit, facing the viewer
    base_color: vec3<f32>,      // linear
    alpha: f32,
    metallic: f32,
    roughness: f32,             // perceptual roughness
    occlusion: f32,
    emissive: vec3<f32>,        // linear radiance
};

