//! The WGSL shaders (native/shaders/), compiled into the library: no
//! folder is read at run time, so the renderer works where there is no
//! file system (the browser). `#include "file.wgsl"` lines are expanded
//! from this table (device.rs).

use crate::error::{Error, Result};

/// Every shader file, by name.
const SHADERS: &[(&str, &str)] = &[
    ("bloom.wgsl", include_str!("../../shaders/bloom.wgsl")),
    ("color.wgsl", include_str!("../../shaders/color.wgsl")),
    (
        "composite.wgsl",
        include_str!("../../shaders/composite.wgsl"),
    ),
    (
        "deferred_lighting.wgsl",
        include_str!("../../shaders/deferred_lighting.wgsl"),
    ),
    (
        "depth_of_field_tiles.wgsl",
        include_str!("../../shaders/depth_of_field_tiles.wgsl"),
    ),
    (
        "depth_of_field.wgsl",
        include_str!("../../shaders/depth_of_field.wgsl"),
    ),
    ("finalize.wgsl", include_str!("../../shaders/finalize.wgsl")),
    ("frame.wgsl", include_str!("../../shaders/frame.wgsl")),
    (
        "fullscreen.wgsl",
        include_str!("../../shaders/fullscreen.wgsl"),
    ),
    ("gbuffer.wgsl", include_str!("../../shaders/gbuffer.wgsl")),
    ("lens.wgsl", include_str!("../../shaders/lens.wgsl")),
    ("lighting.wgsl", include_str!("../../shaders/lighting.wgsl")),
    ("material.wgsl", include_str!("../../shaders/material.wgsl")),
    ("mesh.wgsl", include_str!("../../shaders/mesh.wgsl")),
    ("shadows.wgsl", include_str!("../../shaders/shadows.wgsl")),
    (
        "surface_shader.wgsl",
        include_str!("../../shaders/surface_shader.wgsl"),
    ),
    ("surface.wgsl", include_str!("../../shaders/surface.wgsl")),
    ("vector.wgsl", include_str!("../../shaders/vector.wgsl")),
    (
        "view_distance.wgsl",
        include_str!("../../shaders/view_distance.wgsl"),
    ),
];

/// The code of the shader file `name`.
pub fn source(name: &str) -> Result<String> {
    SHADERS
        .iter()
        .find(|(file, _)| *file == name)
        .map(|(_, code)| code.to_string())
        .ok_or_else(|| Error::Shader(format!("no shader file named {name:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shader_file_is_in_the_table() {
        let folder = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
        let mut files: Vec<String> = std::fs::read_dir(folder)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".wgsl"))
            .collect();
        files.sort();
        let mut listed: Vec<&str> = SHADERS.iter().map(|(name, _)| *name).collect();
        listed.sort();
        assert_eq!(files, listed);
    }
}
