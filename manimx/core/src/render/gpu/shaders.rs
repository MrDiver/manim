//! The WGSL shaders (core/shaders/), compiled into the library: no folder
//! is read at run time, so the renderer works where there is no file
//! system (the browser). `#include "file.wgsl"` lines are expanded from
//! this table (context.rs).
//!
//! A shader can be replaced by other code before the first device is
//! made (`override_shader`); scripts/render_internals.py marks the
//! lighting pass's edge pixels that way.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::render::error::{Error, Result};

/// Every shader file, by name.
const SHADERS: &[(&str, &str)] = &[
    ("bloom.wgsl", include_str!("../../../shaders/bloom.wgsl")),
    ("color.wgsl", include_str!("../../../shaders/color.wgsl")),
    (
        "composite.wgsl",
        include_str!("../../../shaders/composite.wgsl"),
    ),
    (
        "deferred_lighting.wgsl",
        include_str!("../../../shaders/deferred_lighting.wgsl"),
    ),
    (
        "depth_of_field_tiles.wgsl",
        include_str!("../../../shaders/depth_of_field_tiles.wgsl"),
    ),
    (
        "depth_of_field.wgsl",
        include_str!("../../../shaders/depth_of_field.wgsl"),
    ),
    ("finalize.wgsl", include_str!("../../../shaders/finalize.wgsl")),
    ("frame.wgsl", include_str!("../../../shaders/frame.wgsl")),
    (
        "fullscreen.wgsl",
        include_str!("../../../shaders/fullscreen.wgsl"),
    ),
    ("gbuffer.wgsl", include_str!("../../../shaders/gbuffer.wgsl")),
    ("lens.wgsl", include_str!("../../../shaders/lens.wgsl")),
    ("lighting.wgsl", include_str!("../../../shaders/lighting.wgsl")),
    ("material.wgsl", include_str!("../../../shaders/material.wgsl")),
    ("mesh.wgsl", include_str!("../../../shaders/mesh.wgsl")),
    ("shadows.wgsl", include_str!("../../../shaders/shadows.wgsl")),
    (
        "surface_shader.wgsl",
        include_str!("../../../shaders/surface_shader.wgsl"),
    ),
    ("surface.wgsl", include_str!("../../../shaders/surface.wgsl")),
    ("vector.wgsl", include_str!("../../../shaders/vector.wgsl")),
    (
        "view_distance.wgsl",
        include_str!("../../../shaders/view_distance.wgsl"),
    ),
];

/// Code that replaces a shader file, by name (`override_shader`).
static OVERRIDES: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

/// The code of the shader file `name`.
pub fn source(name: &str) -> Result<String> {
    let overrides = OVERRIDES.lock().expect("not poisoned");
    if let Some(code) = overrides.as_ref().and_then(|map| map.get(name)) {
        return Ok(code.clone());
    }
    SHADERS
        .iter()
        .find(|(file, _)| *file == name)
        .map(|(_, code)| code.to_string())
        .ok_or_else(|| Error::Shader(format!("no shader file named {name:?}")))
}

/// Use `code` for the shader file `name` from now on. Devices compile
/// their pipelines once, so this is for before the first render.
pub fn override_shader(name: &str, code: &str) -> Result<()> {
    if !SHADERS.iter().any(|(file, _)| *file == name) {
        return Err(Error::Invalid(format!("no shader file named {name:?}")));
    }
    let mut overrides = OVERRIDES.lock().expect("not poisoned");
    overrides
        .get_or_insert_with(HashMap::new)
        .insert(name.to_string(), code.to_string());
    Ok(())
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
