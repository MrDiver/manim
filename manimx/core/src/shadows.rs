//! Where shadow maps are rendered from, and which of them need rendering
//! again. Ports manimx/renderer/webgpu/shadow_views.py (read its module
//! docs) and `frame_data` from frame.py.
//!
//! The reference caches each light's views; here they are simply computed
//! every frame, which costs a few matrix inverses per light.

use std::f64::consts::PI;

use glam::{DMat4, DVec3, DVec4};
use pyo3::prelude::*;

use crate::callbacks::{bytes_of, Callbacks};
use crate::graph::{FlatScene, LightType, NodeData, SceneGraph, Values};
use crate::lights::LightSource;
use crate::meshes::MeshPacker;
use crate::records::{self, Frame, Light, ShadowView};

const NEAR: f64 = 0.05;
/// Cube faces of a point light: (looking along, up), in the order of
/// `face_index()` in shadows.wgsl.
const CUBE_FACES: [([f64; 3], [f64; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
];

/// The config values a Frame record holds (`RenderConfig`).
pub struct FrameSettings {
    pub time: f64,
    pub layer_depth_step: f64,
    pub exposure: f64,
    pub ambient_linear: [f64; 3],
    pub shadow_softness: f64,
    pub shadow_samples: f64,
    pub shadow_resolution: u32,
    pub shadows: bool,
}

/// `frame_data`: one Frame record for a point of view.
pub fn frame_record(
    view: &DMat4,
    projection: &DMat4,
    position: DVec3,
    size: (u32, u32),
    settings: &FrameSettings,
    light_count: u32,
) -> Frame {
    let (width, height) = (size.0 as f32, size.1 as f32);
    let a = settings.ambient_linear;
    Frame {
        view: records::gpu_matrix(view),
        projection: records::gpu_matrix(projection),
        view_projection: records::gpu_matrix(&(*projection * *view)),
        inverse_view: records::gpu_matrix(&view.inverse()),
        inverse_projection: records::gpu_matrix(&projection.inverse()),
        camera_position: [position.x as f32, position.y as f32, position.z as f32, 1.0],
        viewport: [width, height, 1.0 / width, 1.0 / height],
        time: settings.time as f32,
        layer_depth_step: settings.layer_depth_step as f32,
        light_count,
        exposure: settings.exposure as f32,
        ambient: [a[0] as f32, a[1] as f32, a[2] as f32, 0.0],
        shadow: [
            settings.shadow_softness as f32,
            settings.shadow_samples as f32,
            settings.shadow_resolution as f32,
            if settings.shadows { 1.0 } else { 0.0 },
        ],
    }
}

/// `ShadowViews`: a Frame and a ShadowView record per shadow map layer.
#[derive(Default)]
pub struct ShadowViews {
    pub frames: Vec<Frame>,
    pub views: Vec<ShadowView>,
    /// The clip scope whose casters each layer renders.
    pub scopes: Vec<i32>,
    /// Per layer, whether it may differ from the previous frame's.
    pub changed: Vec<bool>,
}

/// `build_shadow_views`: assigns map layers to the lights that cast
/// shadows (writing each light record's `shadow_view`).
#[allow(clippy::too_many_arguments)]
pub fn build_shadow_views(
    py: Python<'_>,
    graph: &SceneGraph,
    values: &Values,
    scene: &FlatScene,
    lights: &[LightSource],
    rows: &mut [Light],
    bounds: &mut Bounds,
    meshes: &MeshPacker,
    callbacks: &Callbacks,
    settings: &FrameSettings,
) -> PyResult<ShadowViews> {
    let mut shadows = ShadowViews::default();
    if !settings.shadows {
        return Ok(shadows);
    }
    let mut bounds_of: Vec<(i32, Option<(DVec3, f64)>)> = Vec::new();
    for (row, light) in rows.iter_mut().zip(lights) {
        if !light.cast_shadows {
            continue;
        }
        let scope_bounds = match bounds_of.iter().find(|(scope, _)| *scope == light.clip) {
            Some((_, found)) => *found,
            None => {
                let found =
                    bounds.scene_bounds(py, graph, values, scene, light.clip, meshes, callbacks)?;
                bounds_of.push((light.clip, found));
                found
            }
        };
        let Some((center, radius)) = scope_bounds else {
            continue;
        };
        row.shadow_view = shadows.scopes.len() as i32;
        for (view, projection, position, params) in light_views(light, center, radius) {
            let resolution = (settings.shadow_resolution, settings.shadow_resolution);
            shadows.frames.push(frame_record(
                &view,
                &projection,
                position,
                resolution,
                settings,
                0,
            ));
            shadows.views.push(ShadowView {
                view_projection: records::gpu_matrix(&(projection * view)),
                params: params.map(|p| p as f32),
            });
            shadows.scopes.push(light.clip);
        }
    }
    Ok(shadows)
}

/// `_light_views`: (view, projection, position, params) of each view.
fn light_views(
    light: &LightSource,
    center: DVec3,
    radius: f64,
) -> Vec<(DMat4, DMat4, DVec3, [f64; 4])> {
    let position = light.position();
    let direction = light.direction();
    let far = || {
        if light.range > 0.0 {
            light.range
        } else {
            (position - center).length() + radius
        }
    };
    match light.light {
        LightType::Directional => {
            // Look at the scene's bounding sphere from outside it.
            let eye = center - direction * 2.0 * radius;
            let (near, far) = (radius * 0.5, radius * 3.5);
            vec![(
                look(eye, direction, None),
                orthographic(radius, near, far),
                eye,
                [near, far, 2.0 * radius, 0.0],
            )]
        }
        LightType::Spot => {
            let far = far();
            let fov = (2.0 * light.cone.1 + 0.05).min(170f64.to_radians());
            vec![(
                look(position, direction, None),
                perspective(fov, NEAR, far),
                position,
                [NEAR, far, 2.0 * (fov / 2.0).tan(), 1.0],
            )]
        }
        LightType::Point => {
            let far = far();
            CUBE_FACES
                .iter()
                .map(|(forward, up)| {
                    let view = look(position, DVec3::from(*forward), Some(DVec3::from(*up)));
                    (
                        view,
                        perspective(PI / 2.0, NEAR, far),
                        position,
                        [NEAR, far, 2.0, 1.0],
                    )
                })
                .collect()
        }
    }
}

/// `_look`: the view matrix of a camera at `eye` looking along `forward`.
fn look(eye: DVec3, forward: DVec3, up: Option<DVec3>) -> DMat4 {
    let up = up.unwrap_or(if forward.z.abs() < 0.99 {
        DVec3::Z
    } else {
        DVec3::Y
    });
    let right = forward.cross(up).normalize();
    let screen_up = right.cross(forward);
    let rows = [right, screen_up, -forward];
    let translation = DVec3::new(-rows[0].dot(eye), -rows[1].dot(eye), -rows[2].dot(eye));
    DMat4::from_cols(
        DVec4::new(rows[0].x, rows[1].x, rows[2].x, 0.0),
        DVec4::new(rows[0].y, rows[1].y, rows[2].y, 0.0),
        DVec4::new(rows[0].z, rows[1].z, rows[2].z, 0.0),
        translation.extend(1.0),
    )
}

fn orthographic(half_size: f64, near: f64, far: f64) -> DMat4 {
    DMat4::from_cols(
        DVec4::new(1.0 / half_size, 0.0, 0.0, 0.0),
        DVec4::new(0.0, 1.0 / half_size, 0.0, 0.0),
        DVec4::new(0.0, 0.0, -1.0 / (far - near), 0.0),
        DVec4::new(0.0, 0.0, -near / (far - near), 1.0),
    )
}

fn perspective(fov: f64, near: f64, far: f64) -> DMat4 {
    let focal = 1.0 / (fov / 2.0).tan();
    DMat4::from_cols(
        DVec4::new(focal, 0.0, 0.0, 0.0),
        DVec4::new(0.0, focal, 0.0, 0.0),
        DVec4::new(0.0, 0.0, far / (near - far), -1.0),
        DVec4::new(0.0, 0.0, near * far / (near - far), 0.0),
    )
}

/// The boxes of paths (by points version), for `scene_bounds`; meshes
/// keep theirs in `MeshPacker::geometry`.
#[derive(Default)]
pub struct Bounds {
    paths: std::collections::HashMap<u64, (u64, Vec<DVec3>)>,
}

impl Bounds {
    /// `scene_bounds`: the bounding sphere `(center, radius)` of the drawn
    /// objects in `scope`, or None if there are none.
    #[allow(clippy::too_many_arguments)]
    pub fn scene_bounds(
        &mut self,
        py: Python<'_>,
        graph: &SceneGraph,
        values: &Values,
        scene: &FlatScene,
        scope: i32,
        meshes: &MeshPacker,
        callbacks: &Callbacks,
    ) -> PyResult<Option<(DVec3, f64)>> {
        let mut low = DVec3::splat(f64::INFINITY);
        let mut high = DVec3::splat(f64::NEG_INFINITY);
        let mut any = false;
        for item in &scene.items {
            if !scene.in_scope(item.clip, scope) {
                continue;
            }
            let node = &graph.nodes[item.node];
            let corners = match &node.data {
                NodeData::Mesh(fields) => {
                    let Some(geometry) = meshes.geometry.get(&node.uid) else {
                        continue;
                    };
                    if geometry.corners.is_empty() {
                        continue;
                    }
                    match fields.particles {
                        Some(slots) => match particle_box(values, slots, &geometry.corners)? {
                            Some(corners) => corners,
                            None => continue,
                        },
                        None => geometry.corners.clone(),
                    }
                }
                NodeData::Path(fields) => {
                    let points = values.get(fields.points).array()?;
                    if points.rows() < 4 {
                        continue;
                    }
                    self.path_corners(py, values, node.uid, fields.points, callbacks)?
                }
                _ => continue,
            };
            for corner in corners {
                let world = item.world.transform_point3(corner);
                low = low.min(world);
                high = high.max(world);
                any = true;
            }
        }
        if !any {
            return Ok(None);
        }
        Ok(Some((
            (low + high) / 2.0,
            ((high - low).length() / 2.0).max(1e-3),
        )))
    }

    fn path_corners(
        &mut self,
        py: Python<'_>,
        values: &Values,
        uid: u64,
        slot: usize,
        callbacks: &Callbacks,
    ) -> PyResult<Vec<DVec3>> {
        let version = values.state.versions[slot];
        if self.paths.get(&uid).map(|entry| entry.0) != Some(version) {
            let points = values.state.kinds[slot].value_to_py(py, values.get(slot))?;
            let bytes = bytes_of(&callbacks.path_box.bind(py).call1((points,))?)?;
            let corners: &[f64] = bytemuck::cast_slice(&bytes);
            self.paths.insert(
                uid,
                (
                    version,
                    corners
                        .chunks_exact(3)
                        .map(|c| DVec3::new(c[0], c[1], c[2]))
                        .collect(),
                ),
            );
        }
        Ok(self.paths[&uid].1.clone())
    }
}

/// `_particle_box`: a box around every shown particle, grown by the
/// mesh's reach times the largest scale.
fn particle_box(
    values: &Values,
    slots: [usize; 4],
    mesh_corners: &[DVec3],
) -> PyResult<Option<Vec<DVec3>>> {
    let positions = values.get(slots[0]).array()?;
    let scales = values.get(slots[2]).array()?;
    let mut low = DVec3::splat(f64::INFINITY);
    let mut high = DVec3::splat(f64::NEG_INFINITY);
    let mut largest_scale: f64 = 0.0;
    for i in 0..positions.rows() {
        let scale = DVec3::from_slice(&scales.data[3 * i..3 * i + 3]);
        if scale.abs().min_element() <= 1e-9 {
            continue;
        }
        largest_scale = largest_scale.max(scale.abs().max_element());
        let position = DVec3::from_slice(&positions.data[3 * i..3 * i + 3]);
        low = low.min(position);
        high = high.max(position);
    }
    if largest_scale == 0.0 {
        return Ok(None);
    }
    let reach = mesh_corners.iter().map(|c| c.length()).fold(0.0, f64::max) * largest_scale;
    Ok(Some(box_corners(low - reach, high + reach)))
}

/// `_box`: the 8 corners of the box from `low` to `high`.
fn box_corners(low: DVec3, high: DVec3) -> Vec<DVec3> {
    let mut corners = Vec::with_capacity(8);
    for x in [low.x, high.x] {
        for y in [low.y, high.y] {
            for z in [low.z, high.z] {
                corners.push(DVec3::new(x, y, z));
            }
        }
    }
    corners
}

/// `ShadowLayers`: tells which shadow map layers would come out exactly as
/// in the previous frame. A layer depends on its view and on the casters
/// of its scope: their identity, every slot version and world matrix.
#[derive(Default)]
pub struct ShadowLayers {
    keys: Vec<Vec<u64>>,
}

impl ShadowLayers {
    pub fn mark(
        &mut self,
        graph: &SceneGraph,
        values: &Values,
        scene: &FlatScene,
        shadows: &mut ShadowViews,
    ) -> PyResult<()> {
        let mut casters: Vec<(i32, Vec<u64>)> = Vec::new();
        let mut keys = Vec::with_capacity(shadows.frames.len());
        for (index, &scope) in shadows.scopes.iter().enumerate() {
            if !casters.iter().any(|(s, _)| *s == scope) {
                casters.push((scope, caster_key(graph, values, scene, scope)?));
            }
            let mut frame = shadows.frames[index];
            frame.time = 0.0; // no shader reads it in shadow passes
            let mut key: Vec<u64> = bytemuck::cast_slice::<Frame, u32>(&[frame])
                .iter()
                .map(|&w| w as u64)
                .collect();
            key.extend(&casters.iter().find(|(s, _)| *s == scope).unwrap().1);
            keys.push(key);
        }
        shadows.changed = keys
            .iter()
            .enumerate()
            .map(|(index, key)| self.keys.get(index) != Some(key))
            .collect();
        self.keys = keys;
        Ok(())
    }
}

fn caster_key(
    graph: &SceneGraph,
    values: &Values,
    scene: &FlatScene,
    scope: i32,
) -> PyResult<Vec<u64>> {
    let mut key = Vec::new();
    for item in &scene.items {
        let node = &graph.nodes[item.node];
        let cast_shadows = match &node.data {
            NodeData::Path(fields) => fields.cast_shadows,
            NodeData::Mesh(fields) => fields.cast_shadows,
            _ => continue,
        };
        if !values.truthy(cast_shadows)? || !scene.in_scope(item.clip, scope) {
            continue;
        }
        key.push(node.uid);
        key.extend(node.slots.iter().map(|&slot| values.state.versions[slot]));
        key.extend(item.world.to_cols_array().iter().map(|v| v.to_bits()));
    }
    Ok(key)
}
