//! Where shadow maps are rendered from, and which of them need rendering
//! again.
//!
//! Every light that casts shadows gets one or more *shadow views*: a
//! camera at the light that renders the depth of all casters into one
//! layer of the shadow map array.
//!
//! ```text
//! directional  1 view, orthographic, fitted around its scope's bounds
//! spot         1 view, perspective, covering the light's outer cone
//! point        6 views, perspective 90 degrees, one per cube face
//!              (+x, -x, +y, -y, +z, -z; the shader picks the face from
//!              the direction to the light)
//! ```
//!
//! The six views of a point light follow GPU Gems, chapter 12,
//! "Omnidirectional Shadow Mapping" (Gerasimov, 2004:
//! https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-12-omnidirectional-shadow-mapping).
//!
//! Each view gives a Frame record (to render the map with the normal
//! pipelines) and a ShadowView record (to look the map up when lighting).
//! The views are computed every frame, which costs a few matrix inverses
//! per light; `ShadowLayers` tells which layers would come out exactly as
//! before, so the renderer skips their passes.

use std::f64::consts::PI;

use std::collections::BTreeSet;

use glam::{DMat4, DVec3, DVec4};

use crate::cache::Memo;
use crate::flat::{DrawItem, FlatScene};
use crate::render::pack::lights::LightSource;
use crate::render::pack::meshes::{box_around, box_corners, particle_shown, MeshPacker};
use crate::render::packet::records::{self, Frame, Light, ShadowView};
use crate::render::packet::ShadowViews;
use crate::robjects::{Instances, Kind, LightKind, Points};

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

/// The settings a Frame record holds (from the `RenderConfig`).
pub struct FrameSettings {
    pub time: f64,
    pub layer_depth_step: f64,
    pub exposure: f64,
    pub ambient_linear: [f64; 3],
    pub shadow_softness: f64,
    pub shadow_samples: f64,
    pub shadow_resolution: u32,
    pub shadows: bool,
    /// `RenderConfig.shadow_filter`.
    pub shadow_filter: [f32; 4],
}

/// One Frame record for a point of view.
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
        shadow_filter: settings.shadow_filter,
    }
}

/// Assign map layers to the lights that cast
/// shadows (writing each light record's `shadow_view`).
pub fn build_shadow_views(
    scene: &FlatScene,
    lights: &[LightSource],
    rows: &mut [Light],
    bounds: &mut Bounds,
    meshes: &MeshPacker,
    settings: &FrameSettings,
) -> ShadowViews {
    let mut shadows = ShadowViews::default();
    if !settings.shadows {
        return shadows;
    }
    let mut bounds_of: Vec<(i32, Option<(DVec3, f64)>)> = Vec::new();
    for (row, light) in rows.iter_mut().zip(lights) {
        if !light.cast_shadows {
            continue;
        }
        let scope_bounds = match bounds_of.iter().find(|(scope, _)| *scope == light.clip) {
            Some((_, found)) => *found,
            None => {
                let found = bounds.scene_bounds(scene, light.clip, meshes);
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
    shadows
}

/// (view, projection, position, params) of each view.
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
        LightKind::Directional => {
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
        LightKind::Spot => {
            let far = far();
            let fov = (2.0 * light.cone.1 + 0.05).min(170f64.to_radians());
            vec![(
                look(position, direction, None),
                perspective(fov, NEAR, far),
                position,
                [NEAR, far, 2.0 * (fov / 2.0).tan(), 1.0],
            )]
        }
        LightKind::Point => {
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

/// The view matrix of a camera at `eye` looking along `forward`.
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

/// The boxes of paths, by node uid, tagged by the points version they
/// were built from; meshes keep theirs in `MeshPacker::geometry`.
#[derive(Default)]
pub struct Bounds {
    paths: Memo<u64, u64, Vec<DVec3>>,
}

impl Bounds {
    /// The bounding sphere `(center, radius)` of the drawn
    /// objects in `scope`, or None if there are none.
    pub fn scene_bounds(
        &mut self,
        scene: &FlatScene,
        scope: i32,
        meshes: &MeshPacker,
    ) -> Option<(DVec3, f64)> {
        let mut low = DVec3::splat(f64::INFINITY);
        let mut high = DVec3::splat(f64::NEG_INFINITY);
        let mut any = false;
        for item in &scene.items {
            if !scene.in_scope(item.clip, scope) {
                continue;
            }
            let node = &item.node;
            let corners = match &node.kind {
                Kind::Shape3D(surface) | Kind::Particles { surface, .. } => {
                    let Some(geometry) = meshes.geometry.get(&node.uid, &surface.mesh.version)
                    else {
                        continue;
                    };
                    if geometry.corners.is_empty() {
                        continue;
                    }
                    match &node.kind {
                        Kind::Particles { instances, .. } => {
                            match particle_box(instances, &geometry.corners) {
                                Some(corners) => corners,
                                None => continue,
                            }
                        }
                        _ => geometry.corners.clone(),
                    }
                }
                Kind::Shape2D { path, .. } | Kind::Curve3D { path } => {
                    if path.points.count() < 4 {
                        continue;
                    }
                    self.path_corners(node.uid, &path.points)
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
            return None;
        }
        Some(((low + high) / 2.0, ((high - low).length() / 2.0).max(1e-3)))
    }

    /// The corners of the box around a path's points, kept per points version.
    fn path_corners(&mut self, uid: u64, points: &Points) -> Vec<DVec3> {
        self.paths
            .get_or_build(uid, points.version, || {
                let all: Vec<DVec3> = (0..points.count()).map(|i| points.point(i)).collect();
                box_around(&all)
            })
            .clone()
    }

    /// Drop the boxes of the nodes `keep` says no.
    pub fn forget(&mut self, keep: impl Fn(u64) -> bool) {
        self.paths.retain(|uid| keep(*uid));
    }
}

/// A box around every shown particle, grown by the mesh's reach times the
/// largest scale.
fn particle_box(instances: &Instances, mesh_corners: &[DVec3]) -> Option<Vec<DVec3>> {
    let mut low = DVec3::splat(f64::INFINITY);
    let mut high = DVec3::splat(f64::NEG_INFINITY);
    let mut largest_scale: f64 = 0.0;
    for i in 0..instances.count() {
        if !particle_shown(instances, i) {
            continue;
        }
        largest_scale = largest_scale.max(DVec3::from(instances.scales[i]).abs().max_element());
        let position = DVec3::from(instances.positions[i]);
        low = low.min(position);
        high = high.max(position);
    }
    if largest_scale == 0.0 {
        return None;
    }
    let reach = mesh_corners.iter().map(|c| c.length()).fold(0.0, f64::max) * largest_scale;
    Some(box_corners(low - reach, high + reach))
}

/// Which shadow map layers would come out exactly as
/// in the previous frame. A layer depends on its view and on the casters
/// of its scope: their identity, every slot version and world matrix.
#[derive(Default)]
pub struct ShadowLayers {
    keys: Vec<Vec<u64>>,
}

impl ShadowLayers {
    pub fn mark(&mut self, scene: &FlatScene, shadows: &mut ShadowViews) {
        let mut casters: Vec<(i32, Vec<u64>)> = Vec::new();
        let mut keys = Vec::with_capacity(shadows.frames.len());
        for (index, &scope) in shadows.scopes.iter().enumerate() {
            if !casters.iter().any(|(s, _)| *s == scope) {
                casters.push((scope, caster_key(scene, scope)));
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
    }
}

/// Everything the casters of `scope` look like: each caster's (and each
/// mask's that clips one) fingerprint and world matrix.
fn caster_key(scene: &FlatScene, scope: i32) -> Vec<u64> {
    let mut key = Vec::new();
    // The masks that clip the casters shape the shadow too.
    let mut masks = BTreeSet::new();
    let add = |key: &mut Vec<u64>, item: &DrawItem| {
        item.node.fingerprint(key);
        key.extend(item.world.to_cols_array().iter().map(|v| v.to_bits()));
        // Translucent meshes cast shadows as dark as their alpha, which the
        // groups above them multiply in.
        key.push(item.alpha.to_bits());
    };
    for item in &scene.items {
        let cast_shadows = match &item.node.kind {
            Kind::Shape2D { path, .. } | Kind::Curve3D { path } => path.cast_shadows,
            Kind::Shape3D(surface) | Kind::Particles { surface, .. } => surface.cast_shadows,
            _ => continue,
        };
        if !cast_shadows || !scene.in_scope(item.clip, scope) {
            continue;
        }
        masks.extend(scene.mask_chain(item.clip));
        add(&mut key, item);
    }
    for clip in masks {
        for item in &scene.clips[clip as usize].mask {
            add(&mut key, item);
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::flat::NO_CLIP;
    use crate::render::pack::lights::light_sources;
    use crate::robjects::{Material, Mesh, RNode, Surface, Transform};

    fn scene_with_a_mesh() -> FlatScene {
        let mesh = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
            vec![[0.0, 0.0, 1.0]; 3],
            None,
            None,
        )
        .unwrap();
        let node = Arc::new(RNode {
            uid: 1,
            transform: Transform::IDENTITY,
            alpha: 1.0,
            kind: Kind::Shape3D(Surface {
                mesh: Arc::new(mesh),
                material: Arc::new(Material::default()),
                cast_shadows: true,
            }),
        });
        FlatScene {
            items: vec![DrawItem {
                node,
                world: DMat4::IDENTITY,
                order: 0,
                clip: NO_CLIP,
                alpha: 1.0,
            }],
            ..FlatScene::default()
        }
    }

    fn settings() -> FrameSettings {
        FrameSettings {
            time: 0.0,
            layer_depth_step: 2e-7,
            exposure: 1.0,
            ambient_linear: [0.0; 3],
            shadow_softness: 1.0,
            shadow_samples: 16.0,
            shadow_resolution: 512,
            shadows: true,
            shadow_filter: [1.0, 16.0, 0.0, 0.0],
        }
    }

    #[test]
    fn an_unchanged_shadow_layer_is_not_rendered_again() {
        let scene = scene_with_a_mesh();
        let mut meshes = MeshPacker::new();
        meshes.pack(&scene, &DMat4::IDENTITY);
        // No light in the scene: the default key light casts the shadow.
        let sources = light_sources(&scene, true, &DMat4::IDENTITY);
        let (mut bounds, mut layers) = (Bounds::default(), ShadowLayers::default());
        let mut frame = |layers: &mut ShadowLayers| {
            let mut rows: Vec<Light> = sources.iter().map(|light| light.record()).collect();
            let mut views = build_shadow_views(
                &scene,
                &sources,
                &mut rows,
                &mut bounds,
                &meshes,
                &settings(),
            );
            layers.mark(&scene, &mut views);
            (rows[0].shadow_view, views.changed)
        };
        assert_eq!(frame(&mut layers), (0, vec![true]));
        assert_eq!(frame(&mut layers), (0, vec![false]));
    }
}
