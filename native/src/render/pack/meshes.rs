//! The mesh records of `RShape3D` and `RParticles` nodes, in the buffers
//! mesh.wgsl reads:
//!
//! ```text
//! objects    per mesh: model and normal matrix, material, clip scope
//! vertices   all mesh vertices, interleaved
//! indices    all triangle indices, relative to each mesh's first vertex
//! materials  material factors, one per material
//! ```
//!
//! Vertices and indices are large and change only with a node's mesh, so
//! they stay in arenas, a range per node, and only the written ranges are
//! uploaded. Objects, materials and draws are small and built per frame.
//! Draws are grouped by the material's texture set and surface shader
//! (textures are bound per group, a pipeline per shader): one
//! `multi_draw_indexed_indirect` per group. A mesh's interleaved vertices
//! and a material's entry are built once per version of the mesh and the
//! material, and kept in memos (cache/).
//!
//! A mesh whose material alpha (opacity times the node's alpha) is below 1
//! is translucent: it is left out of its group's G-buffer draws and listed
//! in `translucent` instead, sorted back to front by the view depth of its
//! box's center, for the forward pass (mesh.wgsl). Its shadow tints the
//! light instead of blocking it (shadow.rs). A mesh with alpha 0 is not
//! drawn at all, and casts no shadow.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::{DMat3, DMat4, DQuat, DVec3};

use crate::cache::Memo;
use crate::flat::FlatScene;
use crate::render::pack::arena::Arena;
use crate::render::pack::casters::shadow_lists;
use crate::render::pack::color::linear_rgb;
use crate::render::pack::tangents::tangents;
use crate::render::packet::records::{self, DrawIndexed, Material, MeshObject, MeshVertex};
use crate::render::packet::{DrawGroup, MeshOutput};
use crate::robjects::{self, Instances, Kind, Mesh, Texture};

/// A mesh as the renderer needs it, built once per mesh version.
pub struct MeshGeometry {
    pub vertices: Vec<u8>,
    pub indices: Vec<u8>,
    pub index_count: u32,
    pub triangle_count: usize,
    /// The 8 corners of the box around its vertices.
    pub corners: Vec<DVec3>,
}

impl MeshGeometry {
    /// The interleaved vertices (tangents computed from the UVs when the
    /// mesh has UVs but no tangents), indices and box of `mesh`.
    fn new(mesh: &Mesh) -> MeshGeometry {
        if mesh.triangle_count() == 0 {
            return MeshGeometry {
                vertices: Vec::new(),
                indices: Vec::new(),
                index_count: 0,
                triangle_count: 0,
                corners: Vec::new(),
            };
        }
        let tangents = mesh.tangents.clone().or_else(|| {
            let uvs = mesh.uvs.as_ref()?;
            Some(tangents(&mesh.positions, &mesh.normals, uvs, &mesh.indices))
        });
        let vertices: Vec<MeshVertex> = (0..mesh.positions.len())
            .map(|i| {
                let [u, v] = mesh.uvs.as_ref().map_or([0.0; 2], |uvs| uvs[i]);
                MeshVertex {
                    position: mesh.positions[i],
                    u,
                    normal: mesh.normals[i],
                    v,
                    tangent: tangents.as_ref().map_or([0.0; 4], |t| t[i]),
                }
            })
            .collect();
        let positions: Vec<DVec3> = mesh
            .positions
            .iter()
            .map(|p| DVec3::from(p.map(f64::from)))
            .collect();
        MeshGeometry {
            vertices: bytemuck::cast_slice(&vertices).to_vec(),
            indices: bytemuck::cast_slice(&mesh.indices).to_vec(),
            index_count: mesh.indices.len() as u32,
            triangle_count: mesh.triangle_count(),
            corners: box_around(&positions),
        }
    }
}

/// The 8 corners of the axis-aligned box around `points`.
pub fn box_around(points: &[DVec3]) -> Vec<DVec3> {
    let low = points.iter().copied().fold(DVec3::INFINITY, DVec3::min);
    let high = points.iter().copied().fold(DVec3::NEG_INFINITY, DVec3::max);
    box_corners(low, high)
}

/// The 8 corners of the box from `low` to `high`.
pub fn box_corners(low: DVec3, high: DVec3) -> Vec<DVec3> {
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

/// The GPU entry of a material, in linear light.
fn material_entry(material: &robjects::Material) -> Material {
    let [r, g, b] = linear_rgb(material.base_color);
    let emissive = linear_rgb(material.emissive).map(|c| c * material.emissive_strength);
    Material {
        base_color: [r as f32, g as f32, b as f32, material.opacity as f32],
        emissive: [
            emissive[0] as f32,
            emissive[1] as f32,
            emissive[2] as f32,
            0.0,
        ],
        metallic: material.metallic as f32,
        roughness: material.roughness as f32,
        normal_scale: material.normal_scale as f32,
        occlusion_strength: material.occlusion_strength as f32,
    }
}

/// What draws are grouped by: the uids of a material's textures and its
/// surface shader (0 for none).
type GroupKey = [u64; 6];

pub struct MeshPacker {
    pub vertices: Arena,
    pub indices: Arena,
    /// uid -> (mesh version, first vertex, vertex count, first index, index count).
    slots: HashMap<u64, (u64, usize, usize, usize, usize)>,
    seen: HashSet<u64>,
    /// By node uid, tagged by the mesh's version.
    pub geometry: Memo<u64, u64, MeshGeometry>,
    /// By node uid, tagged by the material's version.
    pub materials: Memo<u64, u64, Material>,
}

impl MeshPacker {
    pub fn new() -> MeshPacker {
        MeshPacker {
            vertices: Arena::new(records::MESH_VERTEX_SIZE),
            indices: Arena::new(4),
            slots: HashMap::new(),
            seen: HashSet::new(),
            geometry: Memo::default(),
            materials: Memo::default(),
        }
    }

    pub fn too_much_garbage(&self) -> bool {
        self.vertices.too_much_garbage(1 << 16) || self.indices.too_much_garbage(1 << 16)
    }

    /// Start over with empty arenas (keeps the memos).
    pub fn repack(&mut self) {
        self.vertices = Arena::new(records::MESH_VERTEX_SIZE);
        self.indices = Arena::new(4);
        self.slots.clear();
    }

    /// The geometry of node `uid`'s mesh, built again when its version changed.
    fn mesh(&mut self, uid: u64, mesh: &Mesh) -> &MeshGeometry {
        self.geometry
            .get_or_build(uid, mesh.version, || MeshGeometry::new(mesh))
    }

    fn material(&mut self, uid: u64, material: &robjects::Material) -> Material {
        *self
            .materials
            .get_or_build(uid, material.version, || material_entry(material))
    }

    /// `(first index, base vertex)` of node `uid`'s mesh of version
    /// `version` (its geometry built before), placed in the arenas.
    fn place(&mut self, uid: u64, version: u64) -> (usize, usize) {
        self.seen.insert(uid);
        let geometry = self.geometry.get(&uid).expect("built before placing");
        if let Some(slot) = self.slots.get(&uid) {
            if slot.0 == version {
                return (slot.3, slot.1);
            }
        }
        let old = self.slots.get(&uid).copied().unwrap_or((0, 0, 0, 0, 0));
        let first_vertex = self.vertices.place(&geometry.vertices, old.1, old.2);
        let first_index = self.indices.place(&geometry.indices, old.3, old.4);
        let vertex_count = geometry.vertices.len() / records::MESH_VERTEX_SIZE;
        self.slots.insert(
            uid,
            (
                version,
                first_vertex,
                vertex_count,
                first_index,
                geometry.index_count as usize,
            ),
        );
        (first_index, first_vertex)
    }

    /// Free the ranges of meshes not placed since the last call.
    fn finish(&mut self) {
        let gone: Vec<u64> = self
            .slots
            .keys()
            .filter(|uid| !self.seen.contains(uid))
            .copied()
            .collect();
        for uid in gone {
            let (_, _, vertex_count, _, index_count) = self.slots.remove(&uid).unwrap();
            self.vertices.garbage += vertex_count;
            self.indices.garbage += index_count;
        }
        self.seen.clear();
    }

    /// The frame's mesh records, without the lights; `view` is the
    /// camera's view matrix (world to camera), to sort translucent meshes.
    pub fn pack(&mut self, scene: &FlatScene, view: &DMat4) -> MeshOutput {
        let mut objects = Vec::new();
        let mut materials = Vec::new();
        let mut groups: Vec<(GroupKey, DrawGroup)> = Vec::new();
        let mut translucent: Vec<(f64, usize, DrawIndexed)> = Vec::new();
        let mut shadow_draws: Vec<(DrawIndexed, i32)> = Vec::new();
        let mut translucent_shadow_draws: Vec<(DrawIndexed, i32)> = Vec::new();
        // A node drawn more than once (it sits in several groups) shares
        // its material entry while its alpha is the same.
        let mut material_slots: HashMap<(u64, u64), u32> = HashMap::new();
        for item in &scene.items {
            let node = &item.node;
            let (surface, instances) = match &node.kind {
                Kind::Shape3D(surface) => (surface, None),
                Kind::Particles { surface, instances } => (surface, Some(instances)),
                _ => continue,
            };
            if self.mesh(node.uid, &surface.mesh).triangle_count == 0 {
                continue;
            }
            let mut entry = self.material(node.uid, &surface.material);
            if item.alpha != 1.0 {
                // The item's alpha (RNode.alpha, times every one above) fades the material.
                entry.base_color[3] *= item.alpha as f32;
            }
            if entry.base_color[3] <= 0.0 {
                continue;
            }
            let first_object = objects.len() as u32;
            let count = match instances {
                Some(instances) => {
                    let (placed, tinted) = particle_objects(
                        instances,
                        &item.world,
                        entry,
                        materials.len() as u32,
                        item.clip,
                    );
                    if placed.is_empty() {
                        continue;
                    }
                    let count = placed.len();
                    objects.extend(placed);
                    materials.extend(tinted);
                    count
                }
                None => {
                    let material = *material_slots
                        .entry((node.uid, item.alpha.to_bits()))
                        .or_insert_with(|| {
                            materials.push(entry);
                            materials.len() as u32 - 1
                        });
                    objects.push(mesh_object(&item.world, material, item.clip));
                    1
                }
            };
            let material = &surface.material;
            let uid_of = |texture: &Option<Arc<Texture>>| texture.as_ref().map_or(0, |t| t.uid);
            let key: GroupKey = [
                uid_of(&material.textures[0]),
                uid_of(&material.textures[1]),
                uid_of(&material.textures[2]),
                uid_of(&material.textures[3]),
                uid_of(&material.textures[4]),
                material.surface_shader.as_ref().map_or(0, |s| s.uid),
            ];
            let group = match groups.iter().position(|(seen, _)| *seen == key) {
                Some(index) => index,
                None => {
                    groups.push((
                        key,
                        DrawGroup {
                            textures: material.textures.clone(),
                            surface_shader: material.surface_shader.clone(),
                            draws: Vec::new(),
                        },
                    ));
                    groups.len() - 1
                }
            };
            let (first_index, base_vertex) = self.place(node.uid, surface.mesh.version);
            let geometry = self.geometry.get(&node.uid).expect("built above");
            let draw = DrawIndexed {
                index_count: geometry.index_count,
                instance_count: count as u32,
                first_index: first_index as u32,
                base_vertex: base_vertex as i32,
                first_instance: first_object,
            };
            let is_translucent = entry.base_color[3] < 1.0;
            if is_translucent {
                let corners = &geometry.corners;
                let center = corners.iter().copied().sum::<DVec3>() / corners.len() as f64;
                let depth = -view.transform_point3(item.world.transform_point3(center)).z;
                translucent.push((depth, group, draw));
            } else {
                groups[group].1.draws.push(draw);
            }
            if surface.cast_shadows {
                let casters = if is_translucent {
                    &mut translucent_shadow_draws
                } else {
                    &mut shadow_draws
                };
                casters.push((draw, item.clip));
            }
        }
        self.finish();
        // Farthest first; meshes at the same depth keep the tree's order.
        translucent.sort_by(|a, b| b.0.total_cmp(&a.0));
        MeshOutput {
            objects,
            materials,
            groups: groups.into_iter().map(|(_, group)| group).collect(),
            translucent: translucent
                .into_iter()
                .map(|(_, group, draw)| (group, draw))
                .collect(),
            shadow_draws: shadow_lists(scene, &shadow_draws),
            translucent_shadow_draws: shadow_lists(scene, &translucent_shadow_draws),
        }
    }
}

fn mesh_object(world: &DMat4, material: u32, clip: i32) -> MeshObject {
    // The normal matrix: the inverse transpose of the upper 3x3 block.
    let normal = DMat4::from_mat3(DMat3::from_mat4(*world).inverse().transpose());
    MeshObject {
        model: records::gpu_matrix(world),
        normal_matrix: records::gpu_matrix(&normal),
        material,
        clip,
        _pad: [0; 2],
    }
}

/// Whether particle `i` shows: scaled to something on every axis.
pub fn particle_shown(instances: &Instances, i: usize) -> bool {
    DVec3::from(instances.scales[i]).abs().min_element() > 1e-9
}

/// One object and one tinted material per shown particle.
fn particle_objects(
    instances: &Instances,
    world: &DMat4,
    entry: Material,
    first_material: u32,
    clip: i32,
) -> (Vec<MeshObject>, Vec<Material>) {
    let mut objects = Vec::new();
    let mut materials = Vec::new();
    for i in 0..instances.count() {
        if !particle_shown(instances, i) {
            continue;
        }
        // w, x, y, z; interpolated linearly, so normalized here.
        let [w, x, y, z] = instances.rotations[i];
        let local = DMat4::from_scale_rotation_translation(
            DVec3::from(instances.scales[i]),
            DQuat::from_xyzw(x, y, z, w).normalize(),
            DVec3::from(instances.positions[i]),
        );
        objects.push(mesh_object(
            &(*world * local),
            first_material + materials.len() as u32,
            clip,
        ));
        let color = instances.colors[i];
        let mut tinted = entry;
        for channel in 0..3 {
            tinted.base_color[channel] =
                (tinted.base_color[channel] as f64 * color[channel]) as f32;
            tinted.emissive[channel] =
                (tinted.emissive[channel] as f64 * (color[channel] * color[3])) as f32;
        }
        materials.push(tinted);
    }
    (objects, materials)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::{DrawItem, NO_CLIP};
    use crate::robjects::{RNode, Surface, Transform};

    fn triangle(uid: u64) -> FlatScene {
        let mesh = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
            vec![[0.0, 0.0, 1.0]; 3],
            None,
            None,
        )
        .unwrap();
        let node = Arc::new(RNode {
            uid,
            transform: Transform::IDENTITY,
            alpha: 1.0,
            kind: Kind::Shape3D(Surface {
                mesh: Arc::new(mesh),
                material: Arc::new(robjects::Material::default()),
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

    #[test]
    fn a_mesh_is_built_and_placed_once() {
        let scene = triangle(1);
        let mut packer = MeshPacker::new();
        let output = packer.pack(&scene, &DMat4::IDENTITY);
        assert_eq!(output.groups.len(), 1);
        assert_eq!(output.groups[0].draws[0].index_count, 3);
        assert_eq!(output.shadow_draws[0].1.len(), 1);
        packer.vertices.ranges(); // as uploaded
        packer.indices.ranges();
        packer.pack(&scene, &DMat4::IDENTITY);
        assert_eq!(packer.vertices.ranges(), Some(Vec::new()));
        assert_eq!(packer.indices.ranges(), Some(Vec::new()));
        assert_eq!(packer.geometry.len(), 1);
    }
}
