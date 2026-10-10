//! The mesh records of `Shape3D` and `Particles` nodes. Ports `MeshBatch`
//! and `RetainedMeshes` (manimx/renderer/webgpu/mesh_batch.py).
//!
//! Vertices and indices are large and change only with a node's mesh, so
//! they stay in arenas, a range per node. Objects, materials and draws are
//! small and built per frame. A mesh's interleaved vertices and a
//! material's entry are built by Python once per version
//! (manimx/renderer/core_callbacks.py).

use std::collections::{HashMap, HashSet};

use glam::{DMat3, DMat4, DQuat, DVec3};
use pyo3::prelude::*;

use crate::arena::Arena;
use crate::callbacks::{bytes_of, Callbacks};
use crate::graph::{FlatScene, MeshFields, NodeData, SceneGraph, ShadowList, Values};
use crate::records::{self, DrawIndexed, Material, MeshObject};

/// A mesh as the renderer needs it, built once per mesh version.
pub struct MeshGeometry {
    pub version: u64,
    pub vertices: Vec<u8>,
    pub indices: Vec<u8>,
    pub index_count: u32,
    pub triangle_count: usize,
    /// The 8 corners of the box around its vertices.
    pub corners: Vec<DVec3>,
}

/// A node's material entry, with the versions it was built from and the
/// identities of its textures and surface shader (draws are grouped by them).
struct MaterialEntry {
    versions: Vec<u64>,
    entry: Material,
    texture_key: Vec<usize>,
}

/// Meshes sharing one texture set and surface shader: one bind group, one
/// pipeline, one multi-draw (`DrawGroup`).
pub struct DrawGroup {
    pub textures: Vec<Py<PyAny>>,
    /// A `SurfaceShader`, or None.
    pub surface_shader: Py<PyAny>,
    pub draws: Vec<DrawIndexed>,
}

/// `MeshBatch` without its arenas.
pub struct MeshOutput {
    pub objects: Vec<MeshObject>,
    pub materials: Vec<Material>,
    pub groups: Vec<DrawGroup>,
    /// Per light scope (NO_CLIP first), the shadow-casting meshes' draws
    /// sorted by clip scope, and their runs (`FlatScene::shadow_lists`).
    pub shadow_draws: Vec<ShadowList<DrawIndexed>>,
}

pub struct MeshPacker {
    pub vertices: Arena,
    pub indices: Arena,
    /// uid -> (mesh version, first vertex, vertex count, first index, index count).
    slots: HashMap<u64, (u64, usize, usize, usize, usize)>,
    seen: HashSet<u64>,
    pub geometry: HashMap<u64, MeshGeometry>,
    materials: HashMap<u64, MaterialEntry>,
}

impl MeshPacker {
    pub fn new(py: Python<'_>) -> MeshPacker {
        MeshPacker {
            vertices: Arena::new(py, records::MESH_VERTEX_SIZE),
            indices: Arena::new(py, 4),
            slots: HashMap::new(),
            seen: HashSet::new(),
            geometry: HashMap::new(),
            materials: HashMap::new(),
        }
    }

    pub fn too_much_garbage(&self) -> bool {
        self.vertices.too_much_garbage(1 << 16) || self.indices.too_much_garbage(1 << 16)
    }

    /// Start over with empty arenas (keeps the per-node caches).
    pub fn repack(&mut self, py: Python<'_>) {
        self.vertices = Arena::new(py, records::MESH_VERTEX_SIZE);
        self.indices = Arena::new(py, 4);
        self.slots.clear();
    }

    /// The mesh of node `uid` (built by Python when its version changed).
    pub fn mesh(
        &mut self,
        py: Python<'_>,
        callbacks: &Callbacks,
        values: &Values,
        uid: u64,
        fields: &MeshFields,
    ) -> PyResult<&MeshGeometry> {
        let version = values.state.versions[fields.mesh];
        if self.geometry.get(&uid).map(|g| g.version) != Some(version) {
            let mesh = values.state.kinds[fields.mesh].value_to_py(py, values.get(fields.mesh))?;
            let result = callbacks.mesh_records.bind(py).call1((mesh,))?;
            let corners = bytes_of(&result.get_item(3)?)?;
            let corners: &[f64] = bytemuck::cast_slice(&corners);
            let indices = bytes_of(&result.get_item(1)?)?;
            self.geometry.insert(
                uid,
                MeshGeometry {
                    version,
                    vertices: bytes_of(&result.get_item(0)?)?,
                    index_count: (indices.len() / 4) as u32,
                    indices,
                    triangle_count: result.get_item(2)?.extract()?,
                    corners: corners
                        .chunks_exact(3)
                        .map(|c| DVec3::new(c[0], c[1], c[2]))
                        .collect(),
                },
            );
        }
        Ok(&self.geometry[&uid])
    }

    fn material(
        &mut self,
        py: Python<'_>,
        callbacks: &Callbacks,
        values: &Values,
        uid: u64,
        fields: &MeshFields,
    ) -> PyResult<&MaterialEntry> {
        let versions: Vec<u64> = fields
            .material
            .iter()
            .map(|&slot| values.state.versions[slot])
            .collect();
        if self.materials.get(&uid).map(|m| &m.versions) != Some(&versions) {
            let arguments = fields
                .material
                .iter()
                .map(|&slot| values.state.kinds[slot].value_to_py(py, values.get(slot)))
                .collect::<PyResult<Vec<_>>>()?;
            let bytes = bytes_of(
                &callbacks
                    .material_entry
                    .bind(py)
                    .call1(pyo3::types::PyTuple::new(py, arguments)?)?,
            )?;
            let texture_key = fields
                .textures
                .iter()
                .chain([&fields.surface_shader])
                .map(|&slot| texture_identity(values, slot))
                .collect();
            self.materials.insert(
                uid,
                MaterialEntry {
                    versions,
                    entry: bytemuck::pod_read_unaligned(&bytes),
                    texture_key,
                },
            );
        }
        Ok(&self.materials[&uid])
    }

    /// `RetainedMeshes.place`: ``(first index, base vertex)`` of a node's mesh.
    fn place(&mut self, py: Python<'_>, uid: u64) -> (usize, usize) {
        self.seen.insert(uid);
        let geometry = &self.geometry[&uid];
        if let Some(slot) = self.slots.get(&uid) {
            if slot.0 == geometry.version {
                return (slot.3, slot.1);
            }
        }
        let old = self.slots.get(&uid).copied().unwrap_or((0, 0, 0, 0, 0));
        let first_vertex = self.vertices.place(py, &geometry.vertices, old.1, old.2);
        let first_index = self.indices.place(py, &geometry.indices, old.3, old.4);
        let vertex_count = geometry.vertices.len() / records::MESH_VERTEX_SIZE;
        self.slots.insert(
            uid,
            (
                geometry.version,
                first_vertex,
                vertex_count,
                first_index,
                geometry.index_count as usize,
            ),
        );
        (first_index, first_vertex)
    }

    /// `RetainedMeshes.finish`: free the ranges of meshes not placed since
    /// the last call.
    pub fn finish(&mut self) {
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

    /// `MeshBatch.build` without the lights.
    pub fn pack(
        &mut self,
        py: Python<'_>,
        graph: &SceneGraph,
        values: &Values,
        scene: &FlatScene,
        callbacks: &Callbacks,
    ) -> PyResult<MeshOutput> {
        let mut objects = Vec::new();
        let mut materials = Vec::new();
        let mut groups: Vec<(Vec<usize>, DrawGroup)> = Vec::new();
        let mut shadow_draws: Vec<(DrawIndexed, i32)> = Vec::new();
        // A node drawn more than once (it sits in several groups) shares
        // its material entry while its alpha is the same.
        let mut material_slots: HashMap<(u64, u64), u32> = HashMap::new();
        for item in &scene.items {
            let node = &graph.nodes[item.node];
            let NodeData::Mesh(fields) = &node.data else {
                continue;
            };
            if self
                .mesh(py, callbacks, values, node.uid, fields)?
                .triangle_count
                == 0
            {
                continue;
            }
            let mut entry = self
                .material(py, callbacks, values, node.uid, fields)?
                .entry;
            if item.alpha != 1.0 {
                // The item's alpha (Node.alpha, times every one above) fades the material.
                entry.base_color[3] *= item.alpha as f32;
            }
            let first_object = objects.len() as u32;
            let count = match fields.particles {
                Some(particles) => {
                    let placed = particle_objects(
                        values,
                        particles,
                        &item.world,
                        entry,
                        materials.len() as u32,
                        item.clip,
                    )?;
                    if placed.0.is_empty() {
                        continue;
                    }
                    let count = placed.0.len();
                    objects.extend(placed.0);
                    materials.extend(placed.1);
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
            let texture_key = self.materials[&node.uid].texture_key.clone();
            let group = match groups.iter().position(|(key, _)| *key == texture_key) {
                Some(index) => index,
                None => {
                    let textures = fields
                        .textures
                        .iter()
                        .map(|&slot| values.state.kinds[slot].value_to_py(py, values.get(slot)))
                        .collect::<PyResult<Vec<_>>>()?;
                    let slot = fields.surface_shader;
                    let surface_shader =
                        values.state.kinds[slot].value_to_py(py, values.get(slot))?;
                    groups.push((
                        texture_key,
                        DrawGroup {
                            textures,
                            surface_shader,
                            draws: Vec::new(),
                        },
                    ));
                    groups.len() - 1
                }
            };
            let (first_index, base_vertex) = self.place(py, node.uid);
            let draw = DrawIndexed {
                index_count: self.geometry[&node.uid].index_count,
                instance_count: count as u32,
                first_index: first_index as u32,
                base_vertex: base_vertex as i32,
                first_instance: first_object,
            };
            groups[group].1.draws.push(draw);
            if values.truthy(fields.cast_shadows)? {
                shadow_draws.push((draw, item.clip));
            }
        }
        self.finish();
        Ok(MeshOutput {
            objects,
            materials,
            groups: groups.into_iter().map(|(_, group)| group).collect(),
            shadow_draws: scene.shadow_lists(&shadow_draws),
        })
    }
}

/// A texture's identity, 0 for none (draws are grouped by texture sets).
fn texture_identity(values: &Values, slot: usize) -> usize {
    match values.get(slot) {
        crate::value::Value::Object(object) if !object.is_none(values.py) => {
            object.as_ptr() as usize
        }
        _ => 0,
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

/// One object and one material per shown particle (scale non-zero on
/// every axis): `_particle_matrices` and `_tinted`.
fn particle_objects(
    values: &Values,
    slots: [usize; 4],
    world: &DMat4,
    entry: Material,
    first_material: u32,
    clip: i32,
) -> PyResult<(Vec<MeshObject>, Vec<Material>)> {
    let [positions, rotations, scales, colors] = slots.map(|slot| values.get(slot).array());
    let (positions, rotations, scales, colors) = (positions?, rotations?, scales?, colors?);
    let mut objects = Vec::new();
    let mut materials = Vec::new();
    for i in 0..positions.rows() {
        let scale = DVec3::from_slice(&scales.data[3 * i..3 * i + 3]);
        if scale.abs().min_element() <= 1e-9 {
            continue;
        }
        let r = &rotations.data[4 * i..4 * i + 4]; // w, x, y, z; interpolated linearly, so normalized here
        let rotation = DQuat::from_xyzw(r[1], r[2], r[3], r[0]).normalize();
        let local = DMat4::from_scale_rotation_translation(
            scale,
            rotation,
            DVec3::from_slice(&positions.data[3 * i..3 * i + 3]),
        );
        objects.push(mesh_object(
            &(*world * local),
            first_material + materials.len() as u32,
            clip,
        ));
        let color = &colors.data[4 * i..4 * i + 4];
        let mut tinted = entry;
        for channel in 0..3 {
            tinted.base_color[channel] =
                (tinted.base_color[channel] as f64 * color[channel]) as f32;
            tinted.emissive[channel] =
                (tinted.emissive[channel] as f64 * (color[channel] * color[3])) as f32;
        }
        materials.push(tinted);
    }
    Ok((objects, materials))
}
