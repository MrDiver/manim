//! The packer: a render object tree in, the records of a frame out.
//! `Packer::pack_frame` flattens the tree (flatten.rs), builds the vector, mesh, light and
//! shadow records (vectors.rs, meshes.rs, lights.rs, shadows.rs) and
//! returns them as a `Packet` for the passes. It never touches the GPU.
//!
//! One packer lives as long as its renderer: its arenas mirror the
//! renderer's buffers, and its caches keep what it built from each node's
//! values while their versions hold.

use std::collections::HashMap;

use glam::DMat4;

use crate::render::error::Result;
use crate::render::packing::flatten::flatten;
use crate::render::packing::lights;
use crate::render::packing::meshes::{MeshOutput, MeshPacker};
use crate::render::packing::records::Light;
use crate::render::packing::shadows::{self, Bounds, FrameSettings, ShadowLayers, ShadowViews};
use crate::render::packing::vectors::{CameraViews, VectorDraws, VectorPacker};
use crate::robjects::{Kind, RNode};

/// Packs after which per-node data unused since is dropped.
const CACHE_MAX_AGE: u64 = 1024;

/// What the passes need for one view besides the packer's arenas.
pub struct Packet {
    pub draws: VectorDraws,
    /// Per clip scope, the index of the enclosing one (-1 for none).
    pub scope_parents: Vec<i32>,
    /// None when the view has no mesh: lights, shadows and the 3D passes
    /// only affect meshes, so they are skipped.
    pub meshes: Option<MeshPacket>,
}

pub struct MeshPacket {
    pub output: MeshOutput,
    pub lights: Vec<Light>,
    pub shadows: ShadowViews,
}

/// How to pack one view.
pub struct PackSettings {
    pub frame: FrameSettings,
    /// Light meshes with a key light when the scene has no lights.
    pub default_lights: bool,
    /// The camera's world matrix.
    pub camera: DMat4,
    /// The view's camera views: their pictures' layers by node uid.
    pub camera_views: CameraViews,
    /// Leave the camera views' shapes out (while a picture is rendered).
    pub hide_camera_views: bool,
}

pub struct Packer {
    pub vectors: VectorPacker,
    pub meshes: MeshPacker,
    bounds: Bounds,
    shadow_layers: ShadowLayers,
    generation: u64,
    /// The pack in which each node was last drawn.
    last_used: HashMap<u64, u64>,
}

impl Packer {
    pub fn new() -> Packer {
        Packer {
            vectors: VectorPacker::new(),
            meshes: MeshPacker::new(),
            bounds: Bounds::default(),
            shadow_layers: ShadowLayers::default(),
            generation: 0,
            last_used: HashMap::new(),
        }
    }

    pub fn pack_frame(&mut self, root: &RNode, settings: &PackSettings) -> Result<Packet> {
        self.generation += 1;
        if self.generation.is_multiple_of(CACHE_MAX_AGE) {
            self.sweep();
        }
        let mut flat = flatten(root)?;
        if settings.hide_camera_views {
            let views = &settings.camera_views;
            flat.items
                .retain(|item| !views.contains_key(&item.node.uid));
        }
        for item in flat
            .items
            .iter()
            .chain(flat.clips.iter().flat_map(|scope| scope.mask.iter()))
        {
            self.last_used.insert(item.node.uid, self.generation);
        }
        let draws = self.vectors.pack(
            &flat,
            settings.camera.inverse(),
            &settings.camera_views,
            self.generation,
        );
        let scope_parents = flat.clips.iter().map(|scope| scope.parent).collect();

        // No mesh in the view: no lights, shadows or 3D passes (any Shape3D
        // counts, even one without triangles).
        let has_meshes = flat
            .items
            .iter()
            .any(|item| matches!(item.node.kind, Kind::Shape3D(_) | Kind::Particles { .. }));
        if !has_meshes {
            return Ok(Packet {
                draws,
                scope_parents,
                meshes: None,
            });
        }
        if self.meshes.too_much_garbage() {
            self.meshes.repack();
        }
        let output = self.meshes.pack(&flat, &settings.camera.inverse());
        let sources = lights::light_sources(&flat, settings.default_lights, &settings.camera);
        let mut rows: Vec<Light> = sources.iter().map(|light| light.record()).collect();
        let mut shadow_views = shadows::build_shadow_views(
            &flat,
            &sources,
            &mut rows,
            &mut self.bounds,
            &self.meshes,
            &settings.frame,
        );
        self.shadow_layers.mark(&flat, &mut shadow_views);
        Ok(Packet {
            draws,
            scope_parents,
            meshes: Some(MeshPacket {
                output,
                lights: rows,
                shadows: shadow_views,
            }),
        })
    }

    /// Drop the per-node data of nodes not drawn for `CACHE_MAX_AGE` packs.
    fn sweep(&mut self) {
        let oldest = self.generation.saturating_sub(CACHE_MAX_AGE);
        self.last_used.retain(|_, used| *used >= oldest);
        let live = &self.last_used;
        self.vectors.paths.retain(|uid, _| live.contains_key(uid));
        self.meshes.geometry.retain(|uid, _| live.contains_key(uid));
        self.meshes
            .materials
            .retain(|uid, _| live.contains_key(uid));
        self.bounds.forget(|uid| live.contains_key(&uid));
    }
}

impl Default for Packer {
    fn default() -> Self {
        Packer::new()
    }
}
