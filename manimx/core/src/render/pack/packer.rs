//! The packer: a render object tree in, the records of a frame out.
//! `Packer::pack_frame` brings the scene's render world up to the tree
//! (world.rs), builds the vector, mesh, light and shadow records
//! (vectors.rs, meshes.rs, lights.rs, shadows.rs) and returns them as a
//! `Packet` for the passes. It never touches the GPU.
//!
//! One packer lives as long as its renderer: its arenas mirror the
//! renderer's buffers, it keeps a render world per scene it packs (by the
//! uid of the tree's root), and its caches keep what it built from each
//! node's values while their versions hold.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::DMat4;

use crate::render::error::Result;
use crate::render::pack::flatten::{DrawItem, FlatScene};
use crate::render::pack::lights;
use crate::render::pack::meshes::{MeshOutput, MeshPacker};
use crate::render::pack::records::Light;
use crate::render::pack::shadows::{self, Bounds, FrameSettings, ShadowLayers, ShadowViews};
use crate::render::pack::vectors::{
    CameraViews, Culling, VectorDraws, VectorFrame, VectorItems, VectorPacker, DEFAULT_PAGE_BYTES,
};
use crate::render::pack::world::RenderWorld;
use crate::robjects::RNode;

/// Packs after which a scene not packed since, and the per-node data of
/// nodes in no scene, are dropped.
const CACHE_MAX_AGE: u64 = 1024;

/// One scene's render world, and what the packers keep for it.
struct Scene {
    world: RenderWorld,
    vectors: VectorItems,
    /// The pack in which it was last packed.
    last_packed: u64,
}

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
    /// World to clip space: what the picture shows (culling, vectors.rs).
    pub view_projection: DMat4,
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
    /// By the uid of the scene's root.
    scenes: HashMap<u64, Scene>,
}

impl Packer {
    pub fn new() -> Packer {
        Packer::with_page_bytes(DEFAULT_PAGE_BYTES)
    }

    /// A packer whose vector record pages hold at most `budget` bytes per
    /// arena (the GPU's largest storage binding; see pack/vectors.rs).
    pub fn with_page_bytes(budget: usize) -> Packer {
        Packer {
            vectors: VectorPacker::new(budget),
            meshes: MeshPacker::new(),
            bounds: Bounds::default(),
            shadow_layers: ShadowLayers::default(),
            generation: 0,
            scenes: HashMap::new(),
        }
    }

    /// Bring the render world of the scene under `root` up to it; returns
    /// the world's revision (equal revisions promise an equal scene) and
    /// whether what it shows may change with the time alone.
    pub fn sync(&mut self, root: &Arc<RNode>) -> Result<(u64, bool)> {
        let scene = match self.scenes.get_mut(&root.uid) {
            Some(scene) => {
                scene.world.sync(root)?;
                scene
            }
            None => self.scenes.entry(root.uid).or_insert(Scene {
                world: RenderWorld::new(root)?,
                vectors: VectorItems::default(),
                last_packed: self.generation,
            }),
        };
        Ok((scene.world.revision(), scene.world.uses_time()))
    }

    pub fn pack_frame(&mut self, root: &Arc<RNode>, settings: &PackSettings) -> Result<Packet> {
        self.generation += 1;
        if self.generation.is_multiple_of(CACHE_MAX_AGE) {
            self.sweep();
        }
        self.sync(root)?;
        let scene = self.scenes.get_mut(&root.uid).expect("just synced");
        scene.last_packed = self.generation;
        let changes = scene.world.take_changes();
        // No mesh in the view: no lights, shadows or 3D passes (any Shape3D
        // counts, even one without triangles).
        let has_meshes = scene.world.has_meshes();
        let frame = VectorFrame {
            view: settings.camera.inverse(),
            // Shapes cast shadows only onto meshes: without any, a caster
            // out of view casts nothing into it.
            culling: Culling::new(settings.view_projection, has_meshes),
            camera_views: &settings.camera_views,
            hide_camera_views: settings.hide_camera_views,
        };
        let draws = self
            .vectors
            .pack(&scene.world, &mut scene.vectors, &changes, &frame);
        let shown;
        let flat = if settings.hide_camera_views && has_meshes {
            shown = without_camera_views(&scene.world.flat, &settings.camera_views);
            &shown
        } else {
            &scene.world.flat
        };
        let scope_parents = flat.clips.iter().map(|scope| scope.parent).collect();

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
        let output = self.meshes.pack(flat, &settings.camera.inverse());
        let sources = lights::light_sources(flat, settings.default_lights, &settings.camera);
        let mut rows: Vec<Light> = sources.iter().map(|light| light.record()).collect();
        let mut shadow_views = shadows::build_shadow_views(
            flat,
            &sources,
            &mut rows,
            &mut self.bounds,
            &self.meshes,
            &settings.frame,
        );
        self.shadow_layers.mark(flat, &mut shadow_views);
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

    /// Drop the scenes not packed for `CACHE_MAX_AGE` packs, and the
    /// per-node data of nodes in none of the others.
    fn sweep(&mut self) {
        let oldest = self.generation.saturating_sub(CACHE_MAX_AGE);
        let vectors = &mut self.vectors;
        self.scenes.retain(|_, scene| {
            let keep = scene.last_packed >= oldest;
            if !keep {
                vectors.drop_world(&scene.vectors);
            }
            keep
        });
        let live: HashSet<u64> = self
            .scenes
            .values()
            .flat_map(|scene| scene.world.leaf_uids())
            .collect();
        self.vectors.paths.retain(|uid, _| live.contains(uid));
        self.meshes.geometry.retain(|uid, _| live.contains(uid));
        self.meshes.materials.retain(|uid, _| live.contains(uid));
        self.bounds.forget(|uid| live.contains(&uid));
    }
}

/// `flat` without the shapes of camera views (while a picture of one is
/// rendered), for the mesh passes and shadows.
fn without_camera_views(flat: &FlatScene, views: &CameraViews) -> FlatScene {
    let shown = |item: &&DrawItem| !views.contains_key(&item.node.uid);
    FlatScene {
        items: flat.items.iter().filter(shown).cloned().collect(),
        clips: flat.clips.clone(),
        lights: flat.lights.clone(),
    }
}

impl Default for Packer {
    fn default() -> Self {
        Packer::new()
    }
}
