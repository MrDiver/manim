//! The packer: the scenes it is sent (`Update`s, flat/) in, the records
//! of a frame out. `Packer::apply` brings a scene's store up to date
//! (store.rs); `Packer::pack_frame` builds the vector, mesh, light and
//! shadow records of one scene (vectors.rs, meshes.rs, lights.rs,
//! shadows.rs) and returns them as a `Packet` for the passes. It never
//! touches the GPU.
//!
//! One packer lives as long as its renderer: its arenas mirror the
//! renderer's buffers, it keeps a store per scene until told to forget
//! it (by the uid of the scene's root), and its caches keep what it
//! built from each node's values while their versions hold.

use std::collections::{HashMap, HashSet};

use glam::DMat4;

use crate::error::{Error, Result};
use crate::flat::{Change, DrawItem, FlatScene, Update};
use crate::render::draw::uploads::{Records, Uploads};
use crate::render::pack::arena::Arena;
use crate::render::pack::lights;
use crate::render::pack::meshes::MeshPacker;
use crate::render::pack::shadows::{self, Bounds, FrameSettings, ShadowLayers};
use crate::render::pack::store::SceneStore;
use crate::render::pack::vectors::{
    CameraViews, Culling, VectorFrame, VectorItems, VectorPacker, DEFAULT_PAGE_BYTES,
};
use crate::render::packet::records::Light;
use crate::render::packet::{MeshPacket, Packet};

/// Packs after which the per-node data of nodes in no scene are dropped.
const CACHE_MAX_AGE: u64 = 1024;

/// One scene's store, and what the packers keep for it.
#[derive(Default)]
struct Scene {
    store: SceneStore,
    vectors: VectorItems,
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
    vectors: VectorPacker,
    meshes: MeshPacker,
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

    /// Bring the store of `update`'s scene up to date. A scene's first
    /// update must send it whole.
    pub fn apply(&mut self, update: Update) -> Result<()> {
        let scene = match self.scenes.get_mut(&update.scene) {
            Some(scene) => scene,
            None if matches!(update.changes.first(), Some(Change::Reset(_))) => {
                self.scenes.entry(update.scene).or_default()
            }
            None if update.changes.is_empty() => return Ok(()),
            None => {
                return Err(Error::Invalid(format!(
                    "scene {} was not sent whole first",
                    update.scene
                )))
            }
        };
        scene.store.apply(update);
        Ok(())
    }

    /// The revision of `scene`'s store (equal revisions promise an equal
    /// scene) and whether what it shows may change with the time alone;
    /// None for a scene not sent yet.
    pub fn state(&self, scene: u64) -> Option<(u64, bool)> {
        let store = &self.scenes.get(&scene)?.store;
        Some((store.revision(), store.uses_time()))
    }

    /// Drop everything kept for `scene`.
    pub fn forget(&mut self, scene: u64) {
        if let Some(scene) = self.scenes.remove(&scene) {
            self.vectors.drop_scene(&scene.vectors);
        }
    }

    /// The packet of `scene` (applied before) seen with `settings`.
    pub fn pack_frame(&mut self, scene: u64, settings: &PackSettings) -> Result<Packet> {
        self.generation += 1;
        if self.generation.is_multiple_of(CACHE_MAX_AGE) {
            self.sweep();
        }
        let scene = self
            .scenes
            .get_mut(&scene)
            .ok_or_else(|| Error::Invalid(format!("scene {scene} was never sent")))?;
        let changes = scene.store.take_changes();
        // No mesh in the view: no lights, shadows or 3D passes (any Shape3D
        // counts, even one without triangles).
        let has_meshes = scene.store.has_meshes();
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
            .pack(&scene.store.flat, &mut scene.vectors, &changes, &frame);
        let shown;
        let flat = if settings.hide_camera_views && has_meshes {
            shown = without_camera_views(&scene.store.flat, &settings.camera_views);
            &shown
        } else {
            &scene.store.flat
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

    /// The record buffers of the packet packed last, each with the ranges
    /// written since the last call (the drawer uploads only those); with
    /// `meshes`, the mesh vertices and indices too.
    pub fn uploads(&mut self, meshes: bool) -> Uploads<'_> {
        let vector_pages = self
            .vectors
            .pages
            .iter_mut()
            .map(|page| page.arenas_mut().map(|(_, arena)| written(arena)))
            .collect();
        let meshes = meshes.then(|| {
            let mesh = &mut self.meshes;
            [written(&mut mesh.vertices), written(&mut mesh.indices)]
        });
        Uploads {
            vector_pages,
            meshes,
        }
    }

    /// Drop the per-node data of nodes in no scene kept.
    fn sweep(&mut self) {
        let live: HashSet<u64> = self
            .scenes
            .values()
            .flat_map(|scene| scene.store.leaf_uids())
            .collect();
        self.vectors.paths.retain(|uid| live.contains(uid));
        self.meshes.geometry.retain(|uid| live.contains(uid));
        self.meshes.materials.retain(|uid| live.contains(uid));
        self.bounds.forget(|uid| live.contains(&uid));
    }
}

/// `arena`'s records and the ranges written since the last call.
fn written(arena: &mut Arena) -> Records<'_> {
    let written = arena.ranges();
    Records {
        bytes: arena.bytes(),
        record_size: arena.record_size,
        written,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::DVec3;

    use super::*;
    use crate::render::pack::shadows::FrameSettings;
    use crate::resolve::Resolver;
    use crate::robjects::{Fill, Kind, Path, Points, RNode, Stroke, Transform};

    fn tree() -> Arc<RNode> {
        let square: Vec<f64> = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
            .iter()
            .zip([(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0)])
            .flat_map(|(&(ax, ay), (bx, by))| {
                [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
                    .into_iter()
                    .flat_map(move |t| [ax + t * (bx - ax), ay + t * (by - ay)])
            })
            .collect();
        let shape = Arc::new(RNode {
            uid: 2,
            transform: Transform {
                position: DVec3::new(-0.25, -0.25, 0.0),
                ..Transform::IDENTITY
            },
            alpha: 1.0,
            kind: Kind::Shape2D {
                path: Path {
                    points: Arc::new(Points::new(square, 2)),
                    stroke: Arc::new(Stroke::new(
                        vec![[1.0; 3]],
                        vec![1.0],
                        vec![4.0],
                        vec![0],
                        0,
                    )),
                    cast_shadows: false,
                },
                fill: Arc::new(Fill::new(vec![[0.2, 0.5, 0.8]], vec![1.0])),
            },
        });
        Arc::new(RNode {
            uid: 1,
            transform: Transform::IDENTITY,
            alpha: 1.0,
            kind: Kind::Group {
                children: vec![shape],
            },
        })
    }

    fn settings() -> PackSettings {
        PackSettings {
            frame: FrameSettings {
                time: 0.0,
                layer_depth_step: 2e-7,
                exposure: 1.0,
                ambient_linear: [0.0; 3],
                shadow_softness: 1.0,
                shadow_samples: 16.0,
                shadow_resolution: 512,
                shadows: true,
                shadow_filter: [1.0, 16.0, 0.0, 0.0],
            },
            default_lights: true,
            camera: DMat4::IDENTITY,
            view_projection: DMat4::IDENTITY,
            camera_views: CameraViews::new(),
            hide_camera_views: false,
        }
    }

    #[test]
    fn a_resolved_tree_packs_and_an_unchanged_one_uploads_nothing() {
        let root = tree();
        let (mut resolver, mut packer) = (Resolver::default(), Packer::new());
        packer.apply(resolver.sync(&root).unwrap()).unwrap();
        let packet = packer.pack_frame(root.uid, &settings()).unwrap();
        // The fill is opaque; a stroke is always blended (its edges fade).
        assert_eq!(packet.draws.opaque.runs, vec![(0, 1)]);
        assert_eq!(packet.draws.transparent.runs, vec![(0, 1)]);
        assert!(packet.meshes.is_none()); // no mesh: no 3D passes
        assert!(packer.uploads(false).vector_pages[0]
            .iter()
            .any(|records| records.written.is_none()));

        let again = resolver.sync(&root).unwrap();
        assert!(again.changes.is_empty());
        packer.apply(again).unwrap();
        let packet = packer.pack_frame(root.uid, &settings()).unwrap();
        assert_eq!(packet.draws.opaque.runs, vec![(0, 1)]);
        assert_eq!(packet.draws.transparent.runs, vec![(0, 1)]);
        for records in packer.uploads(false).vector_pages[0].iter() {
            assert_eq!(records.written, Some(Vec::new()));
        }
    }
}
