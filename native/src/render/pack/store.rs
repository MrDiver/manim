//! What the renderer keeps of each scene: the scene in world space
//! (`FlatScene`, flat/), brought up to date by the `Update`s it is sent,
//! and what changed since it was last packed (`Changes`).

use crate::flat::{Change, FlatScene, Place, Update};
use crate::robjects::{Kind, RNode};

/// What changed in a scene since it was last packed.
#[derive(Default)]
pub struct Changes {
    /// Everything: the scene was sent whole.
    pub all: bool,
    /// The leaves whose world matrix, alpha or value changed, or that
    /// were added.
    pub places: Vec<Place>,
}

#[derive(Default)]
pub struct SceneStore {
    pub flat: FlatScene,
    changes: Changes,
    revision: u64,
    /// Leaves that are meshes (else the 3D passes are skipped).
    meshes: usize,
    /// Leaves whose surface shader may read the time.
    time_shaders: usize,
}

impl SceneStore {
    /// Apply the changes of `update`, in order.
    pub fn apply(&mut self, update: Update) {
        for change in update.changes {
            match change {
                Change::Reset(flat) => {
                    (self.meshes, self.time_shaders) = (0, 0);
                    for item in &flat.items {
                        self.count(&item.node, 1);
                    }
                    self.flat = flat;
                    self.changes = Changes {
                        all: true,
                        places: Vec::new(),
                    };
                }
                Change::Append(added) => {
                    let (items, lights) = (self.flat.items.len(), self.flat.lights.len());
                    let scopes = self.flat.clips.len();
                    for item in &added.items {
                        self.count(&item.node, 1);
                    }
                    self.flat.items.extend(added.items);
                    self.flat.clips.extend(added.clips);
                    self.flat.lights.extend(added.lights);
                    let flat = &self.flat;
                    let places = (items..flat.items.len())
                        .map(Place::Item)
                        .chain((scopes..flat.clips.len()).flat_map(|scope| {
                            (0..flat.clips[scope].mask.len())
                                .map(move |index| Place::Mask(scope, index))
                        }))
                        .chain((lights..flat.lights.len()).map(Place::Light));
                    self.changes.places.extend(places);
                }
                Change::Set {
                    place,
                    node,
                    world,
                    alpha,
                } => {
                    let (old, new) = match place {
                        Place::Item(index) => {
                            let item = &mut self.flat.items[index];
                            item.world = world;
                            item.alpha = alpha;
                            (std::mem::replace(&mut item.node, node.clone()), node)
                        }
                        Place::Mask(scope, index) => {
                            let item = &mut self.flat.clips[scope].mask[index];
                            (item.world, item.alpha) = (world, alpha);
                            (std::mem::replace(&mut item.node, node.clone()), node)
                        }
                        Place::Light(index) => {
                            let light = &mut self.flat.lights[index];
                            (light.world, light.alpha) = (world, alpha);
                            (std::mem::replace(&mut light.node, node.clone()), node)
                        }
                    };
                    if matches!(place, Place::Item(_)) {
                        self.count(&old, -1);
                        self.count(&new, 1);
                    }
                    self.changes.places.push(place);
                }
            }
        }
        self.revision = update.revision;
    }

    /// Count `node` in (`by` 1) or out (-1) of the meshes and time shaders.
    fn count(&mut self, node: &RNode, by: isize) {
        if matches!(node.kind, Kind::Shape3D(_) | Kind::Particles { .. }) {
            self.meshes = self.meshes.saturating_add_signed(by);
        }
        if uses_time(node) {
            self.time_shaders = self.time_shaders.saturating_add_signed(by);
        }
    }

    /// The revision of the last update applied.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Whether any leaf is a mesh.
    pub fn has_meshes(&self) -> bool {
        self.meshes > 0
    }

    /// Whether what the scene shows may change with the time alone (a
    /// surface shader may read `frame.time`).
    pub fn uses_time(&self) -> bool {
        self.time_shaders > 0
    }

    /// What changed since the last call.
    pub fn take_changes(&mut self) -> Changes {
        std::mem::take(&mut self.changes)
    }

    /// The uid of every drawn leaf.
    pub fn leaf_uids(&self) -> impl Iterator<Item = u64> + '_ {
        let masks = self.flat.clips.iter().flat_map(|scope| &scope.mask);
        self.flat
            .items
            .iter()
            .chain(masks)
            .map(|item| item.node.uid)
    }
}

/// Whether a leaf's surface shader may read the time.
fn uses_time(node: &RNode) -> bool {
    matches!(&node.kind, Kind::Shape3D(surface) | Kind::Particles { surface, .. } if surface.material.surface_shader.is_some())
}
