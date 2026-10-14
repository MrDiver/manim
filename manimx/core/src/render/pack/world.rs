//! The render world: the scene the renderer keeps from frame to frame.
//!
//! A frame used to be packed from its render object tree alone: every
//! frame flattened the whole tree and visited every leaf, so a frame cost
//! as much as the scene was large, however little of it changed. A
//! `RenderWorld` keeps the flattened scene (`FlatScene`) of one tree
//! between frames, with a mirror of the tree it came from, and updates it
//! in place: what packing does for a frame then depends on what changed.
//!
//! How it learns what changed: `RenderWorld::sync` compares a new tree
//! with the mirror, node by node in tree order. A subtree that is the
//! same object as last time (`Arc::ptr_eq`; the evaluator reuses every
//! leaf whose signals kept their versions) under an unchanged parent is
//! skipped whole. Otherwise:
//!
//! - same structure (the same uids, kinds and children, nothing newly
//!   scaled to nothing or back): the transforms and alphas are carried
//!   down, and each leaf whose world matrix, alpha or value changed is
//!   written into its place in the flat scene and reported as changed
//!   (`Changes`);
//! - a new structure: the tree is flattened again from scratch and every
//!   leaf counts as changed. Adding or removing nodes is rare next to
//!   moving them.
//!
//! The rules of flattening: a node scaled to nothing is skipped with
//! everything below it; a clipping group whose mask or content is
//! missing or scaled to nothing shows nothing; alphas multiply down the
//! tree; a light inside a clipping group's content belongs to that clip
//! scope, one inside a mask lights nothing.
//!
//! Every change gives the world a new revision, from one counter for all
//! worlds: equal revisions promise an equal scene, which is how a frame
//! that shows the same as the one before is recognized without comparing
//! trees.
//!
//! Later the evaluator will send the changes itself (typed per node,
//! addressed by uid), and the comparison will only remain for trees made
//! by hand.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use glam::DMat4;

use crate::render::error::{Error, Result};
use crate::render::pack::flatten::{ClipScope, DrawItem, FlatScene, LightItem, NO_CLIP};
use crate::robjects::{Kind, RNode};

static NEXT_REVISION: AtomicU64 = AtomicU64::new(1);

/// Where a leaf of the tree is in the flat scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// `items[index]`.
    Item(usize),
    /// `clips[scope].mask[index]`.
    Mask(usize, usize),
    /// `lights[index]`.
    Light(usize),
    /// Not drawn: a group, a light inside a mask, a node scaled to
    /// nothing, or anything below one.
    Nowhere,
}

/// What changed in a world since its changes were last taken.
#[derive(Default)]
pub struct Changes {
    /// Everything: the scene was flattened again.
    pub all: bool,
    /// The leaves whose world matrix, alpha or value changed.
    pub places: Vec<Place>,
}

/// One node of the tree as the world last saw it.
struct Mirror {
    source: Arc<RNode>,
    /// Scaled to nothing: skipped, with nothing below it mirrored.
    flat: bool,
    world: DMat4,
    alpha: f64,
    place: Place,
    /// A group's children, or a clipping group's mask and content (empty
    /// when it shows nothing), as indices into `RenderWorld::mirror`.
    children: Vec<usize>,
}

pub struct RenderWorld {
    pub flat: FlatScene,
    /// The tree in depth-first order; the root first.
    mirror: Vec<Mirror>,
    changes: Changes,
    revision: u64,
    /// Whether any leaf is a mesh (else the 3D passes are skipped).
    has_meshes: bool,
    /// Leaves whose surface shader may read the time.
    time_shaders: usize,
}

impl RenderWorld {
    /// The world of the tree under `root`.
    pub fn new(root: &Arc<RNode>) -> Result<RenderWorld> {
        let mut world = RenderWorld {
            flat: FlatScene::default(),
            mirror: Vec::new(),
            changes: Changes::default(),
            revision: 0,
            has_meshes: false,
            time_shaders: 0,
        };
        world.rebuild(root)?;
        Ok(world)
    }

    /// Bring the world to the tree under `root` (see the module docs).
    pub fn sync(&mut self, root: &Arc<RNode>) -> Result<()> {
        let changed_before = self.changes.places.len();
        if !self.update(0, root, &DMat4::IDENTITY, 1.0, false)? {
            self.rebuild(root)?;
        } else if self.changes.places.len() != changed_before {
            self.revision = NEXT_REVISION.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Increases exactly when the scene changes, across all worlds.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn has_meshes(&self) -> bool {
        self.has_meshes
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

    /// Flatten the tree under `root` from scratch.
    fn rebuild(&mut self, root: &Arc<RNode>) -> Result<()> {
        self.flat = FlatScene::default();
        self.mirror.clear();
        self.walk(root, &DMat4::IDENTITY, NO_CLIP, None, 1.0)?;
        self.has_meshes = self
            .flat
            .items
            .iter()
            .any(|item| matches!(item.node.kind, Kind::Shape3D(_) | Kind::Particles { .. }));
        self.time_shaders = self
            .flat
            .items
            .iter()
            .filter(|item| uses_time(&item.node))
            .count();
        self.changes = Changes {
            all: true,
            places: Vec::new(),
        };
        self.revision = NEXT_REVISION.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Mirror `node` and flatten what it shows; returns its mirror index.
    fn walk(
        &mut self,
        node: &Arc<RNode>,
        parent_world: &DMat4,
        clip: i32,
        mask_of: Option<usize>,
        parent_alpha: f64,
    ) -> Result<usize> {
        let index = self.mirror.len();
        let flat = node.transform.is_flat();
        let world = *parent_world * node.transform.matrix();
        let alpha = (parent_alpha * node.alpha).max(0.0);
        self.mirror.push(Mirror {
            source: node.clone(),
            flat,
            world,
            alpha,
            place: Place::Nowhere,
            children: Vec::new(),
        });
        if flat {
            return Ok(index);
        }
        let place = match &node.kind {
            Kind::Light(_) if mask_of.is_some() => Place::Nowhere,
            Kind::Light(_) => {
                self.flat.lights.push(LightItem {
                    node: node.clone(),
                    world,
                    clip,
                    alpha,
                });
                Place::Light(self.flat.lights.len() - 1)
            }
            Kind::Shape3D(_) | Kind::Particles { .. } if mask_of.is_some() => {
                return Err(Error::Unsupported(
                    "3D meshes cannot be part of a mask".into(),
                ));
            }
            Kind::Shape2D { .. }
            | Kind::Curve3D { .. }
            | Kind::Shape3D(_)
            | Kind::Particles { .. } => {
                let target = match mask_of {
                    None => &mut self.flat.items,
                    Some(scope) => &mut self.flat.clips[scope].mask,
                };
                let order = target.len();
                target.push(DrawItem {
                    node: node.clone(),
                    world,
                    order,
                    clip,
                    alpha,
                });
                match mask_of {
                    None => Place::Item(order),
                    Some(scope) => Place::Mask(scope, order),
                }
            }
            Kind::Group { children } => {
                let mut indices = Vec::with_capacity(children.len());
                for child in children {
                    indices.push(self.walk(child, &world, clip, mask_of, alpha)?);
                }
                self.mirror[index].children = indices;
                Place::Nowhere
            }
            Kind::ClippingGroup {
                mask,
                content,
                one_sided,
            } => {
                if mask_of.is_some() {
                    return Err(Error::Unsupported(
                        "a clipping group inside a mask is not supported".into(),
                    ));
                }
                if let (Some(mask), Some(content)) = (mask, content) {
                    if !mask.transform.is_flat() && !content.transform.is_flat() {
                        self.flat.clips.push(ClipScope {
                            one_sided: *one_sided,
                            parent: clip,
                            mask: Vec::new(),
                        });
                        let scope = self.flat.clips.len() - 1;
                        let mask = self.walk(mask, &world, clip, Some(scope), alpha)?;
                        let content = self.walk(content, &world, scope as i32, None, alpha)?;
                        self.mirror[index].children = vec![mask, content];
                    }
                }
                Place::Nowhere // nothing to show through
            }
        };
        self.mirror[index].place = place;
        Ok(index)
    }

    /// Carry `node` into mirror `index` in place, when its structure is
    /// the same; false when it is not (the world must be rebuilt).
    fn update(
        &mut self,
        index: usize,
        node: &Arc<RNode>,
        parent_world: &DMat4,
        parent_alpha: f64,
        parent_changed: bool,
    ) -> Result<bool> {
        let mirror = &self.mirror[index];
        if !parent_changed && Arc::ptr_eq(&mirror.source, node) {
            return Ok(true);
        }
        let old = mirror.source.clone();
        if node.uid != old.uid
            || std::mem::discriminant(&node.kind) != std::mem::discriminant(&old.kind)
            || node.transform.is_flat() != mirror.flat
        {
            return Ok(false);
        }
        if mirror.flat {
            self.mirror[index].source = node.clone();
            return Ok(true); // still nothing shows
        }
        let world = *parent_world * node.transform.matrix();
        let alpha = (parent_alpha * node.alpha).max(0.0);
        let changed = parent_changed || world != mirror.world || alpha != mirror.alpha;
        let place = mirror.place;
        match (&node.kind, &old.kind) {
            (Kind::Group { children }, Kind::Group { children: before }) => {
                if children.len() != before.len() {
                    return Ok(false);
                }
                let indices = self.mirror[index].children.clone();
                for (child, &child_index) in children.iter().zip(&indices) {
                    if !self.update(child_index, child, &world, alpha, changed)? {
                        return Ok(false);
                    }
                }
            }
            (
                Kind::ClippingGroup {
                    mask,
                    content,
                    one_sided,
                },
                Kind::ClippingGroup {
                    mask: mask_before,
                    content: content_before,
                    one_sided: one_sided_before,
                },
            ) => {
                if one_sided != one_sided_before
                    || mask.is_some() != mask_before.is_some()
                    || content.is_some() != content_before.is_some()
                {
                    return Ok(false);
                }
                let shown = match (mask, content) {
                    (Some(mask), Some(content)) => (!mask.transform.is_flat()
                        && !content.transform.is_flat())
                    .then_some((mask, content)),
                    _ => None,
                };
                let indices = self.mirror[index].children.clone();
                match (shown, indices.as_slice()) {
                    (Some((mask, content)), &[mask_index, content_index]) => {
                        if !self.update(mask_index, mask, &world, alpha, changed)?
                            || !self.update(content_index, content, &world, alpha, changed)?
                        {
                            return Ok(false);
                        }
                    }
                    (None, []) => {}       // still nothing shows through
                    _ => return Ok(false), // something newly shows through, or stopped
                }
            }
            // A leaf: a new object with equal values (equal fingerprints)
            // changes nothing.
            _ if changed || !Arc::ptr_eq(&old, node) && fingerprint(&old) != fingerprint(node) => {
                if uses_time(&old) != uses_time(node) {
                    if uses_time(node) {
                        self.time_shaders += 1;
                    } else {
                        self.time_shaders -= 1;
                    }
                }
                self.place(place, node, world, alpha);
            }
            _ => self.keep(place, node),
        }
        let mirror = &mut self.mirror[index];
        mirror.source = node.clone();
        mirror.world = world;
        mirror.alpha = alpha;
        Ok(true)
    }

    /// Hold the newest object of an unchanged leaf in its place (the old
    /// one may then be freed), without reporting a change.
    fn keep(&mut self, place: Place, node: &Arc<RNode>) {
        match place {
            Place::Item(index) => self.flat.items[index].node = node.clone(),
            Place::Mask(scope, index) => self.flat.clips[scope].mask[index].node = node.clone(),
            Place::Light(index) => self.flat.lights[index].node = node.clone(),
            Place::Nowhere => {}
        }
    }

    /// Write a leaf's new value, world matrix and alpha into its place.
    fn place(&mut self, place: Place, node: &Arc<RNode>, world: DMat4, alpha: f64) {
        let item = match place {
            Place::Item(index) => &mut self.flat.items[index],
            Place::Mask(scope, index) => &mut self.flat.clips[scope].mask[index],
            Place::Light(index) => {
                let light = &mut self.flat.lights[index];
                (light.node, light.world, light.alpha) = (node.clone(), world, alpha);
                self.changes.places.push(place);
                return;
            }
            Place::Nowhere => return,
        };
        (item.node, item.world, item.alpha) = (node.clone(), world, alpha);
        self.changes.places.push(place);
    }
}

/// A leaf's values as `RNode::fingerprint` keys them.
fn fingerprint(node: &RNode) -> Vec<u64> {
    let mut key = Vec::new();
    node.fingerprint(&mut key);
    key
}

/// Whether a leaf's surface shader may read the time.
fn uses_time(node: &RNode) -> bool {
    matches!(&node.kind, Kind::Shape3D(surface) | Kind::Particles { surface, .. } if surface.material.surface_shader.is_some())
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::robjects::{Fill, Path, Points, Stroke, Transform};

    fn shape(uid: u64, x: f64) -> Arc<RNode> {
        scaled_shape(uid, x, 1.0)
    }

    fn scaled_shape(uid: u64, x: f64, scale: f64) -> Arc<RNode> {
        Arc::new(RNode {
            uid,
            transform: Transform {
                position: DVec3::new(x, 0.0, 0.0),
                scale: DVec3::splat(scale),
                ..Transform::IDENTITY
            },
            alpha: 1.0,
            kind: Kind::Shape2D {
                path: Path {
                    points: Arc::new(Points::new(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 2)),
                    stroke: Arc::new(Stroke::new(
                        vec![[1.0; 3]],
                        vec![1.0],
                        vec![1.0],
                        vec![0],
                        0,
                    )),
                    cast_shadows: false,
                },
                fill: Arc::new(Fill::new(vec![[1.0; 3]], vec![1.0])),
            },
        })
    }

    fn group(uid: u64, x: f64, children: Vec<Arc<RNode>>) -> Arc<RNode> {
        Arc::new(RNode {
            uid,
            transform: Transform {
                position: DVec3::new(x, 0.0, 0.0),
                ..Transform::IDENTITY
            },
            alpha: 1.0,
            kind: Kind::Group { children },
        })
    }

    #[test]
    fn the_same_tree_changes_nothing() {
        let (a, b) = (shape(2, 0.0), shape(3, 1.0));
        let root = group(1, 0.0, vec![a.clone(), b.clone()]);
        let mut world = RenderWorld::new(&root).unwrap();
        assert!(world.take_changes().all);
        let revision = world.revision();
        world.sync(&root).unwrap();
        world.sync(&group(1, 0.0, vec![a, b])).unwrap(); // a new group of the same leaves
        let changes = world.take_changes();
        assert!(!changes.all && changes.places.is_empty());
        assert_eq!(world.revision(), revision);
    }

    #[test]
    fn a_moved_group_moves_only_what_it_holds() {
        let (a, b, c) = (shape(3, 0.0), shape(4, 1.0), shape(5, 2.0));
        let mut world = RenderWorld::new(&group(
            1,
            0.0,
            vec![group(2, 0.0, vec![a.clone(), b.clone()]), c.clone()],
        ))
        .unwrap();
        world.take_changes();
        let revision = world.revision();
        world
            .sync(&group(1, 0.0, vec![group(2, 5.0, vec![a, b]), c]))
            .unwrap();
        let changes = world.take_changes();
        assert!(!changes.all);
        assert_eq!(changes.places, vec![Place::Item(0), Place::Item(1)]);
        assert_eq!(world.flat.items[1].world.w_axis.x, 6.0);
        assert_eq!(world.flat.items[2].world.w_axis.x, 2.0);
        assert!(world.revision() > revision);
    }

    #[test]
    fn a_new_structure_flattens_again() {
        let (a, b) = (shape(2, 0.0), shape(3, 1.0));
        let mut world = RenderWorld::new(&group(1, 0.0, vec![a.clone()])).unwrap();
        world.take_changes();
        world.sync(&group(1, 0.0, vec![a.clone(), b])).unwrap(); // added
        assert!(world.take_changes().all);
        assert_eq!(world.flat.items.len(), 2);
        world
            .sync(&group(
                1,
                0.0,
                vec![scaled_shape(2, 0.0, 0.0), shape(3, 1.0)],
            ))
            .unwrap(); // scaled to nothing
        assert!(world.take_changes().all);
        assert_eq!(world.flat.items.len(), 1);
        world
            .sync(&group(
                1,
                0.0,
                vec![scaled_shape(2, 0.0, 0.0), shape(3, 2.0)],
            ))
            .unwrap(); // still nothing: in place
        let changes = world.take_changes();
        assert!(!changes.all && changes.places == vec![Place::Item(0)]);
    }
}
