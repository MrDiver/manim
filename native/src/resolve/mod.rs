//! Resolve: render object trees turned into scenes in world space, sent
//! as changes (`Update`s, flat/).
//!
//! ```text
//! render object tree ──Resolver::sync──▶ Update ──▶ a renderer
//! ```
//!
//! A tree says where each node is relative to its parent; a renderer
//! wants every leaf in world space and only what changed. A `Resolver`
//! keeps a mirror of the last tree of one scene and compares each new
//! tree with it, node by node in tree order. A subtree that is the same
//! object as last time (`Arc::ptr_eq`; a scene side that reuses the
//! render objects of what did not change makes this the common case)
//! under an unchanged parent is skipped whole. Otherwise:
//!
//! - same structure (the same uids, kinds and children, nothing newly
//!   scaled to nothing or back): the transforms and alphas are carried
//!   down, and each leaf whose world matrix, alpha or value changed is
//!   sent in its place (`Change::Set`);
//! - nodes added at the end of the root (a scene's newest nodes come
//!   last): they are resolved on their own and sent to go after
//!   everything else (`Change::Append`); nothing before them moves;
//! - any other new structure: the tree is resolved again from scratch
//!   and sent whole (`Change::Reset`).
//!
//! The rules of resolving: a node scaled to nothing is skipped with
//! everything below it; a clipping group whose mask or content is
//! missing or scaled to nothing shows nothing; alphas multiply down the
//! tree; a light inside a clipping group's content belongs to that clip
//! scope, one inside a mask lights nothing.
//!
//! Every change gives the scene a new revision, from one counter for all
//! scenes: equal revisions promise an equal scene, which is how a frame
//! that shows the same as the one before is recognized without comparing
//! trees.
//!
//! This module knows the render objects and the flat scene, and nothing
//! of how scenes are made or drawn.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use glam::DMat4;

use crate::flat::{Change, ClipScope, DrawItem, FlatScene, LightItem, Place, Update, NO_CLIP};
use crate::robjects::{Kind, RNode};

static NEXT_REVISION: AtomicU64 = AtomicU64::new(1);

/// Syncs after which a scene not synced since is dropped (`Resolvers`).
const MAX_AGE: u64 = 1024;

/// A tree no renderer can show: a mesh in a mask, a clipping group in a
/// mask.
#[derive(Debug)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unsupported {}

pub type Result<T> = std::result::Result<T, Unsupported>;

/// How many items, clip scopes and lights a scene has.
#[derive(Clone, Copy, Default)]
struct Sizes {
    items: usize,
    clips: usize,
    lights: usize,
}

/// One node of the tree as the resolver last saw it.
struct Mirror {
    source: Arc<RNode>,
    /// Scaled to nothing: skipped, with nothing below it mirrored.
    flat: bool,
    world: DMat4,
    alpha: f64,
    /// Where it is in the scene; None when it is not drawn: a group, a
    /// light inside a mask, a node scaled to nothing, or anything below
    /// one.
    place: Option<Place>,
    /// A group's children, or a clipping group's mask and content (empty
    /// when it shows nothing), as indices into `Resolver::mirror`.
    children: Vec<usize>,
}

/// The leaves resolved by one walk, to go after `base`.
struct Resolved {
    flat: FlatScene,
    base: Sizes,
}

impl Resolved {
    fn new(base: Sizes) -> Resolved {
        Resolved {
            flat: FlatScene::default(),
            base,
        }
    }

    fn sizes(&self) -> Sizes {
        Sizes {
            items: self.base.items + self.flat.items.len(),
            clips: self.base.clips + self.flat.clips.len(),
            lights: self.base.lights + self.flat.lights.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.flat.items.is_empty() && self.flat.clips.is_empty() && self.flat.lights.is_empty()
    }
}

/// One scene's resolver (see the module docs).
#[derive(Default)]
pub struct Resolver {
    /// The tree in depth-first order; the root first. Empty before the
    /// first sync.
    mirror: Vec<Mirror>,
    sizes: Sizes,
    changes: Vec<Change>,
    revision: u64,
}

impl Resolver {
    /// The changes that bring the scene to the tree under `root`. The
    /// first sync sends the whole scene.
    pub fn sync(&mut self, root: &Arc<RNode>) -> Result<Update> {
        if self.mirror.is_empty() || !self.update(0, root, &DMat4::IDENTITY, 1.0, false)? {
            self.rebuild(root)?;
        }
        if !self.changes.is_empty() {
            self.revision = NEXT_REVISION.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Update {
            scene: root.uid,
            revision: self.revision,
            changes: std::mem::take(&mut self.changes),
        })
    }

    /// Resolve the tree under `root` from scratch.
    fn rebuild(&mut self, root: &Arc<RNode>) -> Result<()> {
        self.mirror.clear();
        let mut resolved = Resolved::new(Sizes::default());
        self.walk(root, &DMat4::IDENTITY, NO_CLIP, None, 1.0, &mut resolved)?;
        self.sizes = resolved.sizes();
        self.changes.clear(); // the whole scene replaces whatever was found before
        self.changes.push(Change::Reset(resolved.flat));
        Ok(())
    }

    /// Resolve `children`, new at the end of the root (mirror `root`),
    /// to go after everything else: nothing before them moves.
    fn append(
        &mut self,
        root: usize,
        children: &[Arc<RNode>],
        world: &DMat4,
        alpha: f64,
    ) -> Result<()> {
        let mut resolved = Resolved::new(self.sizes);
        for child in children {
            let added = self.walk(child, world, NO_CLIP, None, alpha, &mut resolved)?;
            self.mirror[root].children.push(added);
        }
        self.sizes = resolved.sizes();
        if !resolved.is_empty() {
            self.changes.push(Change::Append(resolved.flat));
        }
        Ok(())
    }

    /// Mirror `node` and resolve what it shows into `resolved`; returns
    /// its mirror index.
    fn walk(
        &mut self,
        node: &Arc<RNode>,
        parent_world: &DMat4,
        clip: i32,
        mask_of: Option<usize>,
        parent_alpha: f64,
        resolved: &mut Resolved,
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
            place: None,
            children: Vec::new(),
        });
        if flat {
            return Ok(index);
        }
        let place = match &node.kind {
            Kind::Light(_) if mask_of.is_some() => None,
            Kind::Light(_) => {
                let place = Place::Light(resolved.sizes().lights);
                resolved.flat.lights.push(LightItem {
                    node: node.clone(),
                    world,
                    clip,
                    alpha,
                });
                Some(place)
            }
            Kind::Shape3D(_) | Kind::Particles { .. } if mask_of.is_some() => {
                return Err(Unsupported("3D meshes cannot be part of a mask".into()));
            }
            Kind::Shape2D { .. }
            | Kind::Curve3D { .. }
            | Kind::Shape3D(_)
            | Kind::Particles { .. } => {
                let (target, place) = match mask_of {
                    None => {
                        let order = resolved.sizes().items;
                        (&mut resolved.flat.items, Place::Item(order))
                    }
                    Some(scope) => {
                        // Only new scopes get masks: an appended walk never
                        // reaches into the scopes before it.
                        let mask = &mut resolved.flat.clips[scope - resolved.base.clips].mask;
                        let order = mask.len();
                        (mask, Place::Mask(scope, order))
                    }
                };
                let order = match place {
                    Place::Item(order) | Place::Mask(_, order) | Place::Light(order) => order,
                };
                target.push(DrawItem {
                    node: node.clone(),
                    world,
                    order,
                    clip,
                    alpha,
                });
                Some(place)
            }
            Kind::Group { children } => {
                let mut indices = Vec::with_capacity(children.len());
                for child in children {
                    indices.push(self.walk(child, &world, clip, mask_of, alpha, resolved)?);
                }
                self.mirror[index].children = indices;
                None
            }
            Kind::ClippingGroup {
                mask,
                content,
                one_sided,
            } => {
                if mask_of.is_some() {
                    return Err(Unsupported(
                        "a clipping group inside a mask is not supported".into(),
                    ));
                }
                if let (Some(mask), Some(content)) = (mask, content) {
                    if !mask.transform.is_flat() && !content.transform.is_flat() {
                        let scope = resolved.sizes().clips;
                        resolved.flat.clips.push(ClipScope {
                            one_sided: *one_sided,
                            parent: clip,
                            mask: Vec::new(),
                        });
                        let mask = self.walk(mask, &world, clip, Some(scope), alpha, resolved)?;
                        let content =
                            self.walk(content, &world, scope as i32, None, alpha, resolved)?;
                        self.mirror[index].children = vec![mask, content];
                    }
                }
                None // nothing to show through
            }
        };
        self.mirror[index].place = place;
        Ok(index)
    }

    /// Carry `node` into mirror `index` in place, when its structure is
    /// the same; false when it is not (the scene must be resolved again).
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
                // The root may gain children at its end (nodes added to the
                // scene); anything else of another length is a new structure.
                let appended = index == 0 && children.len() > before.len();
                if children.len() != before.len() && !appended {
                    return Ok(false);
                }
                // Taken out while the children are updated (a scene's root
                // may hold many thousands: not copied every frame).
                let indices = std::mem::take(&mut self.mirror[index].children);
                let mut same = Ok(true);
                for (child, &child_index) in children.iter().zip(&indices) {
                    same = self.update(child_index, child, &world, alpha, changed);
                    if !matches!(same, Ok(true)) {
                        break;
                    }
                }
                self.mirror[index].children = indices;
                if !same? {
                    return Ok(false);
                }
                if appended {
                    self.append(index, &children[before.len()..], &world, alpha)?;
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
                if let Some(place) = place {
                    self.changes.push(Change::Set {
                        place,
                        node: node.clone(),
                        world,
                        alpha,
                    });
                }
            }
            _ => {}
        }
        let mirror = &mut self.mirror[index];
        mirror.source = node.clone();
        mirror.world = world;
        mirror.alpha = alpha;
        Ok(true)
    }
}

/// A leaf's values as `RNode::fingerprint` keys them.
fn fingerprint(node: &RNode) -> Vec<u64> {
    let mut key = Vec::new();
    node.fingerprint(&mut key);
    key
}

/// The resolvers of every scene shown, by the uid of the scene's root.
#[derive(Default)]
pub struct Resolvers {
    /// Each scene's resolver and the sync in which it was last synced.
    scenes: HashMap<u64, (Resolver, u64)>,
    syncs: u64,
}

impl Resolvers {
    /// The changes that bring the scene under `root` to it.
    pub fn sync(&mut self, root: &Arc<RNode>) -> Result<Update> {
        self.syncs += 1;
        let (resolver, last) = self.scenes.entry(root.uid).or_default();
        *last = self.syncs;
        resolver.sync(root)
    }

    /// Drop the scenes not synced for `MAX_AGE` syncs; returns them, for
    /// the renderers to forget too (a scene that comes back is then sent
    /// whole again).
    pub fn sweep(&mut self) -> Vec<u64> {
        if !self.syncs.is_multiple_of(MAX_AGE) {
            return Vec::new();
        }
        let oldest = self.syncs.saturating_sub(MAX_AGE);
        let gone: Vec<u64> = self
            .scenes
            .iter()
            .filter(|(_, (_, last))| *last < oldest)
            .map(|(scene, _)| *scene)
            .collect();
        for scene in &gone {
            self.scenes.remove(scene);
        }
        gone
    }
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

    /// The scene of a `Reset` or `Append`; panics on anything else.
    fn scene(change: &Change) -> &FlatScene {
        match change {
            Change::Reset(flat) | Change::Append(flat) => flat,
            Change::Set { .. } => panic!("a leaf in place, not a scene"),
        }
    }

    /// The places and x positions of the `Set`s of `update`.
    fn moved(update: &Update) -> Vec<(Place, f64)> {
        update
            .changes
            .iter()
            .map(|change| match change {
                Change::Set { place, world, .. } => (*place, world.w_axis.x),
                _ => panic!("expected leaves in place"),
            })
            .collect()
    }

    #[test]
    fn the_same_tree_changes_nothing() {
        let (a, b) = (shape(2, 0.0), shape(3, 1.0));
        let root = group(1, 0.0, vec![a.clone(), b.clone()]);
        let mut resolver = Resolver::default();
        let first = resolver.sync(&root).unwrap();
        assert!(matches!(first.changes.as_slice(), [Change::Reset(_)]));
        let again = resolver.sync(&root).unwrap();
        let regrouped = resolver.sync(&group(1, 0.0, vec![a, b])).unwrap(); // a new group of the same leaves
        assert!(again.changes.is_empty() && regrouped.changes.is_empty());
        assert_eq!(regrouped.revision, first.revision);
    }

    #[test]
    fn a_moved_group_moves_only_what_it_holds() {
        let (a, b, c) = (shape(3, 0.0), shape(4, 1.0), shape(5, 2.0));
        let mut resolver = Resolver::default();
        let first = resolver
            .sync(&group(
                1,
                0.0,
                vec![group(2, 0.0, vec![a.clone(), b.clone()]), c.clone()],
            ))
            .unwrap();
        let update = resolver
            .sync(&group(1, 0.0, vec![group(2, 5.0, vec![a, b]), c]))
            .unwrap();
        assert_eq!(
            moved(&update),
            vec![(Place::Item(0), 5.0), (Place::Item(1), 6.0)]
        );
        assert!(update.revision > first.revision);
    }

    #[test]
    fn nodes_added_at_the_end_go_after_everything_else() {
        let (a, b, c) = (shape(3, 0.0), shape(4, 1.0), shape(5, 2.0));
        let mut resolver = Resolver::default();
        let first = resolver
            .sync(&group(1, 0.0, vec![group(2, 0.0, vec![a.clone()])]))
            .unwrap();
        let update = resolver
            .sync(&group(
                1,
                0.0,
                vec![group(2, 0.0, vec![a.clone()]), b.clone()],
            ))
            .unwrap();
        let [Change::Append(added)] = update.changes.as_slice() else {
            panic!("expected one append");
        };
        assert_eq!((added.items.len(), added.items[0].order), (1, 1));
        assert!(update.revision > first.revision);
        // Added inside a group, not at the end of the root: resolved again.
        let update = resolver
            .sync(&group(1, 0.0, vec![group(2, 0.0, vec![a, c]), b]))
            .unwrap();
        let [Change::Reset(flat)] = update.changes.as_slice() else {
            panic!("expected the whole scene");
        };
        assert_eq!(flat.items.len(), 3);
    }

    #[test]
    fn a_new_structure_resolves_again() {
        let (a, b) = (shape(2, 0.0), shape(3, 1.0));
        let mut resolver = Resolver::default();
        resolver
            .sync(&group(1, 0.0, vec![a.clone(), b.clone()]))
            .unwrap();
        let reordered = resolver
            .sync(&group(1, 0.0, vec![b.clone(), a.clone()]))
            .unwrap();
        assert_eq!(scene(&reordered.changes[0]).items.len(), 2);
        let hidden = resolver
            .sync(&group(
                1,
                0.0,
                vec![scaled_shape(2, 0.0, 0.0), shape(3, 1.0)],
            ))
            .unwrap(); // scaled to nothing
        assert_eq!(scene(&hidden.changes[0]).items.len(), 1);
        let update = resolver
            .sync(&group(
                1,
                0.0,
                vec![scaled_shape(2, 0.0, 0.0), shape(3, 2.0)],
            ))
            .unwrap(); // still nothing: in place
        assert_eq!(moved(&update), vec![(Place::Item(0), 2.0)]);
    }

    /// The whole scene a fresh resolver sends for `root`.
    fn resolved(root: &Arc<RNode>) -> FlatScene {
        let mut update = Resolver::default().sync(root).unwrap();
        match update.changes.pop() {
            Some(Change::Reset(flat)) => flat,
            _ => panic!("expected the whole scene"),
        }
    }

    fn clipping(uid: u64, mask: Option<Arc<RNode>>, content: Option<Arc<RNode>>) -> Arc<RNode> {
        Arc::new(RNode {
            uid,
            transform: Transform::IDENTITY,
            alpha: 1.0,
            kind: Kind::ClippingGroup {
                mask,
                content,
                one_sided: false,
            },
        })
    }

    #[test]
    fn transforms_and_alphas_multiply_down_the_tree() {
        let root = Arc::new(RNode {
            uid: 1,
            transform: Transform {
                position: DVec3::new(1.0, 0.0, 0.0),
                ..Transform::IDENTITY
            },
            alpha: 0.5,
            kind: Kind::Group {
                children: vec![shape(2, 2.0)],
            },
        });
        let flat = resolved(&root);
        let item = &flat.items[0];
        assert_eq!(
            (item.world.w_axis.x, item.alpha, item.clip),
            (3.0, 0.5, NO_CLIP)
        );
    }

    #[test]
    fn a_clipping_group_opens_a_scope_with_its_mask() {
        let flat = resolved(&clipping(1, Some(shape(2, 0.0)), Some(shape(3, 0.0))));
        assert_eq!((flat.clips.len(), flat.clips[0].parent), (1, NO_CLIP));
        assert_eq!(flat.clips[0].mask[0].node.uid, 2);
        assert_eq!((flat.items[0].node.uid, flat.items[0].clip), (3, 0));
    }

    #[test]
    fn a_clipping_group_without_content_shows_nothing() {
        let flat = resolved(&clipping(1, Some(shape(2, 0.0)), None));
        assert!(flat.clips.is_empty() && flat.items.is_empty());
    }
}
