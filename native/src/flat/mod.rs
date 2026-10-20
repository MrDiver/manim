//! The flat scene: a render object tree resolved into world space.
//!
//! ```text
//! RNode tree (robjects/) ──flatten──▶ FlatScene ──▶ a renderer
//! ```
//!
//! A tree says where each node is relative to its parent; drawing wants
//! every leaf in world space. A `FlatScene` holds the drawable leaves in
//! painter's order, each with its world matrix, alpha and clip scope, the
//! masks of its clip scopes, and its lights. Nothing in it is relative to
//! anything else: no groups, no parents of nodes, only the nesting of clip
//! scopes.
//!
//! This module depends on the render objects alone. It knows nothing of
//! how a scene is made or how a renderer draws it.

use std::sync::Arc;

use glam::DMat4;

use crate::robjects::RNode;

/// Clip index of something not inside any clipping group.
pub const NO_CLIP: i32 = -1;

/// One drawable leaf with everything resolved from its ancestors.
#[derive(Clone)]
pub struct DrawItem {
    pub node: Arc<RNode>,
    pub world: DMat4,
    /// Position in depth-first tree order (painter's order).
    pub order: usize,
    pub clip: i32,
    pub alpha: f64,
}

/// The mask of one clipping group.
#[derive(Clone)]
pub struct ClipScope {
    /// Whether the mask only counts where its front faces the camera.
    pub one_sided: bool,
    /// The clip scope this one is nested in, or `NO_CLIP`.
    pub parent: i32,
    pub mask: Vec<DrawItem>,
}

#[derive(Clone)]
pub struct LightItem {
    pub node: Arc<RNode>,
    pub world: DMat4,
    pub clip: i32,
    pub alpha: f64,
}

#[derive(Default)]
pub struct FlatScene {
    pub items: Vec<DrawItem>,
    pub clips: Vec<ClipScope>,
    pub lights: Vec<LightItem>,
}

impl FlatScene {
    /// The clip scopes whose masks a point in `clip` must be inside,
    /// outermost first.
    pub fn mask_chain(&self, mut clip: i32) -> Vec<i32> {
        let mut chain = Vec::new();
        while clip != NO_CLIP {
            chain.push(clip);
            clip = self.clips[clip as usize].parent;
        }
        chain.reverse();
        chain
    }

    /// Whether something in clip scope `clip` belongs to `scope` (the scope
    /// itself or one nested in it); everything is in NO_CLIP.
    pub fn in_scope(&self, mut clip: i32, scope: i32) -> bool {
        loop {
            if clip == scope || scope == NO_CLIP {
                return true;
            }
            if clip == NO_CLIP {
                return false;
            }
            clip = self.clips[clip as usize].parent;
        }
    }
}
