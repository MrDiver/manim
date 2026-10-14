//! A flattened scene: the drawable leaves with their world matrices,
//! painter's order, clip scopes and alphas, and the lights (a
//! `FlatScene`), what packing reads. The render world (world.rs) keeps
//! one per scene, flattened from its tree by the rules described there,
//! and updates it in place from frame to frame.

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

/// The shadow casters of one light scope: the scope, its casters sorted
/// by clip scope (unclipped first) and the runs `(clip scope, count)` of
/// equal clip scopes among them.
pub type ShadowList<T> = (i32, Vec<T>, Vec<(i32, u32)>);

impl FlatScene {
    /// The shadow casters `(draw, clip)` of each light scope, NO_CLIP first.
    pub fn shadow_lists<T: Copy>(&self, casters: &[(T, i32)]) -> Vec<ShadowList<T>> {
        std::iter::once(NO_CLIP)
            .chain(0..self.clips.len() as i32)
            .map(|scope| {
                let mut chosen: Vec<(T, i32)> = casters
                    .iter()
                    .copied()
                    .filter(|(_, clip)| self.in_scope(*clip, scope))
                    .collect();
                chosen.sort_by_key(|(_, clip)| *clip); // stable: painter's order within a clip scope
                let mut runs: Vec<(i32, u32)> = Vec::new();
                for (_, clip) in &chosen {
                    match runs.last_mut() {
                        Some((last, count)) if last == clip => *count += 1,
                        _ => runs.push((*clip, 1)),
                    }
                }
                (
                    scope,
                    chosen.into_iter().map(|(draw, _)| draw).collect(),
                    runs,
                )
            })
            .collect()
    }

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
