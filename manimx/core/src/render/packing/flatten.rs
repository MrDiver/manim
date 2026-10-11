//! Flattening a scene: the drawable leaves with their world matrices,
//! painter's order, clip scopes and alphas, and the lights (a
//! `FlatScene`). The first step of packing a frame.
//!
//! The tree's rules: a node scaled to nothing is skipped with everything
//! below it; a clipping group whose mask or content is scaled to nothing
//! shows nothing; alphas multiply down the tree; a light inside a
//! clipping group's content belongs to that clip scope.

use glam::DMat4;

use crate::render::error::{Error, Result};
use crate::robjects::{Kind, RNode};

/// Clip index of something not inside any clipping group.
pub const NO_CLIP: i32 = -1;

/// One drawable leaf with everything resolved from its ancestors.
pub struct DrawItem<'a> {
    pub node: &'a RNode,
    pub world: DMat4,
    /// Position in depth-first tree order (painter's order).
    pub order: usize,
    pub clip: i32,
    pub alpha: f64,
}

/// The mask of one clipping group.
pub struct ClipScope<'a> {
    /// Whether the mask only counts where its front faces the camera.
    pub one_sided: bool,
    pub parent: i32,
    pub mask: Vec<DrawItem<'a>>,
}

pub struct LightItem<'a> {
    pub node: &'a RNode,
    pub world: DMat4,
    pub clip: i32,
    pub alpha: f64,
}

#[derive(Default)]
pub struct FlatScene<'a> {
    pub items: Vec<DrawItem<'a>>,
    pub clips: Vec<ClipScope<'a>>,
    pub lights: Vec<LightItem<'a>>,
}

/// The shadow casters of one light scope: the scope, its casters sorted
/// by clip scope (unclipped first) and the runs `(clip scope, count)` of
/// equal clip scopes among them.
pub type ShadowList<T> = (i32, Vec<T>, Vec<(i32, u32)>);

/// The drawable leaves, clip scopes and lights of the tree under `root`.
pub fn flatten(root: &RNode) -> Result<FlatScene<'_>> {
    let mut flat = FlatScene::default();
    walk(root, &DMat4::IDENTITY, NO_CLIP, &mut flat, None, 1.0)?;
    Ok(flat)
}

fn walk<'a>(
    node: &'a RNode,
    parent_world: &DMat4,
    clip: i32,
    flat: &mut FlatScene<'a>,
    mask_of: Option<usize>,
    parent_alpha: f64,
) -> Result<()> {
    if node.transform.is_flat() {
        return Ok(());
    }
    let world = *parent_world * node.transform.matrix();
    let alpha = (parent_alpha * node.alpha).max(0.0);
    match &node.kind {
        Kind::Light(_) => {
            if mask_of.is_none() {
                flat.lights.push(LightItem {
                    node,
                    world,
                    clip,
                    alpha,
                });
            }
        }
        Kind::Shape3D(_) | Kind::Particles { .. } if mask_of.is_some() => {
            return Err(Error::Unsupported(
                "3D meshes cannot be part of a mask".into(),
            ));
        }
        Kind::Shape2D { .. } | Kind::Curve3D { .. } | Kind::Shape3D(_) | Kind::Particles { .. } => {
            let target = match mask_of {
                None => &mut flat.items,
                Some(scope) => &mut flat.clips[scope].mask,
            };
            let order = target.len();
            target.push(DrawItem {
                node,
                world,
                order,
                clip,
                alpha,
            });
        }
        Kind::Group { children } => {
            for child in children {
                walk(child, &world, clip, flat, mask_of, alpha)?;
            }
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
            let (Some(mask), Some(content)) = (mask, content) else {
                return Ok(()); // nothing to show through
            };
            if mask.transform.is_flat() || content.transform.is_flat() {
                return Ok(());
            }
            flat.clips.push(ClipScope {
                one_sided: *one_sided,
                parent: clip,
                mask: Vec::new(),
            });
            let scope = flat.clips.len() - 1;
            walk(mask, &world, clip, flat, Some(scope), alpha)?;
            walk(content, &world, scope as i32, flat, None, alpha)?;
        }
    }
    Ok(())
}

impl FlatScene<'_> {
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
