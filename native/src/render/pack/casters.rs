//! Shadow casters grouped for the shadow pass: per light scope, the
//! casters it lights, sorted by clip scope so each clip scope's casters
//! are drawn as one run.

use crate::flat::{FlatScene, NO_CLIP};
use crate::render::packet::ShadowList;

/// The shadow casters `(draw, clip)` of each light scope of `scene`,
/// NO_CLIP first: every caster in the scope or one nested in it, sorted
/// stably by clip scope (so painter's order holds within one), and the
/// runs `(clip scope, count)` of equal clip scopes among them.
pub fn shadow_lists<T: Copy>(scene: &FlatScene, casters: &[(T, i32)]) -> Vec<ShadowList<T>> {
    std::iter::once(NO_CLIP)
        .chain(0..scene.clips.len() as i32)
        .map(|scope| {
            let mut chosen: Vec<(T, i32)> = casters
                .iter()
                .copied()
                .filter(|(_, clip)| scene.in_scope(*clip, scope))
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
