//! The packet: everything the drawing stage (draw/) is given for one
//! view, and the only thing it is given besides the camera's `Frame`, the
//! targets' settings and the pictures of its camera views.
//!
//! ```text
//! pack/ (Packer) ──Packet + Uploads──▶ draw/ (Drawer) ──▶ layers
//! ```
//!
//! - `Packet`: the frame's draw lists (vector draws, mesh objects,
//!   materials and draws, lights, shadow views) and the clip scopes'
//!   nesting.
//! - records.rs: the GPU records themselves, field for field as the WGSL
//!   structs declare them.
//!
//! `Uploads`, the record buffers the draws read, each with the record
//! ranges written since the last upload, are the drawer's
//! (draw/uploads.rs): it uploads them, the packer fills them.
//!
//! The drawer keeps nothing but GPU buffers, pipelines and textures; how
//! the records came about (scenes, caches, culling) is the packer's.
//! Nothing here touches the GPU, so the packer never depends on it.

pub mod records;

use std::sync::Arc;

use crate::render::packet::records::{
    DrawIndexed, DrawIndirect, Frame, Light, Material, MeshObject, ShadowView,
};
use crate::robjects::{SurfaceShader, Texture};

/// Clip index of something not inside any clipping group (as
/// `flat::NO_CLIP`, repeated so the drawer does not depend on flat/).
pub const NO_CLIP: i32 = -1;

/// The shadow casters of one light scope: the scope, its casters sorted
/// by clip scope (unclipped first) and the runs `(clip scope, count)` of
/// equal clip scopes among them.
pub type ShadowList<T> = (i32, Vec<T>, Vec<(i32, u32)>);

/// What the passes draw for one view.
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

/// One draw list: its indirect draws, and the runs of consecutive draws
/// on one page, `(page, count)`, in order.
#[derive(Default)]
pub struct DrawList {
    pub bytes: Vec<u8>,
    pub runs: Vec<(usize, u32)>,
}

impl DrawList {
    pub fn new(draws: impl IntoIterator<Item = (DrawIndirect, usize)>) -> DrawList {
        let mut list = DrawList::default();
        for (args, page) in draws {
            list.bytes.extend_from_slice(bytemuck::bytes_of(&args));
            match list.runs.last_mut() {
                Some((last, count)) if *last == page => *count += 1,
                _ => list.runs.push((page, 1)),
            }
        }
        list
    }
}

/// One light scope's shadow casters: the scope, the casters sorted by clip
/// scope (their page runs split where clip scopes change) and the runs of
/// clip scopes `(clip, count)`.
pub type ShadowDraws = (i32, DrawList, Vec<(i32, u32)>);

/// The vector draw lists of one frame (the records are in `Uploads`).
#[derive(Default)]
pub struct VectorDraws {
    pub opaque: DrawList,
    pub transparent: DrawList,
    pub masks: Vec<DrawList>,
    /// Per light scope (NO_CLIP first), its shadow casters.
    pub shadows: Vec<ShadowDraws>,
    /// The images fills show, by layer of the image texture.
    pub images: Vec<Arc<Texture>>,
}

/// Meshes sharing one texture set and surface shader: one bind group, one
/// pipeline, one multi-draw.
pub struct DrawGroup {
    pub textures: [Option<Arc<Texture>>; 5],
    pub surface_shader: Option<Arc<SurfaceShader>>,
    /// The opaque meshes' draws (the G-buffer's).
    pub draws: Vec<DrawIndexed>,
}

/// A frame's mesh records, without the vertices and indices (`Uploads`).
pub struct MeshOutput {
    pub objects: Vec<MeshObject>,
    pub materials: Vec<Material>,
    pub groups: Vec<DrawGroup>,
    /// The translucent meshes' draws, back to front, with the index of the
    /// group whose textures and surface shader they use.
    pub translucent: Vec<(usize, DrawIndexed)>,
    /// Per light scope (NO_CLIP first), the opaque shadow-casting meshes'
    /// draws sorted by clip scope, and their runs.
    pub shadow_draws: Vec<ShadowList<DrawIndexed>>,
    /// The same for translucent casters, which tint the light.
    pub translucent_shadow_draws: Vec<ShadowList<DrawIndexed>>,
}

/// `ShadowViews`: a Frame and a ShadowView record per shadow map layer.
#[derive(Default)]
pub struct ShadowViews {
    pub frames: Vec<Frame>,
    pub views: Vec<ShadowView>,
    /// The clip scope whose casters each layer renders.
    pub scopes: Vec<i32>,
    /// Per layer, whether it may differ from the previous frame's.
    pub changed: Vec<bool>,
}
