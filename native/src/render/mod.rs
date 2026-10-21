//! The wgpu renderer: scenes in world space (their `Update`s, flat/) and
//! the settings of a view in, the view's layers out. It knows nothing of
//! mobjects, trees, frames or Python.
//!
//! Pack (pack/) keeps what it needs of each scene and turns it into GPU
//! records; Draw (draw/) is given only the result, a `Packet` (packet/)
//! and its `Uploads` (draw/uploads.rs, the drawer's: it uploads them),
//! and keeps only GPU buffers, pipelines and textures.

pub mod draw;
pub mod pack;
pub mod packet;
pub mod renderer;
pub mod settings;
