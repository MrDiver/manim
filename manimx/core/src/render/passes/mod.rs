//! The render passes of one view, in the order the renderer runs them
//! (renderer.rs):
//!
//! | Module | What it draws |
//! |---|---|
//! | `camera_views` | the pictures of camera views, before the frame |
//! | `vector` | clip masks, 2D shapes and 3D curves (and their shadows) |
//! | `shadow` | shadow maps: the casters' depth seen from each light |
//! | `mesh` | meshes into the G-buffer (the geometry pass) |
//! | `lighting` | the G-buffer lit by every light (deferred lighting) |

pub mod camera_views;
pub mod lighting;
pub mod mesh;
pub mod shadow;
pub mod vector;
