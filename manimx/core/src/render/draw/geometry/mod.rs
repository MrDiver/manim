//! What the passes draw: per kind of render object its GPU buffers,
//! pipelines and bind groups, and how it draws in each role (color, mask,
//! shadow, G-buffer). `VectorGeometry` (vectors.rs) draws `RShape2D` and
//! `RCurve3D` with the images and camera views their fills show;
//! `MeshGeometry` (meshes.rs) draws `RShape3D` and `RParticles` with their
//! material textures.

pub mod camera_views;
pub mod images;
pub mod meshes;
pub mod textures;
pub mod vectors;
