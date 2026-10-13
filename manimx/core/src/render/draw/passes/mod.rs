//! The passes of one view, in the order the drawer runs them: mask,
//! shadow, gbuffer, lighting, vector, translucent. A pass chooses its
//! target and which geometry (geometry/) it draws in which role.

pub mod gbuffer;
pub mod lighting;
pub mod mask;
pub mod shadow;
pub mod translucent;
pub mod vector;
