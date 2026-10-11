//! Packing: a render object tree turned into the records the GPU reads.
//! Nothing here touches the GPU.
//!
//! | Module | What it does |
//! |---|---|
//! | `packer` | `Packer`: one view's tree in, a `Packet` of draws and records out |
//! | `flatten` | the tree's drawable leaves with world matrices, clip scopes and lights |
//! | `vectors` | the retained records of `RShape2D` and `RCurve3D` nodes |
//! | `meshes` | the records of `RShape3D` and `RParticles` nodes |
//! | `lights` | the light records, and the default light |
//! | `shadows` | where shadow maps are rendered from, and which changed |
//! | `records` | every GPU record, field for field as the shaders declare it |
//! | `arena` | growable record arrays that mirror GPU buffers |

pub mod arena;
pub mod flatten;
pub mod lights;
pub mod meshes;
pub mod packer;
pub mod records;
pub mod shadows;
pub mod vectors;
