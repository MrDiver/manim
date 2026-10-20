//! Pack: the scenes the renderer is sent (`Update`s, flat/) turned into
//! the records the GPU reads, through the store it keeps of each scene
//! (store.rs). Nothing here touches the GPU.

pub mod arena;
pub mod bvh;
pub mod casters;
pub mod color;
pub mod lights;
pub mod meshes;
pub mod shapes;
pub mod store;
pub mod tangents;
pub mod vectors;
