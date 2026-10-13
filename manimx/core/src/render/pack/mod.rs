//! Pack: a view's render object tree turned into the records the GPU
//! reads (`Packer`, packer.rs). Nothing here touches the GPU.

pub mod arena;
pub mod color;
pub mod flatten;
pub mod lights;
pub mod meshes;
pub mod packer;
pub mod records;
pub mod shadows;
pub mod shapes;
pub mod vectors;
