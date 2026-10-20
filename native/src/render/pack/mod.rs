//! Pack: the scenes the renderer is sent (`Update`s, flat/) turned into
//! the records the GPU reads, through the store it keeps of each scene
//! (store.rs). Nothing here touches the GPU.

pub mod store;
