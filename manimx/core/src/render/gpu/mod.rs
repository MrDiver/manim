//! What every stage uses on the GPU: the device, the shaders compiled
//! into the library, growing buffers, fullscreen passes, timing, and
//! reading textures back to look at them.

pub mod buffer;
pub mod device;
pub mod fullscreen;
pub mod inspect;
pub mod profiler;
pub mod shaders;
