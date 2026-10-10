//! The core's renderer: packets in, frames out, on wgpu. Ports the Python
//! renderer (manimx/renderer/webgpu/) module by module, with the same WGSL.
//!
//! | Module | Python counterpart |
//! |---|---|
//! | `context` | `GPUContext`, `load_wgsl` (gpu.py) |
//! | `buffer` | `GrowableBuffer` (gpu.py) |
//! | `targets` | `RenderTargets` (targets.py) |
//! | `vector_pass` | `VectorPass` (vector_pass.py) |
//! | `mesh_pass` | `MeshPass` (mesh_pass.py) |
//! | `lighting_pass` | `LightingPass` (lighting_pass.py) |
//! | `shadow_pass` | `ShadowPass` (shadow_pass.py) |
//! | `profiler` | `GpuProfiler` (profiler.py) |
//! | `renderer` | `WGPURenderer.render_layers` (renderer.py) |
//! | `composer` | the GPU side of `FrameComposer` (composer.py) |
//! | `frame_renderer` | `FrameComposer` + `FramePipeline` behind one Python class |

pub mod buffer;
pub mod composer;
pub mod context;
pub mod frame_renderer;
pub mod lighting_pass;
pub mod mesh_pass;
pub mod profiler;
pub mod renderer;
pub mod shadow_pass;
pub mod targets;
pub mod vector_pass;
