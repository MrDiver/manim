//! The core's renderer on wgpu: a packet (packer.rs) in, a frame out. The
//! WGSL shaders live in manimx/renderer/shaders/.
//!
//! | Module | What it does |
//! |---|---|
//! | `frame_renderer` | `FrameRenderer`, the Python class: frames in flight, held frames, read-back |
//! | `composer` | sub-frames and views into one frame: effects, compositing, depth of field |
//! | `renderer` | one view: mask, shadow, geometry, lighting and vector passes into the lit and vector layers |
//! | `camera_views` | the pictures of camera views, rendered before the frame |
//! | `vector_pass` | 2D shapes and curves (and clip masks) |
//! | `mesh_pass` | meshes into the G-buffer |
//! | `lighting_pass` | deferred lighting of the G-buffer |
//! | `shadow_pass` | shadow maps |
//! | `targets` | every texture a frame renders into |
//! | `context` | the device and shader loading |
//! | `buffer` | GPU buffers that grow |
//! | `profiler` | GPU time per pass |
//! | `inspect` | reads textures back to look at them (scripts/render_internals.py) |

pub mod buffer;
pub mod camera_views;
pub mod composer;
pub mod context;
pub mod frame_renderer;
pub mod inspect;
pub mod lighting_pass;
pub mod mesh_pass;
pub mod profiler;
pub mod renderer;
pub mod shadow_pass;
pub mod targets;
pub mod vector_pass;
