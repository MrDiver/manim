//! The core's renderer on wgpu: render object trees (robjects/) and their
//! `ViewSettings` in, frames out. A library of its own: it knows nothing
//! of signals, scenes, timelines or Python (python/frame_renderer.rs is
//! its Python class). The WGSL shaders live in core/shaders/ and are
//! compiled into the library (shaders.rs).
//!
//! | Module | What it does |
//! |---|---|
//! | `frame_renderer` | `FrameRenderer`: sub-frames and views in, frames in flight, read back |
//! | `settings` | `ViewSettings`: the camera, camera views, lighting and shadow settings of a view |
//! | `packing/` | a tree turned into GPU records (no GPU work) |
//! | `geometry/` | the shape math of packing: fills and strokes |
//! | `renderer` | one view: mask, shadow, geometry, lighting and vector passes into the lit and vector layers |
//! | `passes/` | the passes, and material textures |
//! | `composer` | sub-frames and views into one frame: effects, compositing, depth of field |
//! | `effects/` | bloom, the user's shader effects, depth of field |
//! | `targets` | every texture a frame renders into |
//! | `fullscreen` | full-screen passes (composite, effects) |
//! | `context` | the device and shader loading |
//! | `shaders` | the WGSL shaders, compiled in |
//! | `buffer` | GPU buffers that grow |
//! | `color` | sRGB and linear light |
//! | `profiler` | GPU time per pass |
//! | `inspect` | reads textures back to look at them (scripts/render_internals.py) |
//! | `error` | what goes wrong |

pub mod buffer;
pub mod color;
pub mod composer;
pub mod context;
pub mod effects;
pub mod error;
pub mod frame_renderer;
pub mod fullscreen;
pub mod geometry;
pub mod inspect;
pub mod packing;
pub mod passes;
pub mod profiler;
pub mod renderer;
pub mod settings;
pub mod shaders;
pub mod targets;
