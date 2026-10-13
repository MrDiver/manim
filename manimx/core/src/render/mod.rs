//! The renderer: render object trees (robjects/) and their settings in,
//! frames out, through the stages Pack, Draw, Composite and Output that
//! `RenderPipeline` (pipeline.rs) runs. It knows nothing of signals,
//! scenes or Python. core/README.md describes the architecture.

pub mod composite;
pub mod draw;
pub mod error;
pub mod gpu;
pub mod output;
pub mod pack;
pub mod pipeline;
pub mod settings;
