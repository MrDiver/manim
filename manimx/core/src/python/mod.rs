//! The Python classes of the core that are not part of evaluation: the
//! render objects (`manimx.robjects`) and the renderer.
//!
//! | Module | What it holds |
//! |---|---|
//! | `robjects` | `RGroup`, `RShape2D`, ..., `Fill`, `Stroke`, `MeshData`, `PbrMaterial`, `Texture`, `SurfaceShader` |
//! | `frame_renderer` | `FrameRenderer`: render/ behind a Python class |

pub mod frame_renderer;
pub mod robjects;
