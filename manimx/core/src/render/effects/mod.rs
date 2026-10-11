//! The effects of the frame composer, each with its settings (from
//! manimx/renderer/effects.py) and the GPU work it does. `RenderSamples`
//! is not here: the caller decides the sub-frames (their times, camera
//! offsets and weights), and the composer averages them.
//!
//! | Module | What it does |
//! |---|---|
//! | `bloom` | bright light spreads into a glow (lit stage) |
//! | `shader_effect` | the user's own WGSL on the lit layer or the frame |
//! | `depth_of_field` | the camera's lens: blur by the distance from the focus |

pub mod bloom;
pub mod depth_of_field;
pub mod shader_effect;

pub use bloom::Bloom;
pub use shader_effect::ShaderEffect;

/// One effect of the chain, in order.
pub enum Effect {
    Bloom(Bloom),
    Shader(ShaderEffect),
}
