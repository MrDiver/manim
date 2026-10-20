//! The effects of Composite, each with its settings and its GPU work.
//! Motion blur is not one: the caller decides the sub-frames, and the
//! compositor averages them.

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
