//! The errors of the resolver, the renderer and the compositor. They are
//! libraries of their own: they report what went wrong as these, and the
//! Python bindings (python/) turn them into Python exceptions.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// No usable GPU, or the GPU failed.
    Gpu(String),
    /// A shader does not compile (e.g. a user's surface shader).
    Shader(String),
    /// The scene asks for something the renderer cannot draw.
    Unsupported(String),
    /// A setting or input is not valid.
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Gpu(message)
            | Error::Shader(message)
            | Error::Unsupported(message)
            | Error::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}
