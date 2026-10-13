//! The shape math packing uses.
//!
//! | Module | What it does |
//! |---|---|
//! | `tessellation` | fills as triangles, by the nonzero winding rule (lyon) |
//! | `strokes` | strokes as linked curve records with per-point width and color |

pub mod strokes;
pub mod tessellation;
