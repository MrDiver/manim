//! The shape math of packing: what a path's fill and stroke are made of.
//!
//! | Module | What it does |
//! |---|---|
//! | `tessellation` | a fill as triangles with exact curved edges |
//! | `strokes` | a stroke as curve records with per-point width and color |

pub mod strokes;
pub mod tessellation;
