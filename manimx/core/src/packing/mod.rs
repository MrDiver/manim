//! Packing: a scene graph and the values of its signals turned into the
//! records the GPU reads. `Packer::pack_frame` (packer.rs) runs it:
//!
//! ```text
//! scene_graph  the nodes flattened: world matrices, painter's order,
//!              clip scopes, lights
//! vectors      records of Shape2D and Curve3D nodes, kept across frames
//! meshes       records of Shape3D and Particles nodes
//! lights       records of the lights
//! shadows      where shadow maps are rendered from, which changed
//! ```
//!
//! | Module | What it does |
//! |---|---|
//! | `packer` | `Packer`: runs the steps above, returns a `Packet` |
//! | `scene_graph` | `SceneGraph`: which nodes a scene has, which slot feeds which field; flattening |
//! | `vectors` | records of 2D shapes and 3D curves, in arenas kept across frames |
//! | `meshes` | records of meshes and particles |
//! | `lights` | light records and the default light |
//! | `shadows` | shadow views, scene bounds, which shadow layers changed |
//! | `records` | the GPU record structs, as the WGSL structs declare them |
//! | `arena` | growable record arrays that mirror GPU buffers |
//! | `callbacks` | the Python functions called once per version |

pub mod arena;
pub mod callbacks;
pub mod lights;
pub mod meshes;
pub mod packer;
pub mod records;
pub mod scene_graph;
pub mod shadows;
pub mod vectors;
