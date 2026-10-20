//! The render objects: what a scene shows at one time, as a tree of plain
//! values. This tree and the `RenderConfig` are everything a renderer is
//! given:
//!
//! ```text
//! scene side ──snapshot──▶ RNode tree ◀──reads── renderer
//! ```
//!
//! An `RNode` knows nothing of mobjects, animations or time. Any renderer
//! can read it.
//!
//! One `RNode` per scene object: `uid` (the scene object's, the same in
//! every snapshot), `transform` (relative to the parent), `alpha`
//! (multiplies the opacity of the whole subtree) and what kind of node it
//! is (`Kind`). Only groups have children.
//!
//! Change detection: values that are costly to turn into what a renderer
//! draws (points, fills, strokes, meshes, materials, particles) carry a
//! `version`. Equal versions promise equal values, so a renderer may keep
//! what it built from a value while the version holds. A new value gets a
//! version never handed out before (`next_version()`).

pub mod values;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use glam::{DMat4, DQuat, DVec3};

pub use values::{Fill, Instances, Material, Mesh, Points, Stroke, SurfaceShader, Texture};

static NEXT_UID: AtomicU64 = AtomicU64::new(1);
static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

/// A uid never handed out before in this process: scene objects, render
/// objects made by hand, textures and surface shaders get one.
pub fn new_uid() -> u64 {
    NEXT_UID.fetch_add(1, Ordering::Relaxed)
}

/// A version never handed out before in this process.
pub fn next_version() -> u64 {
    NEXT_VERSION.fetch_add(1, Ordering::Relaxed)
}

/// Where a node sits relative to its parent: scaled first, then rotated,
/// then moved (`T @ R @ S`, like `Transform` in manim/utils/transform.py).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: DVec3,
    pub rotation: DQuat,
    pub scale: DVec3,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        position: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        scale: DVec3::ONE,
    };

    pub fn matrix(&self) -> DMat4 {
        DMat4::from_scale_rotation_translation(self.scale, self.rotation.normalize(), self.position)
    }

    /// Scaled to nothing on some axis: nothing of the node shows.
    pub fn is_flat(&self) -> bool {
        self.scale.abs().min_element() < 1e-9
    }
}

/// One node of the tree.
pub struct RNode {
    /// The scene object's uid: the same in every snapshot of it, so a
    /// renderer recognizes the node from frame to frame.
    pub uid: u64,
    pub transform: Transform,
    /// Multiplies the opacity of this node and everything below it
    /// (fills, strokes, materials; a light's intensity).
    pub alpha: f64,
    pub kind: Kind,
}

pub enum Kind {
    /// `RGroup`: holds other nodes, draws nothing itself.
    Group { children: Vec<Arc<RNode>> },
    /// `RClippingGroup`: shows `content` only where `mask` covers it
    /// (times the mask's alpha). With `one_sided`, the mask only counts
    /// where its front faces the camera. Lights in `content` light only
    /// `content`. Without a mask or content nothing shows.
    ClippingGroup {
        mask: Option<Arc<RNode>>,
        content: Option<Arc<RNode>>,
        one_sided: bool,
    },
    /// `RShape2D`: a filled and stroked path in the local XY plane.
    Shape2D { path: Path, fill: Arc<Fill> },
    /// `RCurve3D`: a stroked path in 3D whose ribbon faces the camera.
    Curve3D { path: Path },
    /// `RShape3D`: a lit triangle mesh.
    Shape3D(Surface),
    /// `RParticles`: many copies of one mesh.
    Particles {
        surface: Surface,
        instances: Arc<Instances>,
    },
    /// `RDirectionalLight`, `RPointLight`, `RSpotLight`.
    Light(Light),
}

/// The points and stroke of an `RShape2D` or `RCurve3D`.
pub struct Path {
    pub points: Arc<Points>,
    pub stroke: Arc<Stroke>,
    /// Whether it blocks light for meshes behind it.
    pub cast_shadows: bool,
}

/// The mesh and material of an `RShape3D` or `RParticles`.
pub struct Surface {
    pub mesh: Arc<Mesh>,
    pub material: Arc<Material>,
    /// Whether it blocks light (it always receives shadows).
    pub cast_shadows: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    /// Parallel light along the node's local -z (like the sun).
    Directional,
    /// Light in all directions from the node's position.
    Point,
    /// A point light restricted to a cone around the local -z axis.
    Spot,
}

/// A light, after glTF's KHR_lights_punctual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    pub kind: LightKind,
    /// sRGB, as colors are written.
    pub color: [f64; 3],
    /// Scales the radiance: a white diffuse surface lit head-on by a
    /// directional light of intensity pi appears pure white.
    pub intensity: f64,
    /// How large the light appears, which softens its shadows: a radius
    /// for point and spot lights, an angular diameter in radians for
    /// directional ones.
    pub size: f64,
    pub cast_shadows: bool,
    /// Where point and spot lights fade to nothing (None: never).
    pub range: Option<f64>,
    /// Inner and outer cone angles in radians from the axis (spot lights).
    pub cone: Option<(f64, f64)>,
}

impl RNode {
    /// The children in drawing order (a clipping group's are its mask and
    /// content).
    pub fn children(&self) -> Vec<&Arc<RNode>> {
        match &self.kind {
            Kind::Group { children } => children.iter().collect(),
            Kind::ClippingGroup { mask, content, .. } => {
                mask.iter().chain(content.iter()).collect()
            }
            _ => Vec::new(),
        }
    }
}
