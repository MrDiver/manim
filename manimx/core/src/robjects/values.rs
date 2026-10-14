//! The values render objects hold: paths and their styles, meshes,
//! materials, textures, surface shaders and particles (in Python `Fill`,
//! `Stroke`, `MeshData`, `PbrMaterial`, `Texture`, `SurfaceShader`).
//! Colors are sRGB, as they are written; a renderer converts them to
//! linear light. Every value is immutable once made; the versioned ones
//! get their version when they are made (see mod.rs).

use std::sync::{Arc, OnceLock};

use glam::DVec3;

use super::{new_uid, next_version};

/// Points per cubic Bezier curve: anchor, handle, handle, anchor.
pub const POINTS_PER_CURVE: usize = 4;
/// Points closer than this are the same point (where subpaths join).
pub const TOLERANCE_FOR_POINT_EQUALITY: f64 = 1e-6;

/// Points of cubic Bezier curves, four per curve, `dimension` (2 or 3)
/// numbers per point, row by row. A new subpath starts wherever a curve
/// does not start where the one before ended.
pub struct Points {
    pub version: u64,
    pub data: Vec<f64>,
    pub dimension: usize,
    /// `bounds()`, computed on first use: the points never change.
    bounds: OnceLock<Option<(DVec3, DVec3)>>,
}

impl Points {
    pub fn new(data: Vec<f64>, dimension: usize) -> Points {
        Points {
            version: next_version(),
            data,
            dimension,
            bounds: OnceLock::new(),
        }
    }

    /// The smallest box (lowest, highest corner) around every point, in
    /// the path's own space; None without points. A cubic curve lies
    /// inside the hull of its four points, so the curves lie inside it too
    /// (culling, render/pack/vectors.rs).
    pub fn bounds(&self) -> Option<(DVec3, DVec3)> {
        *self.bounds.get_or_init(|| {
            let mut points = (0..self.count()).map(|index| self.point(index));
            let first = points.next()?;
            Some(points.fold((first, first), |(low, high), point| {
                (low.min(point), high.max(point))
            }))
        })
    }

    pub fn count(&self) -> usize {
        self.data.len() / self.dimension.max(1)
    }

    /// Point `index` in 3D (z = 0 for 2D points).
    pub fn point(&self, index: usize) -> DVec3 {
        let at = &self.data[index * self.dimension..(index + 1) * self.dimension];
        DVec3::new(at[0], at[1], if self.dimension == 3 { at[2] } else { 0.0 })
    }
}

/// A fill: one color and opacity, or lists of them spread evenly over the
/// points (a gradient along the path), and optionally an image the colors
/// tint, stretched over the shape's local box around its origin.
pub struct Fill {
    pub version: u64,
    pub colors: Vec<[f64; 3]>,
    pub opacities: Vec<f64>,
    pub image: Option<Arc<Texture>>,
}

impl Fill {
    pub fn new(colors: Vec<[f64; 3]>, opacities: Vec<f64>) -> Fill {
        Fill::with_image(colors, opacities, None)
    }

    pub fn with_image(
        colors: Vec<[f64; 3]>,
        opacities: Vec<f64>,
        image: Option<Arc<Texture>>,
    ) -> Fill {
        Fill {
            version: next_version(),
            colors,
            opacities,
            image,
        }
    }
}

/// A stroke: colors, opacities and widths are one value or a list spread
/// over the points; `joints` (`JointType` values) one value or a list
/// spread over the curves; one `cap` (a `CapStyle` value). Widths are in Manim's units: 4 is
/// 0.04 world units (`STROKE_WIDTH_SCALE`).
pub struct Stroke {
    pub version: u64,
    pub colors: Vec<[f64; 3]>,
    pub opacities: Vec<f64>,
    pub widths: Vec<f64>,
    pub joints: Vec<u32>,
    pub cap: u32,
}

/// World units per unit of stroke width.
pub const STROKE_WIDTH_SCALE: f64 = 0.01;

impl Stroke {
    pub fn new(
        colors: Vec<[f64; 3]>,
        opacities: Vec<f64>,
        widths: Vec<f64>,
        joints: Vec<u32>,
        cap: u32,
    ) -> Stroke {
        Stroke {
            version: next_version(),
            colors,
            opacities,
            widths,
            joints,
            cap,
        }
    }
}

/// An indexed triangle mesh in local coordinates. Tangents (xyz and the
/// handedness in w, as in glTF) are only needed for normal maps.
pub struct Mesh {
    pub version: u64,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Three per triangle.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// A mesh, with smooth normals when `normals` is None. Fails when the
    /// indices do not make triangles of the vertices.
    pub fn new(
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        normals: Option<Vec<[f32; 3]>>,
        uvs: Option<Vec<[f32; 2]>>,
        tangents: Option<Vec<[f32; 4]>>,
    ) -> Result<Mesh, String> {
        if !indices.len().is_multiple_of(3) {
            return Err(format!(
                "indices must form triangles, got {} indices",
                indices.len()
            ));
        }
        if indices
            .iter()
            .any(|&index| index as usize >= positions.len())
        {
            return Err("an index points past the last vertex".into());
        }
        let normals = normals.unwrap_or_else(|| smooth_normals(&positions, &indices));
        Ok(Mesh {
            version: next_version(),
            positions,
            normals,
            uvs,
            tangents,
            indices,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The tangents normal maps need: the mesh's own, else computed from
    /// its UVs; None without UVs.
    pub fn tangents_or_computed(&self) -> Option<Vec<[f32; 4]>> {
        if let Some(tangents) = &self.tangents {
            return Some(tangents.clone());
        }
        let uvs = self.uvs.as_ref()?;
        Some(tangents(&self.positions, &self.normals, uvs, &self.indices))
    }
}

/// Per vertex, the area-weighted average of the normals of the triangles
/// around it ((0, 0, 1) where there are none).
fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let point = |index: u32| DVec3::from(positions[index as usize].map(f64::from));
    let mut sums = vec![DVec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [point(triangle[0]), point(triangle[1]), point(triangle[2])];
        // Unnormalized: its length is twice the triangle's area.
        let face = (b - a).cross(c - a);
        for &corner in triangle {
            sums[corner as usize] += face;
        }
    }
    sums.iter()
        .map(|sum| {
            let length = sum.length();
            let normal = if length > 0.0 {
                *sum / length
            } else {
                DVec3::Z
            };
            normal.as_vec3().to_array()
        })
        .collect()
}

/// Per-vertex tangents from UV gradients, after Eric Lengyel, "Computing
/// Tangent Space Basis Vectors for an Arbitrary Mesh"
/// (https://terathon.com/code/tangent.html): each triangle's tangent and
/// bitangent are summed at its corners, the tangent is made orthogonal to
/// the normal (Gram-Schmidt), and w holds the bitangent's handedness.
fn tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let point = |index: u32| DVec3::from(positions[index as usize].map(f64::from));
    let uv = |index: u32| uvs[index as usize].map(f64::from);
    let mut tangent_sums = vec![DVec3::ZERO; positions.len()];
    let mut bitangent_sums = vec![DVec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let (p0, p1, p2) = (point(triangle[0]), point(triangle[1]), point(triangle[2]));
        let (t0, t1, t2) = (uv(triangle[0]), uv(triangle[1]), uv(triangle[2]));
        let (edge1, edge2) = (p1 - p0, p2 - p0);
        let duv1 = [t1[0] - t0[0], t1[1] - t0[1]];
        let duv2 = [t2[0] - t0[0], t2[1] - t0[1]];
        let determinant = duv1[0] * duv2[1] - duv2[0] * duv1[1];
        let scale = if determinant.abs() > 1e-12 {
            1.0 / determinant
        } else {
            0.0
        };
        let tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * scale;
        let bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) * scale;
        for &corner in triangle {
            tangent_sums[corner as usize] += tangent;
            bitangent_sums[corner as usize] += bitangent;
        }
    }
    (0..positions.len())
        .map(|vertex| {
            let normal = DVec3::from(normals[vertex].map(f64::from));
            let tangent = tangent_sums[vertex] - normal * normal.dot(tangent_sums[vertex]);
            let length = tangent.length();
            let tangent = if length > 1e-12 {
                tangent / length
            } else {
                // No UV gradient: any direction across the normal.
                normal.cross(DVec3::Z) + normal.cross(DVec3::Y) * 1e-3
            };
            let handedness = if normal.cross(tangent).dot(bitangent_sums[vertex]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            [
                tangent.x as f32,
                tangent.y as f32,
                tangent.z as f32,
                handedness,
            ]
        })
        .collect()
}

/// A glTF 2.0 metallic-roughness material (`PbrMaterial`). Factors are
/// multiplied with their texture where one is set. The textures are base
/// color (sRGB RGBA), metallic-roughness (linear, roughness in G,
/// metallic in B), normal (tangent space), occlusion (linear, in R) and
/// emissive (sRGB RGB).
pub struct Material {
    pub version: u64,
    /// sRGB.
    pub base_color: [f64; 3],
    pub opacity: f64,
    pub metallic: f64,
    pub roughness: f64,
    /// sRGB.
    pub emissive: [f64; 3],
    pub emissive_strength: f64,
    pub normal_scale: f64,
    pub occlusion_strength: f64,
    pub textures: [Option<Arc<Texture>>; 5],
    pub surface_shader: Option<Arc<SurfaceShader>>,
}

impl Default for Material {
    /// `PbrMaterial()`: white, half rough, not metallic, no textures.
    fn default() -> Material {
        Material {
            version: next_version(),
            base_color: [1.0; 3],
            opacity: 1.0,
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            emissive_strength: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            textures: Default::default(),
            surface_shader: None,
        }
    }
}

/// An RGBA8 image, first row at the top. Its `uid` lets a renderer upload
/// it once however many materials share it.
pub struct Texture {
    pub uid: u64,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Texture {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Texture {
        assert_eq!(pixels.len(), (4 * width * height) as usize, "RGBA8 pixels");
        Texture {
            uid: new_uid(),
            width,
            height,
            pixels,
        }
    }
}

/// WGSL defining `fn shade(surface: Surface, in: SurfaceInput) -> Surface`,
/// which changes a mesh's surface per pixel before lighting (see
/// `SurfaceShader` in Python). Its `uid` lets a renderer compile it once.
pub struct SurfaceShader {
    pub uid: u64,
    pub code: String,
}

/// The line of surface_shader.wgsl that a `SurfaceShader`'s code replaces.
pub const DEFAULT_SURFACE_SHADER: &str =
    "fn shade_surface(surface: Surface, in: SurfaceInput) -> Surface { return surface; }";

impl SurfaceShader {
    pub fn new(code: String) -> SurfaceShader {
        SurfaceShader {
            uid: new_uid(),
            code,
        }
    }

    /// The code that takes the place of `DEFAULT_SURFACE_SHADER`.
    pub fn wgsl(&self) -> String {
        format!(
            "{}\n\nfn shade_surface(surface: Surface, in: SurfaceInput) -> Surface {{\n    return shade(surface, in);\n}}\n",
            self.code
        )
    }
}

/// The copies of `RParticles`: per particle a position, a rotation (a
/// quaternion w, x, y, z), a scale, and r, g, b and a glow that tint the
/// material. A particle scaled to nothing on any axis is not drawn.
pub struct Instances {
    pub version: u64,
    pub positions: Vec<[f64; 3]>,
    pub rotations: Vec<[f64; 4]>,
    pub scales: Vec<[f64; 3]>,
    pub colors: Vec<[f64; 4]>,
}

impl Instances {
    pub fn new(
        positions: Vec<[f64; 3]>,
        rotations: Vec<[f64; 4]>,
        scales: Vec<[f64; 3]>,
        colors: Vec<[f64; 4]>,
    ) -> Instances {
        Instances {
            version: next_version(),
            positions,
            rotations,
            scales,
            colors,
        }
    }

    /// How many particles there are (the shortest of the four lists).
    pub fn count(&self) -> usize {
        self.positions
            .len()
            .min(self.rotations.len())
            .min(self.scales.len())
            .min(self.colors.len())
    }
}
