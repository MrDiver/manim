//! Per-vertex tangents for normal maps, computed from a mesh's UVs when it
//! has none of its own.

use glam::DVec3;

/// Per-vertex tangents from UV gradients, after Eric Lengyel, "Computing
/// Tangent Space Basis Vectors for an Arbitrary Mesh"
/// (https://terathon.com/code/tangent.html): each triangle's tangent and
/// bitangent are summed at its corners, the tangent is made orthogonal to
/// the normal (Gram-Schmidt), and w holds the bitangent's handedness.
pub fn tangents(
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
