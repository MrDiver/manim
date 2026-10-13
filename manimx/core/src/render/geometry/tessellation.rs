//! Fills of 2D paths as triangles, by the nonzero winding rule.
//!
//! A point is inside a path when the subpaths around it do not cancel
//! out: their windings (+1 counterclockwise, -1 clockwise) add up to
//! anything but 0. It is the rule of Cairo, SVG and fonts, and it holds
//! for any path: subpaths that cross each other or themselves (a path in
//! the middle of a morph) fill like the same path drawn by Cairo, and a
//! path that changes a little fills a little differently, never flipping.
//!
//! [lyon](https://github.com/nical/lyon)'s fill tessellator does it: the
//! cubics are flattened to lines within a tolerance relative to the path's
//! size, and a sweep line splits the edges where they cross and emits
//! monotone pieces as triangles (lyon's design notes:
//! <https://github.com/nical/lyon/wiki/Fill-tessellator>).
//!
//! Each vertex records where it comes from, the index of its source cubic
//! and the parameter `t` on it, so colors are computed on the GPU and a fill
//! depends on the points alone. They ride through the tessellator as one
//! attribute, `curve + t`, which it interpolates along edges and where
//! edges cross.

use glam::DVec2;
use lyon_tessellation::geom::point;
use lyon_tessellation::path::Path;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex as LyonVertex,
    VertexBuffers,
};

/// Flattening tolerance, as a fraction of the path's size (width or height,
/// the larger): a glyph 1000 pixels tall is off by at most 0.1 pixel.
const TOLERANCE: f64 = 1e-4;
/// Points closer than this are the same point.
const POINT_EQUALITY: f64 = 1e-8;

/// One vertex of a fill (`FILL_VERTEX_SIZE`, 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FillVertex {
    pub position: [f32; 2],
    pub curve: u32,
    pub t: f32,
}

/// The fill of a path in the RBezierPath layout (four 2D points per cubic,
/// a new subpath wherever a curve does not start where the last one
/// ended) as a non-indexed triangle list. Open subpaths are closed with a
/// straight line.
pub fn tessellate_fill(points: &[DVec2]) -> Vec<FillVertex> {
    let curves: Vec<[DVec2; 4]> = points
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    if curves.is_empty() {
        return Vec::new();
    }
    let (mut low, mut high) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
    for point in points {
        low = low.min(*point);
        high = high.max(*point);
    }
    let size = (high - low).max_element();
    if size <= POINT_EQUALITY {
        return Vec::new();
    }
    // Relative to the path's lowest corner, so f32 keeps the precision.
    let local = |p: DVec2| point((p.x - low.x) as f32, (p.y - low.y) as f32);
    let mut builder = Path::builder_with_attributes(1);
    for (start, stop) in subpath_ranges(&curves) {
        builder.begin(local(curves[start][0]), &[start as f32]);
        for (index, curve) in curves[start..stop].iter().enumerate() {
            let end = (start + index + 1) as f32; // the end of cubic i is curve + t = i + 1
            builder.cubic_bezier_to(local(curve[1]), local(curve[2]), local(curve[3]), &[end]);
        }
        builder.end(true);
    }
    let path = builder.build();
    let options = FillOptions::tolerance((size * TOLERANCE) as f32).with_fill_rule(FillRule::NonZero);
    let last_curve = curves.len() - 1;
    let mut buffers: VertexBuffers<FillVertex, u32> = VertexBuffers::new();
    let result = FillTessellator::new().tessellate_path(
        &path,
        &options,
        &mut BuffersBuilder::new(&mut buffers, |mut vertex: LyonVertex| {
            let position = vertex.position();
            let along = vertex.interpolated_attributes()[0].max(0.0);
            let curve = (along.floor() as usize).min(last_curve);
            FillVertex {
                position: [(position.x as f64 + low.x) as f32, (position.y as f64 + low.y) as f32],
                curve: curve as u32,
                t: (along - curve as f32).clamp(0.0, 1.0),
            }
        }),
    );
    if result.is_err() {
        return Vec::new(); // a path lyon cannot fill (NaN points): drawn without a fill
    }
    buffers.indices.iter().map(|&index| buffers.vertices[index as usize]).collect()
}

/// `[start, stop)` curve ranges of the continuous subpaths.
fn subpath_ranges(curves: &[[DVec2; 4]]) -> Vec<(usize, usize)> {
    let mut breaks = vec![0];
    for index in 1..curves.len() {
        if curves[index][0].distance(curves[index - 1][3]) > POINT_EQUALITY {
            breaks.push(index);
        }
    }
    breaks.push(curves.len());
    breaks.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(size: f64, offset: DVec2, clockwise: bool) -> Vec<DVec2> {
        let mut corners = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(size, 0.0),
            DVec2::new(size, size),
            DVec2::new(0.0, size),
        ];
        if clockwise {
            corners.reverse();
        }
        let mut points = Vec::new();
        for index in 0..4 {
            let (a, b) = (corners[index] + offset, corners[(index + 1) % 4] + offset);
            points.extend([a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0, b]);
        }
        points
    }

    fn area(vertices: &[FillVertex]) -> f64 {
        vertices
            .chunks_exact(3)
            .map(|t| {
                let p = |v: &FillVertex| DVec2::new(v.position[0] as f64, v.position[1] as f64);
                ((p(&t[1]) - p(&t[0])).perp_dot(p(&t[2]) - p(&t[0])) / 2.0).abs()
            })
            .sum()
    }

    #[test]
    fn nested_subpaths_fill_by_the_nonzero_rule() {
        // Wound against the outer square: a hole.
        let mut points = square(4.0, DVec2::ZERO, false);
        points.extend(square(2.0, DVec2::new(1.0, 1.0), true));
        assert!((area(&tessellate_fill(&points)) - 12.0).abs() < 1e-4);
        // Wound the same way: covered twice, filled.
        let mut points = square(4.0, DVec2::ZERO, false);
        points.extend(square(2.0, DVec2::new(1.0, 1.0), false));
        assert!((area(&tessellate_fill(&points)) - 16.0).abs() < 1e-4);
    }

    #[test]
    fn crossing_subpaths_fill_by_the_nonzero_rule() {
        let mut points = square(2.0, DVec2::ZERO, false);
        points.extend(square(2.0, DVec2::new(1.0, 1.0), false));
        assert!((area(&tessellate_fill(&points)) - 7.0).abs() < 1e-4); // the union
        let mut points = square(2.0, DVec2::ZERO, false);
        points.extend(square(2.0, DVec2::new(1.0, 1.0), true));
        assert!((area(&tessellate_fill(&points)) - 6.0).abs() < 1e-4); // the overlap cancels
    }

    #[test]
    fn a_subpath_crossing_itself_fills_both_loops() {
        // A bow tie: two triangles of opposite winding, each filled.
        let corners = [DVec2::new(0.0, 0.0), DVec2::new(2.0, 2.0), DVec2::new(2.0, 0.0), DVec2::new(0.0, 2.0)];
        let mut points = Vec::new();
        for index in 0..4 {
            let (a, b) = (corners[index], corners[(index + 1) % 4]);
            points.extend([a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0, b]);
        }
        assert!((area(&tessellate_fill(&points)) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn curves_are_within_the_tolerance_and_vertices_know_their_source() {
        // A circle of radius 1 from four cubics (kappa = 0.5523).
        let k = 0.552_284_749_8;
        let quarter = |a: DVec2, b: DVec2| [a, a + k * b, b + k * a, b];
        let (x, y) = (DVec2::X, DVec2::Y);
        let points: Vec<DVec2> = [quarter(x, y), quarter(y, -x), quarter(-x, -y), quarter(-y, x)].concat();
        let vertices = tessellate_fill(&points);
        assert!((area(&vertices) - std::f64::consts::PI).abs() < 2e-3);
        for vertex in &vertices {
            let (curve, t) = (vertex.curve as usize, vertex.t);
            assert!(curve < 4 && (0.0..=1.0).contains(&t));
        }
        let left = vertices.iter().find(|v| (v.position[0] + 1.0).abs() < 1e-6).unwrap();
        assert!(left.curve == 1 && (left.t - 1.0).abs() < 1e-4 || left.curve == 2 && left.t.abs() < 1e-4);
    }

    #[test]
    fn nothing_without_area() {
        let line = vec![DVec2::ZERO, DVec2::X, DVec2::X * 2.0, DVec2::X * 3.0];
        assert!(area(&tessellate_fill(&line)) < 1e-12);
        assert!(tessellate_fill(&[]).is_empty());
    }

    /// `MANIMX_BENCH_PATHS=file cargo test --release --lib tessellation_benchmark -- --ignored --nocapture`
    /// (the file: u32 path count, then per path a u32 point count and f64 x, y pairs).
    #[test]
    #[ignore]
    fn tessellation_benchmark() {
        let bytes = std::fs::read(std::env::var("MANIMX_BENCH_PATHS").unwrap()).unwrap();
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let float = |at: usize| f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let (mut paths, mut at) = (Vec::new(), 4);
        for _ in 0..word(0) {
            let count = word(at);
            at += 4;
            paths.push((0..count).map(|i| DVec2::new(float(at + 16 * i), float(at + 16 * i + 8))).collect::<Vec<_>>());
            at += 16 * count;
        }
        let total = 1_000_000;
        let started = std::time::Instant::now();
        let mut vertices = 0;
        for index in 0..total {
            vertices += tessellate_fill(&paths[index % paths.len()]).len();
        }
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "{total} fills of {} paths: {seconds:.2} s, {:.2} us/path, {:.0} vertices/path",
            paths.len(),
            seconds * 1e6 / total as f64,
            vertices as f64 / total as f64
        );
    }
}
