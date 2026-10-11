//! Fills of 2D paths as triangles with exact curves.
//!
//! The approach is ManimGL's, after Charles Loop and Jim Blinn,
//! [Resolution Independent Curve Rendering using Programmable Graphics
//! Hardware](https://www.microsoft.com/en-us/research/publication/resolution-independent-curve-rendering-using-programmable-graphics-hardware/)
//! (SIGGRAPH 2005):
//!
//! 1. Every cubic Bezier becomes quadratics within a bounded error.
//! 2. Every subpath becomes a polygon *ring*: a quadratic that bulges into
//!    the filled region (concave) adds its control point to the ring, one
//!    that bulges out (convex) only its chord.
//! 3. Rings are grouped into outer rings and their holes (even-odd rule)
//!    and triangulated with [earcut](https://github.com/mapbox/earcut)
//!    (ear clipping with holes bridged in, see David Eberly,
//!    [Triangulation by Ear Clipping](https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf);
//!    the Rust port is the [earcutr](https://crates.io/crates/earcutr) crate).
//! 4. Every bending quadratic adds one *curve triangle* (start, control,
//!    end); the fragment shader keeps the side of the curve that is inside.
//!
//! Each vertex records where it comes from, the index of its source cubic
//! and the parameter `t` on it, so colors are computed on the GPU and a fill
//! depends on the points alone.

use glam::DVec2;

/// Max distance between a cubic and its quadratic approximation.
const TOLERANCE: f64 = 1e-5;
const MAX_QUADRATICS_PER_CUBIC: usize = 32;
/// Points closer than this are the same point.
const POINT_EQUALITY: f64 = 1e-8;
/// Quadratics bending less than this (relative to their chord) are lines.
const BEND_EPSILON: f64 = 1e-9;

/// Which side of a triangle the fragment shader keeps (`SIDE_*`).
const SIDE_SOLID: f32 = 0.0;
const SIDE_CONTROL: f32 = 1.0;
const SIDE_CHORD: f32 = -1.0;

/// Implicit-curve coordinates of a quadratic's (start, control, end).
const CURVE_UV: [[f32; 2]; 3] = [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]];

/// One vertex of a fill (`FILL_VERTEX_DTYPE`, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FillVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub side: f32,
    pub curve: u32,
    pub t: f32,
    pub pad: f32,
}

/// Where a point of a quadratic lies on the original cubics.
#[derive(Clone, Copy)]
struct Source {
    curve: u32,
    t: f64,
}

#[derive(Clone, Copy)]
struct Quadratic {
    start: DVec2,
    control: DVec2,
    end: DVec2,
    start_source: Source,
    control_source: Source,
    end_source: Source,
}

impl Quadratic {
    fn reversed(&self) -> Quadratic {
        Quadratic {
            start: self.end,
            control: self.control,
            end: self.start,
            start_source: self.end_source,
            control_source: self.control_source,
            end_source: self.start_source,
        }
    }

    /// Signed bend: > 0 when the control point is left of the chord.
    fn bend(&self) -> f64 {
        let chord = self.end - self.start;
        let length_sq = chord.length_squared();
        if length_sq > 0.0 {
            chord.perp_dot(self.control - self.start) / length_sq
        } else {
            0.0
        }
    }

    fn sample(&self, t: f64) -> DVec2 {
        let s = 1.0 - t;
        s * s * self.start + 2.0 * s * t * self.control + t * t * self.end
    }
}

type Ring = Vec<Quadratic>;

/// The fill of a path in the RBezierPath layout (four 2D points per cubic,
/// a new subpath wherever a curve does not start where the last one
/// ended) as a non-indexed triangle list. Open subpaths are closed with a
/// straight line.
pub fn tessellate_fill(points: &[DVec2]) -> Vec<FillVertex> {
    let curves: Vec<[DVec2; 4]> = points
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    let rings: Vec<Ring> = subpath_ranges(&curves)
        .into_iter()
        .map(|(start, stop)| ring(&curves[start..stop], start))
        .filter(|ring| !ring.is_empty())
        .collect();
    let mut vertices = Vec::new();
    for (outer, holes) in classify_rings(rings) {
        add_polygon(&mut vertices, &outer, &holes);
    }
    vertices
}

// --------------------------------------------------------------------------
// Step 1: subpaths and cubic -> quadratic conversion
// --------------------------------------------------------------------------

/// `[start, stop)` curve ranges of the continuous subpaths.
fn subpath_ranges(curves: &[[DVec2; 4]]) -> Vec<(usize, usize)> {
    if curves.is_empty() {
        return Vec::new();
    }
    let mut breaks = vec![0];
    for index in 1..curves.len() {
        if curves[index][0].distance(curves[index - 1][3]) > POINT_EQUALITY {
            breaks.push(index);
        }
    }
    breaks.push(curves.len());
    breaks.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

/// The closed ring of quadratics of one subpath (empty if it has no area).
fn ring(curves: &[[DVec2; 4]], first_curve: usize) -> Ring {
    let mut ring: Ring = curves
        .iter()
        .enumerate()
        .flat_map(|(index, curve)| cubic_to_quadratics(curve, (first_curve + index) as u32))
        .collect();
    let (first, last) = (ring[0], ring[ring.len() - 1]);
    if last.end.distance(first.start) > POINT_EQUALITY {
        // The closing line is straight: its control point never becomes a
        // vertex, so any source does for it.
        ring.push(Quadratic {
            start: last.end,
            control: (last.end + first.start) / 2.0,
            end: first.start,
            start_source: last.end_source,
            control_source: last.end_source,
            end_source: first.start_source,
        });
    }
    if signed_area(&ring).abs() > 1e-12 {
        ring
    } else {
        Vec::new()
    }
}

/// One cubic as quadratics whose error stays below `TOLERANCE`. A single
/// quadratic with control `(3 (p1 + p2) - p0 - p3) / 4` is off by at most
/// `sqrt(3) / 36 * |p3 - 3 p2 + 3 p1 - p0|`; `n` pieces divide that by `n^3`
/// (<http://caffeineowl.com/graphics/2d/vectorial/cubic2quad01.html>).
fn cubic_to_quadratics(curve: &[DVec2; 4], curve_index: u32) -> Vec<Quadratic> {
    let [p0, p1, p2, p3] = *curve;
    let error = 3f64.sqrt() / 36.0 * (p3 - 3.0 * p2 + 3.0 * p1 - p0).length();
    let count =
        ((error / TOLERANCE).powf(1.0 / 3.0).ceil() as usize).clamp(1, MAX_QUADRATICS_PER_CUBIC);
    (0..count)
        .map(|index| {
            let (t0, t1) = (
                index as f64 / count as f64,
                (index + 1) as f64 / count as f64,
            );
            // The sub-cubic on [t0, t1] from its ends and end tangents.
            let (q0, q3) = (bezier(curve, t0), bezier(curve, t1));
            let q1 = q0 + (t1 - t0) / 3.0 * bezier_derivative(curve, t0);
            let q2 = q3 - (t1 - t0) / 3.0 * bezier_derivative(curve, t1);
            let source = |t| Source {
                curve: curve_index,
                t,
            };
            Quadratic {
                start: q0,
                control: (3.0 * (q1 + q2) - q0 - q3) / 4.0,
                end: q3,
                start_source: source(t0),
                control_source: source((t0 + t1) / 2.0),
                end_source: source(t1),
            }
        })
        .collect()
}

fn bezier(c: &[DVec2; 4], t: f64) -> DVec2 {
    let s = 1.0 - t;
    s * s * s * c[0] + 3.0 * s * s * t * c[1] + 3.0 * s * t * t * c[2] + t * t * t * c[3]
}

fn bezier_derivative(c: &[DVec2; 4], t: f64) -> DVec2 {
    let s = 1.0 - t;
    3.0 * (s * s * (c[1] - c[0]) + 2.0 * s * t * (c[2] - c[1]) + t * t * (c[3] - c[2]))
}

// --------------------------------------------------------------------------
// Step 2: outer rings, holes and orientation
// --------------------------------------------------------------------------

/// Rings grouped into `(outer, holes)` by nesting depth (even-odd rule),
/// outer rings counterclockwise and holes clockwise, so the filled region
/// is always on the left of a ring.
fn classify_rings(rings: Vec<Ring>) -> Vec<(Ring, Vec<Ring>)> {
    let outlines: Vec<Vec<DVec2>> = rings
        .iter()
        .map(|ring| {
            ring.iter()
                .flat_map(|q| [0.0, 0.25, 0.5, 0.75].map(|t| q.sample(t)))
                .collect()
        })
        .collect();
    let containers: Vec<Vec<usize>> = rings
        .iter()
        .enumerate()
        .map(|(i, ring)| {
            let probe = ring[0].sample(0.5);
            (0..rings.len())
                .filter(|&j| j != i && contains(&outlines[j], probe))
                .collect()
        })
        .collect();
    let depth: Vec<usize> = containers.iter().map(Vec::len).collect();

    // Outer rings first, in ring order; then each hole goes to the
    // innermost outer ring around it. Rings that cross (a path in the middle
    // of a morph) can disagree about nesting: a hole without an outer ring
    // around it is filled as an outer ring of its own.
    let mut groups: Vec<(usize, Ring, Vec<Ring>)> = Vec::new();
    for (i, ring) in rings.iter().enumerate() {
        if depth[i].is_multiple_of(2) {
            groups.push((i, oriented(ring, true), Vec::new()));
        }
    }
    for (i, ring) in rings.iter().enumerate() {
        if depth[i] % 2 == 1 {
            let parent = containers[i]
                .iter()
                .filter(|&&j| groups.iter().any(|group| group.0 == j))
                .max_by_key(|&&j| depth[j])
                .copied();
            match parent {
                Some(parent) => {
                    let group = groups.iter_mut().find(|group| group.0 == parent).unwrap();
                    group.2.push(oriented(ring, false));
                }
                None => groups.push((i, oriented(ring, true), Vec::new())),
            }
        }
    }
    groups
        .into_iter()
        .map(|(_, outer, holes)| (outer, holes))
        .collect()
}

fn oriented(ring: &Ring, counter_clockwise: bool) -> Ring {
    if (signed_area(ring) > 0.0) == counter_clockwise {
        ring.clone()
    } else {
        ring.iter().rev().map(Quadratic::reversed).collect()
    }
}

/// The exact signed area of a closed ring of quadratics (counterclockwise
/// positive): the chord's shoelace term plus the parabolic segment, 2/3 of
/// the triangle (start, control, end).
fn signed_area(ring: &Ring) -> f64 {
    ring.iter()
        .map(|q| {
            q.start.perp_dot(q.end) / 2.0
                + (2.0 / 3.0) * (q.control - q.start).perp_dot(q.end - q.start) / 2.0
        })
        .sum()
}

/// Even-odd point-in-polygon test.
fn contains(outline: &[DVec2], point: DVec2) -> bool {
    let mut inside = false;
    for (index, a) in outline.iter().enumerate() {
        let b = outline[(index + 1) % outline.len()];
        if (a.y > point.y) != (b.y > point.y) {
            let x_at_y = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if point.x < x_at_y {
                inside = !inside;
            }
        }
    }
    inside
}

// --------------------------------------------------------------------------
// Step 3 + 4: earcut polygons and curve triangles
// --------------------------------------------------------------------------

fn add_polygon(vertices: &mut Vec<FillVertex>, outer: &Ring, holes: &[Ring]) {
    let mut corners: Vec<(DVec2, Source)> = Vec::new();
    let mut hole_starts = Vec::new();
    for (index, ring) in std::iter::once(outer).chain(holes).enumerate() {
        if index > 0 {
            hole_starts.push(corners.len());
        }
        for q in ring {
            corners.push((q.start, q.start_source));
            let bend = q.bend();
            if bend > BEND_EPSILON {
                // Concave: the ring walks around the control point.
                corners.push((q.control, q.control_source));
                add_curve_triangle(vertices, q, SIDE_CONTROL);
            } else if bend < -BEND_EPSILON {
                // Convex: the curve triangle adds the bulge.
                add_curve_triangle(vertices, q, SIDE_CHORD);
            }
        }
    }
    let flat: Vec<f64> = corners.iter().flat_map(|(p, _)| [p.x, p.y]).collect();
    let indices = earcutr::earcut(&flat, &hole_starts, 2).unwrap_or_default();
    for index in indices {
        let (position, source) = corners[index];
        vertices.push(vertex(position, [0.0, 0.0], SIDE_SOLID, source));
    }
}

fn add_curve_triangle(vertices: &mut Vec<FillVertex>, q: &Quadratic, side: f32) {
    for (position, source, uv) in [
        (q.start, q.start_source, CURVE_UV[0]),
        (q.control, q.control_source, CURVE_UV[1]),
        (q.end, q.end_source, CURVE_UV[2]),
    ] {
        vertices.push(vertex(position, uv, side, source));
    }
}

fn vertex(position: DVec2, uv: [f32; 2], side: f32, source: Source) -> FillVertex {
    FillVertex {
        position: [position.x as f32, position.y as f32],
        uv,
        side,
        curve: source.curve,
        t: source.t as f32,
        pad: 0.0,
    }
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
            .filter(|t| t[0].side == SIDE_SOLID)
            .map(|t| {
                let p = |v: &FillVertex| DVec2::new(v.position[0] as f64, v.position[1] as f64);
                ((p(&t[1]) - p(&t[0])).perp_dot(p(&t[2]) - p(&t[0])) / 2.0).abs()
            })
            .sum()
    }

    #[test]
    fn a_square_with_a_hole_fills_the_ring_between() {
        let mut points = square(4.0, DVec2::ZERO, false);
        points.extend(square(2.0, DVec2::new(1.0, 1.0), false)); // same direction: still a hole
        let vertices = tessellate_fill(&points);
        assert!((area(&vertices) - 12.0).abs() < 1e-9);
    }

    #[test]
    fn crossing_rings_are_filled_on_their_own() {
        let mut points = square(2.0, DVec2::ZERO, false);
        points.extend(square(2.0, DVec2::new(1.0, 1.0), true));
        assert!((area(&tessellate_fill(&points)) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn nothing_without_area() {
        let line = vec![DVec2::ZERO, DVec2::X, DVec2::X * 2.0, DVec2::X * 3.0];
        assert!(tessellate_fill(&line).is_empty());
        assert!(tessellate_fill(&[]).is_empty());
    }
}
