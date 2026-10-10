//! The part of a path drawn so far. Ports manimx/animation/paths.py with
//! the same operations in the same order, so both agree; keep them in step.
//!
//! Points are rows of `dimension` floats, four per cubic curve.

use crate::value::Array;

const POINTS_PER_CURVE: usize = 4;

type Point = Vec<f64>;

/// `p + (q - p) * t`, per component.
fn lerp(p: &[f64], q: &[f64], t: f64) -> Point {
    p.iter().zip(q).map(|(a, b)| a + (b - a) * t).collect()
}

/// `split_curve`: the curve cut at `t` (de Casteljau).
fn split_curve(curve: &[Point], t: f64) -> ([Point; 4], [Point; 4]) {
    let (p0, p1, p2, p3) = (&curve[0], &curve[1], &curve[2], &curve[3]);
    let a = lerp(p0, p1, t);
    let b = lerp(p1, p2, t);
    let c = lerp(p2, p3, t);
    let d = lerp(&a, &b, t);
    let e = lerp(&b, &c, t);
    let f = lerp(&d, &e, t);
    ([p0.clone(), a, d, f.clone()], [f, e, c, p3.clone()])
}

/// `partial_curve`: the part of one curve between `lower` and `upper`.
fn partial_curve(curve: &[Point], lower: f64, upper: f64) -> Vec<Point> {
    if lower >= 1.0 {
        return vec![curve[3].clone(); POINTS_PER_CURVE];
    }
    let (_, rest) = split_curve(curve, lower);
    let (part, _) = split_curve(&rest, (upper - lower) / (1.0 - lower));
    part.to_vec()
}

/// `_curve_at`: the curve the fraction `alpha` of a path lies on, and how
/// far along it.
fn curve_at(count: usize, alpha: f64) -> (usize, f64) {
    if alpha >= 1.0 {
        return (count - 1, 1.0);
    }
    if alpha <= 0.0 {
        return (0, 0.0);
    }
    let position = count as f64 * alpha;
    (position as usize, position - position.floor())
}

/// `partial_path`: the path from `lower` to `upper` (fractions of its curves).
pub fn partial_path(points: &Array, lower: f64, upper: f64) -> Array {
    let dimension = points.cols();
    let count = points.rows() / POINTS_PER_CURVE;
    if lower <= 0.0 && upper >= 1.0 {
        return Array { shape: points.shape.clone(), data: points.data.clone() };
    }
    if count == 0 || upper <= lower {
        return Array { shape: vec![0, dimension], data: Vec::new() };
    }
    let rows: Vec<Point> = points.data.chunks_exact(dimension).map(|row| row.to_vec()).collect();
    let curve = |index: usize| &rows[index * POINTS_PER_CURVE..(index + 1) * POINTS_PER_CURVE];
    let (lower_index, lower_residue) = curve_at(count, lower);
    let (upper_index, upper_residue) = curve_at(count, upper);
    let mut result: Vec<Point> = Vec::new();
    if lower_index == upper_index {
        result.extend(partial_curve(curve(lower_index), lower_residue, upper_residue));
    } else {
        result.extend(partial_curve(curve(lower_index), lower_residue, 1.0));
        for index in lower_index + 1..upper_index {
            result.extend_from_slice(curve(index));
        }
        result.extend(partial_curve(curve(upper_index), 0.0, upper_residue));
    }
    Array { shape: vec![result.len(), dimension], data: result.concat() }
}
