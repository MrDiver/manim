//! A path's stroke records from its points and stroke style: the curve
//! records linked inside each subpath and the per-point position, width
//! and color.
//!
//! The style is one color or a list (rows of RGB), opacities and widths,
//! joints (one, or a list spread over the curves) and a cap. Lists are
//! spread over the points like `resample` does: linear interpolation of
//! the list stretched over them, as numpy's `interp` computes it.

use crate::render::pack::records::{StrokeCurve, NO_NEIGHBOR, STROKE_POINT_SIZE};
use crate::robjects::values::{
    POINTS_PER_CURVE, STROKE_WIDTH_SCALE as WIDTH_SCALE, TOLERANCE_FOR_POINT_EQUALITY as TOLERANCE,
};

/// A stroke's style, lists as given (a single value is a list of one).
pub struct StrokeStyle<'a> {
    /// RGB rows.
    pub colors: &'a [[f64; 3]],
    pub opacities: &'a [f64],
    pub widths: &'a [f64],
    /// `JointType` values, spread over the curves.
    pub joints: &'a [u32],
    /// The `CapStyle` value.
    pub cap: u32,
}

/// What `stroke_records` returns: curve records (local point indices),
/// point records, whether the stroke shows at all and whether it is
/// opaque.
pub struct Stroke {
    pub curves: Vec<u8>,
    pub points: Vec<u8>,
    pub visible: bool,
    pub opaque: bool,
}

/// `STROKE_POINT_DTYPE`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StrokePoint {
    position: [f32; 3],
    width: f32,
    color: [f32; 4],
}

const _: () = assert!(size_of::<StrokePoint>() == STROKE_POINT_SIZE);

/// The records of a stroke over `points` (`dimension` 2 or 3 numbers per
/// point, a multiple of four points).
pub fn stroke_records(points: &[f64], dimension: usize, style: &StrokeStyle) -> Stroke {
    // style.stroke: opacities clamped to [0, 1], widths not negative.
    let opacities: Vec<f64> = style.opacities.iter().map(|o| o.clamp(0.0, 1.0)).collect();
    let widths: Vec<f64> = style.widths.iter().map(|w| w.max(0.0)).collect();
    let visible = widths.iter().any(|&w| w > 0.0) && opacities.iter().any(|&o| o > 0.0);
    if !visible {
        return Stroke {
            curves: Vec::new(),
            points: Vec::new(),
            visible: false,
            opaque: false,
        };
    }
    let opaque = opacities.iter().all(|&o| o >= 1.0);
    let count = points.len() / dimension;
    let point = |index: usize| -> [f64; 3] {
        let at = &points[index * dimension..(index + 1) * dimension];
        [at[0], at[1], if dimension == 3 { at[2] } else { 0.0 }]
    };

    let curves = link_curves(count / POINTS_PER_CURVE, &point, style);

    let widths = resample(&widths, count);
    let opacities = resample(&opacities, count);
    let channels: Vec<Vec<f64>> = (0..3)
        .map(|channel| {
            let values: Vec<f64> = style.colors.iter().map(|color| color[channel]).collect();
            resample(&values, count)
        })
        .collect();
    let mut records: Vec<StrokePoint> = (0..count)
        .map(|index| {
            let [x, y, z] = point(index);
            StrokePoint {
                position: [x as f32, y as f32, z as f32],
                width: (widths[index] * WIDTH_SCALE) as f32,
                color: [
                    channels[0][index] as f32,
                    channels[1][index] as f32,
                    channels[2][index] as f32,
                    opacities[index] as f32,
                ],
            }
        })
        .collect();
    // Anchors shared by consecutive curves get the mean of their widths
    // and colors (in float32, like the numpy record array), so a resampled
    // list shows no step at the joints.
    for curve in &curves {
        if curve.next_point == NO_NEIGHBOR {
            continue;
        }
        let end = curve.first_point as usize + POINTS_PER_CURVE - 1;
        let start = curve.next_point as usize;
        let width = (records[end].width + records[start].width) / 2.0;
        records[end].width = width;
        records[start].width = width;
        for channel in 0..4 {
            let mean = (records[end].color[channel] + records[start].color[channel]) / 2.0;
            records[end].color[channel] = mean;
            records[start].color[channel] = mean;
        }
    }
    Stroke {
        curves: bytemuck::cast_slice(&curves).to_vec(),
        points: bytemuck::cast_slice(&records).to_vec(),
        visible,
        opaque,
    }
}

/// One record per curve with the previous and next curve
/// of its subpath; a closed subpath links its last curve to its first.
fn link_curves(
    count: usize,
    point: &impl Fn(usize) -> [f64; 3],
    style: &StrokeStyle,
) -> Vec<StrokeCurve> {
    let joints = spread_joints(style.joints, count);
    let start = |curve: usize| (curve * POINTS_PER_CURVE) as u32;
    let mut curves: Vec<StrokeCurve> = (0..count)
        .map(|curve| StrokeCurve {
            first_point: start(curve),
            prev_point: NO_NEIGHBOR,
            next_point: NO_NEIGHBOR,
            start_joint: 0,
            end_joint: joints[curve],
            start_cap: style.cap,
            end_cap: style.cap,
            _pad: 0,
        })
        .collect();
    let mut begin = 0;
    while begin < count {
        // A subpath continues while a curve starts where the last ended.
        let mut end = begin + 1;
        while end < count
            && distance(
                point(end * POINTS_PER_CURVE),
                point(end * POINTS_PER_CURVE - 1),
            ) <= TOLERANCE
        {
            end += 1;
        }
        for curve in begin..end {
            if curve > begin {
                curves[curve].prev_point = start(curve - 1) as i32;
                curves[curve].start_joint = joints[curve - 1];
            }
            if curve < end - 1 {
                curves[curve].next_point = start(curve + 1) as i32;
            }
        }
        let first = point(begin * POINTS_PER_CURVE);
        let last = point(end * POINTS_PER_CURVE - 1);
        if distance(first, last) <= TOLERANCE {
            curves[begin].prev_point = start(end - 1) as i32;
            curves[begin].start_joint = joints[end - 1];
            curves[end - 1].next_point = start(begin) as i32;
        }
        begin = end;
    }
    curves
}

/// `Stroke.joints`: the joint list spread over `count` curves, nearest
/// entry (numpy's rint rounds halves to even).
fn spread_joints(joints: &[u32], count: usize) -> Vec<u32> {
    let positions = linspace(0.0, (joints.len() - 1) as f64, count);
    positions
        .iter()
        .map(|&at| joints[at.round_ties_even() as usize])
        .collect()
}

/// `resample`: `values` spread over `count` slots (one value repeated,
/// as many returned as they are, else linear interpolation).
pub fn resample(values: &[f64], count: usize) -> Vec<f64> {
    if values.len() == count {
        return values.to_vec();
    }
    if values.len() == 1 {
        return vec![values[0]; count];
    }
    let source = linspace(0.0, 1.0, values.len());
    linspace(0.0, 1.0, count)
        .iter()
        .map(|&at| interp(at, &source, values))
        .collect()
}

/// `np.linspace(start, stop, count)`.
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut values: Vec<f64> = (0..count)
        .map(|index| index as f64 * step + start)
        .collect();
    values[count - 1] = stop;
    values
}

/// `np.interp` at `x` for increasing `xs`.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let last = xs.len() - 1;
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[last] {
        return ys[last];
    }
    let j = xs.partition_point(|&value| value <= x) - 1;
    if xs[j] == x {
        return ys[j];
    }
    let slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
    slope * (x - xs[j]) + ys[j]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
