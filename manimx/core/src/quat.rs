//! Quaternion arithmetic as numpy-quaternion computes it (w, x, y, z), so
//! the core's rotations match the Python reference path.

pub type Quat = [f64; 4];

pub const ONE: Quat = [1.0, 0.0, 0.0, 0.0];

/// numpy-quaternion's _QUATERNION_EPS.
const EPS: f64 = 1e-14;

pub fn multiply(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// ``a / b``, i.e. ``a * b⁻¹``.
pub fn divide(a: Quat, b: Quat) -> Quat {
    let norm = b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3];
    [
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]) / norm,
        (-a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2]) / norm,
        (-a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1]) / norm,
        (-a[0] * b[3] - a[1] * b[2] + a[2] * b[1] + a[3] * b[0]) / norm,
    ]
}

pub fn absolute(q: Quat) -> f64 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

pub fn normalized(q: Quat) -> Quat {
    let norm = absolute(q);
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

pub fn negated(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], -q[3]]
}

pub fn exp(q: Quat) -> Quat {
    let vnorm = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if vnorm > EPS {
        let s = vnorm.sin() / vnorm;
        let e = q[0].exp();
        [e * vnorm.cos(), e * s * q[1], e * s * q[2], e * s * q[3]]
    } else {
        [q[0].exp(), 0.0, 0.0, 0.0]
    }
}

pub fn log(q: Quat) -> Quat {
    let b = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if b <= EPS * q[0].abs() {
        if q[0] < 0.0 {
            // log of a negative real: a half turn about an arbitrary axis.
            return [(-q[0]).ln(), std::f64::consts::PI, 0.0, 0.0];
        }
        return [q[0].ln(), 0.0, 0.0, 0.0];
    }
    let v = b.atan2(q[0]);
    let f = v / b;
    [
        (q[0] * q[0] + b * b).ln() / 2.0,
        f * q[1],
        f * q[2],
        f * q[3],
    ]
}

pub fn scalar_power(q: Quat, s: f64) -> Quat {
    if q == [0.0; 4] {
        return if s == 0.0 { ONE } else { [0.0; 4] };
    }
    let l = log(q);
    exp([l[0] * s, l[1] * s, l[2] * s, l[3] * s])
}

/// numpy-quaternion's ``slerp_evaluate(q1, q2, tau)``.
pub fn slerp_evaluate(q1: Quat, q2: Quat, tau: f64) -> Quat {
    let d = [q1[0] - q2[0], q1[1] - q2[1], q1[2] - q2[2], q1[3] - q2[3]];
    if absolute(d) <= EPS {
        return q1;
    }
    multiply(scalar_power(divide(q2, q1), tau), q1)
}

/// ``quaternion_from_axis_angle(axis, angle)``: numpy-quaternion's
/// ``from_rotation_vector(axis / |axis| * angle)``, ``exp((0, v / 2))``.
pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Quat {
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let v = |i: usize| axis[i] / length * angle / 2.0;
    exp([0.0, v(0), v(1), v(2)])
}
