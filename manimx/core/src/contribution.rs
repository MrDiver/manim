//! What one animation does to one signal at a given progress: the data
//! form of `Animation.contribution()` (manimx/animation/contributions.py
//! lists the same cases on the Python side).
//!
//! The progress `alpha` is already rate-mapped. `start` is the signal's
//! value when the animation starts (`Entry.start_value`).

use pyo3::prelude::*;

use glam::{DQuat, DVec3};

use std::sync::Arc;

use crate::paths;
use crate::value::{Kind, Value};

pub enum Contribution {
    /// From the start value to `value` (`SignalChange` with ABSOLUTE).
    To(Value),
    /// The fraction `alpha` of a relative change (`SignalChange` with
    /// RELATIVE).
    By(Value),
    /// `base * alpha`, or `base * (1 - alpha)` when `remaining`; the base
    /// is `factor`, or the start value when there is none. Fades and
    /// growing: `FadeIn` is `Ramp { factor: Some(1.0), remaining: false }`.
    Ramp {
        factor: Option<f64>,
        remaining: bool,
    },
    /// `base ** alpha` (`Indicate`).
    Power { base: f64 },
    /// A rotation by `angle * alpha` about `axis` (`Rotate`).
    Turn { axis: [f64; 3], angle: f64 },
    /// Recorded values of a `Simulation`, interpolated.
    Keyframes(Keyframes),
    /// From `start` to `end` while the progress runs through `window`
    /// (`Morph`, the style changes of `Write`).
    Between {
        start: Value,
        end: Value,
        window: Window,
    },
    /// The part of a path drawn so far (`Create`, `Write`); backwards when
    /// `reverse` (`Uncreate`).
    Partial {
        points: Value,
        window: Window,
        reverse: bool,
    },
    /// No data form: `animation.contribution(signal, alpha, start)` in Python.
    Python {
        animation: Py<PyAny>,
        signal: Py<PyAny>,
    },
}

/// A stretch of an animation's progress, from `lower` to `upper`.
pub struct Window {
    pub lower: f64,
    pub upper: f64,
}

impl Window {
    /// `window()` in contributions.py: how far `alpha` is through the
    /// window, in [0, 1]; a window of no width switches at its end.
    fn progress(&self, alpha: f64) -> f64 {
        if self.upper <= self.lower {
            return if alpha >= self.upper { 1.0 } else { 0.0 };
        }
        ((alpha - self.lower) / (self.upper - self.lower)).clamp(0.0, 1.0)
    }
}

/// `Simulation.times` and the recorded values of one signal.
pub struct Keyframes {
    pub run_time: f64,
    pub times: Vec<f64>,
    pub values: Vec<Value>,
}

impl Contribution {
    pub fn at(
        &self,
        py: Python<'_>,
        kind: &Kind,
        alpha: f64,
        start: &Value,
        start_py: &Py<PyAny>,
    ) -> PyResult<Value> {
        match self {
            Contribution::To(value) => kind.interpolate(py, start, value, alpha),
            Contribution::By(change) => kind.scale(py, change, alpha),
            Contribution::Ramp { factor, remaining } => {
                let base = match factor {
                    Some(factor) => *factor,
                    None => start.scalar()?,
                };
                Ok(Value::Scalar(if *remaining {
                    base * (1.0 - alpha)
                } else {
                    base * alpha
                }))
            }
            Contribution::Power { base } => Ok(Value::Scalar(base.powf(alpha))),
            Contribution::Turn { axis, angle } => Ok(Value::Quat(DQuat::from_axis_angle(
                DVec3::from(*axis).normalize(),
                angle * alpha,
            ))),
            Contribution::Keyframes(keyframes) => keyframes.at(py, kind, alpha),
            Contribution::Between { start, end, window } => {
                kind.interpolate(py, start, end, window.progress(alpha))
            }
            Contribution::Partial {
                points,
                window,
                reverse,
            } => {
                let progress = window.progress(alpha);
                let upper = if *reverse { 1.0 - progress } else { progress };
                Ok(Value::Array(Arc::new(paths::partial_path(
                    points.array()?,
                    0.0,
                    upper,
                ))))
            }
            Contribution::Python { animation, signal } => {
                let result = animation
                    .bind(py)
                    .call_method1("contribution", (signal, alpha, start_py))?;
                kind.value_from_py(&result)
            }
        }
    }
}

impl Keyframes {
    /// `Simulation.contribution()`: the keyframe before the simulated time
    /// and how far past it, then the values between.
    fn at(&self, py: Python<'_>, kind: &Kind, alpha: f64) -> PyResult<Value> {
        let t = alpha * self.run_time;
        let last = self.times.len() - 2;
        let after = self.times.partition_point(|&time| time <= t);
        let index = after.saturating_sub(1).min(last);
        let (t0, t1) = (self.times[index], self.times[index + 1]);
        let fraction = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        let (start, end) = (&self.values[index], &self.values[index + 1]);
        if start.same(end) {
            return Ok(start.clone()); // unchanged between the keyframes
        }
        if let (Value::Array(a), Value::Array(b)) = (start, end) {
            if a.shape != b.shape {
                // The path changed its point count: no blending.
                return Ok(if fraction < 1.0 {
                    start.clone()
                } else {
                    end.clone()
                });
            }
        }
        kind.interpolate(py, start, end, fraction)
    }
}
