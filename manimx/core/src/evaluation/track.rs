//! One root signal's track: its base value and the entries of the
//! animations on it. Ports `SignalTrack` and `Placement.progress` from
//! manimx/animation/signals/signal_tracks.py.

use pyo3::prelude::*;

use crate::evaluation::contribution::Contribution;
use crate::evaluation::rates::Rate;
use crate::evaluation::value::{Kind, Value};

/// A placement's appearance on one track (`Entry` with its `Placement`).
pub struct Entry {
    pub start: f64,
    pub duration: f64,
    pub rate: Rate,
    /// RELATIVE changes combine with the value so far; ABSOLUTE ones
    /// replace it.
    pub relative: bool,
    /// The signal's value when the animation starts (`Entry.start_value`).
    pub start_value: Value,
    /// The same value as a Python object, for contributions called back.
    pub start_value_py: Py<PyAny>,
    pub contribution: Contribution,
}

impl Entry {
    /// `Placement.progress`: the rate-mapped progress at time `t`.
    fn progress(&self, py: Python<'_>, t: f64) -> PyResult<f64> {
        if self.duration <= 0.0 {
            return Ok(1.0); // instantaneous: done from its start time on
        }
        // At and after the end exactly 1 (`Placement.progress`): the
        // quotient can round to just below 1 and miss a switch at the end.
        let linear = if t >= self.start + self.duration {
            1.0
        } else {
            ((t - self.start) / self.duration).clamp(0.0, 1.0)
        };
        self.rate.at(py, linear)
    }
}

pub struct Track {
    /// Index of the signal in the evaluator's slots.
    pub slot: usize,
    pub base: Value,
    /// In authoring order.
    pub entries: Vec<Entry>,
}

impl Track {
    /// `SignalTrack.value_at(t)`: the entries that have started, folded in
    /// authoring order.
    pub fn value_at(&self, py: Python<'_>, kind: &Kind, t: f64) -> PyResult<Value> {
        let mut value = self.base.clone();
        for entry in &self.entries {
            if entry.start > t {
                continue;
            }
            let alpha = entry.progress(py, t)?;
            let contribution = entry.contribution.at(
                py,
                kind,
                alpha,
                &entry.start_value,
                &entry.start_value_py,
            )?;
            value = if entry.relative {
                kind.combine(py, &value, &contribution)?
            } else {
                kind.validate(py, contribution)?
            };
        }
        Ok(value)
    }
}
