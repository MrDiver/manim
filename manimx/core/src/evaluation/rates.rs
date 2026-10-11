//! Rate functions: how an animation's linear progress becomes its eased
//! progress. Ports `linear` and `smooth` from manimx/animation/signals/rates.py;
//! any other rate function stays in Python and is called back.

use pyo3::prelude::*;

pub enum Rate {
    Linear,
    /// `smooth(t, inflection)`.
    Smooth {
        inflection: f64,
    },
    /// A rate function the core does not know.
    Python(Py<PyAny>),
}

impl Rate {
    /// The rate at linear progress `t` in [0, 1].
    pub fn at(&self, py: Python<'_>, t: f64) -> PyResult<f64> {
        match self {
            Rate::Linear => Ok(t),
            Rate::Smooth { inflection } => Ok(smooth(t, *inflection)),
            Rate::Python(function) => function.bind(py).call1((t,))?.extract::<f64>(),
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn smooth(t: f64, inflection: f64) -> f64 {
    let error = sigmoid(-inflection / 2.0);
    ((sigmoid(inflection * (t - 0.5)) - error) / (1.0 - 2.0 * error)).clamp(0.0, 1.0)
}
