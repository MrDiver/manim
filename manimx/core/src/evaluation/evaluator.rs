//! The evaluator: every signal of a prepared scene in one table, set to
//! its value at any time. The Rust side of `ScenePlan`
//! (manimx/animation/plan.py), and the counterpart of
//! `PreparedScene.evaluate()`:
//!
//! 1. the two time signals are set (`scene_time`, `global_time`);
//! 2. every track folds its entries at the compiled time;
//! 3. every derived signal whose inputs changed is computed again, in
//!    dependency order.
//!
//! Each slot keeps a version that increases exactly when its value
//! changes, like `Signal.version`, so later stages can skip what did not
//! change.

use pyo3::exceptions::PyValueError;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::evaluation::contribution::{Contribution, Keyframes, Window};
use crate::evaluation::paths;
use crate::evaluation::rates::Rate;
use crate::evaluation::track::{Entry, Track};
use crate::evaluation::value::{Kind, Value};

/// One signal's current value and version (its kind is in
/// `Evaluator::kinds`, shared with every `FrameState`).
struct Slot {
    value: Value,
    version: u64,
    /// The value as a Python object, made on first request after a change.
    value_py: Option<Py<PyAny>>,
}

/// How a derived signal computes its value from its inputs.
enum Derivation {
    /// `input + offset` (`PositionMixin.follow()`).
    Offset(f64),
    /// `partial_path(points, start, end)`: the drawn part of a path
    /// (`DrawnPart`, the `drawn_points` of Shape2D and Curve3D).
    DrawnPart,
    /// A Python function of the input values (`derive()`).
    Python(Py<PyAny>),
}

struct Derived {
    slot: usize,
    inputs: Vec<usize>,
    derivation: Derivation,
    /// The input versions it was last computed from (empty: never).
    seen: Vec<u64>,
}

#[pyclass(module = "manimx._core")]
pub struct Evaluator {
    kinds: Arc<Vec<Kind>>,
    slots: Vec<Slot>,
    tracks: Vec<Track>,
    /// In dependency order: inputs before the signals computed from them.
    derived: Vec<Derived>,
    scene_time: usize,
    global_time: usize,
}

#[pymethods]
impl Evaluator {
    /// `kinds[i]` is `(code, kind)`: the kind's name for the kinds the core
    /// knows, and the `SignalKind` object; `values[i]` is the signal's
    /// current value.
    #[new]
    fn new(
        kinds: Vec<(String, Py<PyAny>)>,
        values: Vec<Bound<'_, PyAny>>,
        scene_time: usize,
        global_time: usize,
    ) -> PyResult<Self> {
        if kinds.len() != values.len() {
            return Err(PyValueError::new_err("one kind per value"));
        }
        let kinds: Vec<Kind> = kinds
            .into_iter()
            .map(|(code, kind)| Kind::from_code(&code, kind))
            .collect();
        let mut slots = Vec::with_capacity(kinds.len());
        for (kind, value) in kinds.iter().zip(values) {
            slots.push(Slot {
                value: kind.value_from_py(&value)?,
                version: 1,
                value_py: None,
            });
        }
        Ok(Evaluator {
            kinds: Arc::new(kinds),
            slots,
            tracks: Vec::new(),
            derived: Vec::new(),
            scene_time,
            global_time,
        })
    }

    /// Add the track of signal `slot`. Each entry is `(start, duration,
    /// rate, relative, start_value, contribution)` with `rate` and
    /// `contribution` tagged tuples (see manimx/animation/plan.py).
    fn add_track(
        &mut self,
        slot: usize,
        base: Bound<'_, PyAny>,
        entries: Vec<Bound<'_, PyTuple>>,
    ) -> PyResult<()> {
        let kind = &self.kinds[slot];
        let base = kind.value_from_py(&base)?;
        let mut parsed = Vec::with_capacity(entries.len());
        for entry in entries {
            let start_value_py = entry.get_item(4)?;
            parsed.push(Entry {
                start: entry.get_item(0)?.extract()?,
                duration: entry.get_item(1)?.extract()?,
                rate: parse_rate(&entry.get_item(2)?.cast_into::<PyTuple>()?)?,
                relative: entry.get_item(3)?.extract()?,
                start_value: kind.value_from_py(&start_value_py)?,
                start_value_py: start_value_py.unbind(),
                contribution: parse_contribution(
                    kind,
                    &entry.get_item(5)?.cast_into::<PyTuple>()?,
                )?,
            });
        }
        self.tracks.push(Track {
            slot,
            base,
            entries: parsed,
        });
        Ok(())
    }

    /// Add derived signal `slot`, computed from `inputs` by `derivation`:
    /// `("offset", d)`, `("drawn_part",)` or `("python", function)`. Add them in dependency order.
    fn add_derived(
        &mut self,
        slot: usize,
        inputs: Vec<usize>,
        derivation: Bound<'_, PyTuple>,
    ) -> PyResult<()> {
        let tag: String = derivation.get_item(0)?.extract()?;
        let derivation = match tag.as_str() {
            "offset" => Derivation::Offset(derivation.get_item(1)?.extract()?),
            "drawn_part" => Derivation::DrawnPart,
            "python" => Derivation::Python(derivation.get_item(1)?.unbind()),
            _ => return Err(PyValueError::new_err(format!("unknown derivation {tag:?}"))),
        };
        self.derived.push(Derived {
            slot,
            inputs,
            derivation,
            seen: Vec::new(),
        });
        Ok(())
    }

    /// Set every signal to its value at compiled time `t`.
    fn evaluate(
        &mut self,
        py: Python<'_>,
        t: f64,
        scene_time: f64,
        global_time: f64,
    ) -> PyResult<()> {
        self.set(py, self.scene_time, Value::Scalar(scene_time))?;
        self.set(py, self.global_time, Value::Scalar(global_time))?;
        for index in 0..self.tracks.len() {
            let track = &self.tracks[index];
            let slot = track.slot;
            let value = track.value_at(py, &self.kinds[slot], t)?;
            let value = self.kinds[slot].validate(py, value)?;
            self.set(py, slot, value)?;
        }
        for index in 0..self.derived.len() {
            self.update_derived(py, index)?;
        }
        Ok(())
    }

    /// The values of every slot right now, kept while the evaluator moves
    /// on to other times (motion blur evaluates every sub-frame first).
    fn state(&self) -> FrameState {
        FrameState {
            kinds: self.kinds.clone(),
            values: Arc::new(self.slots.iter().map(|slot| slot.value.clone()).collect()),
            versions: Arc::new(self.slots.iter().map(|slot| slot.version).collect()),
        }
    }

    /// The value of signal `slot`, as the Python signal holds it.
    fn value(&mut self, py: Python<'_>, slot: usize) -> PyResult<Py<PyAny>> {
        let entry = &mut self.slots[slot];
        if entry.value_py.is_none() {
            entry.value_py = Some(self.kinds[slot].value_to_py(py, &entry.value)?);
        }
        Ok(entry.value_py.as_ref().unwrap().clone_ref(py))
    }

    /// Increases exactly when the value of signal `slot` changes.
    fn version(&self, slot: usize) -> u64 {
        self.slots[slot].version
    }

    #[getter]
    fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

impl Evaluator {
    /// Store `value` in `slot` unless it equals the current one.
    fn set(&mut self, py: Python<'_>, slot: usize, value: Value) -> PyResult<()> {
        let entry = &mut self.slots[slot];
        if self.kinds[slot].equal(py, &entry.value, &value)? {
            return Ok(());
        }
        entry.value = value;
        entry.version += 1;
        entry.value_py = None;
        Ok(())
    }

    /// Compute derived signal `index` again if any of its inputs changed.
    fn update_derived(&mut self, py: Python<'_>, index: usize) -> PyResult<()> {
        let inputs = self.derived[index].inputs.clone();
        let versions: Vec<u64> = inputs
            .iter()
            .map(|&input| self.slots[input].version)
            .collect();
        if versions == self.derived[index].seen {
            return Ok(()); // no input changed
        }
        let slot = self.derived[index].slot;
        let value = match &self.derived[index].derivation {
            Derivation::Offset(offset) => {
                Value::Scalar(self.slots[inputs[0]].value.scalar()? + offset)
            }
            Derivation::DrawnPart => {
                let start = self.slots[inputs[1]].value.scalar()?;
                let end = self.slots[inputs[2]].value.scalar()?;
                match &self.slots[inputs[0]].value {
                    // Fully drawn (nearly always): the same array, not a copy.
                    Value::Array(points) if start <= 0.0 && end >= 1.0 => {
                        Value::Array(points.clone())
                    }
                    Value::Array(points) => {
                        Value::Array(Arc::new(paths::partial_path(points, start, end)))
                    }
                    _ => return Err(PyValueError::new_err("drawn_points needs a path")),
                }
            }
            Derivation::Python(function) => {
                let function = function.clone_ref(py);
                let arguments = inputs
                    .iter()
                    .map(|&input| self.value(py, input))
                    .collect::<PyResult<Vec<_>>>()?;
                let result = function.bind(py).call1(PyTuple::new(py, arguments)?)?;
                self.kinds[slot].value_from_py(&result)?
            }
        };
        let value = self.kinds[slot].validate(py, value)?;
        self.set(py, slot, value)?;
        self.derived[index].seen = versions;
        Ok(())
    }
}

fn parse_rate(rate: &Bound<'_, PyTuple>) -> PyResult<Rate> {
    let tag: String = rate.get_item(0)?.extract()?;
    Ok(match tag.as_str() {
        "linear" => Rate::Linear,
        "smooth" => Rate::Smooth {
            inflection: rate.get_item(1)?.extract()?,
        },
        "python" => Rate::Python(rate.get_item(1)?.unbind()),
        _ => return Err(PyValueError::new_err(format!("unknown rate {tag:?}"))),
    })
}

fn parse_contribution(kind: &Kind, contribution: &Bound<'_, PyTuple>) -> PyResult<Contribution> {
    let tag: String = contribution.get_item(0)?.extract()?;
    let item = |index: usize| contribution.get_item(index);
    Ok(match tag.as_str() {
        "to" => Contribution::To(kind.value_from_py(&item(1)?)?),
        "by" => Contribution::By(kind.value_from_py(&item(1)?)?),
        "ramp" => Contribution::Ramp {
            factor: item(1)?.extract()?,
            remaining: item(2)?.extract()?,
        },
        "power" => Contribution::Power {
            base: item(1)?.extract()?,
        },
        "turn" => Contribution::Turn {
            axis: item(1)?.extract()?,
            angle: item(2)?.extract()?,
        },
        "keyframes" => {
            let values: Vec<Bound<'_, PyAny>> = item(3)?.extract()?;
            // A value recorded again unchanged is the same object: keep
            // that, so unchanged stretches are recognized without comparing.
            let mut parsed: Vec<Value> = Vec::with_capacity(values.len());
            for (index, value) in values.iter().enumerate() {
                if index > 0 && value.is(&values[index - 1]) {
                    parsed.push(parsed[index - 1].clone());
                } else {
                    parsed.push(kind.value_from_py(value)?);
                }
            }
            Contribution::Keyframes(Keyframes {
                run_time: item(1)?.extract()?,
                times: item(2)?.extract()?,
                values: parsed,
            })
        }
        "between" => Contribution::Between {
            start: kind.value_from_py(&item(1)?)?,
            end: kind.value_from_py(&item(2)?)?,
            window: Window {
                lower: item(3)?.extract()?,
                upper: item(4)?.extract()?,
            },
        },
        "partial" => Contribution::Partial {
            points: kind.value_from_py(&item(1)?)?,
            window: Window {
                lower: item(2)?.extract()?,
                upper: item(3)?.extract()?,
            },
            reverse: item(4)?.extract()?,
        },
        "python" => Contribution::Python {
            animation: item(1)?.unbind(),
            signal: item(2)?.unbind(),
        },
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown contribution {tag:?}"
            )))
        }
    })
}

/// The values and versions of all slots at one time: what a frame shows.
/// Cheap to keep: values are shared with the evaluator, not copied.
#[pyclass(module = "manimx._core", frozen)]
pub struct FrameState {
    pub kinds: Arc<Vec<Kind>>,
    pub values: Arc<Vec<Value>>,
    pub versions: Arc<Vec<u64>>,
}

#[pymethods]
impl FrameState {
    /// The given values as one frame's (every version 1): `kinds` as for
    /// `Evaluator`. `Node.snapshot()` builds the render object of a node's
    /// current values from one.
    #[new]
    fn new(kinds: Vec<(String, Py<PyAny>)>, values: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        if kinds.len() != values.len() {
            return Err(PyValueError::new_err("one kind per value"));
        }
        let kinds: Vec<Kind> = kinds
            .into_iter()
            .map(|(code, kind)| Kind::from_code(&code, kind))
            .collect();
        let values = kinds
            .iter()
            .zip(&values)
            .map(|(kind, value)| kind.value_from_py(value))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(FrameState {
            versions: Arc::new(vec![1; values.len()]),
            kinds: Arc::new(kinds),
            values: Arc::new(values),
        })
    }

    /// The value of `slot`, as the Python signal holds it.
    fn value(&self, py: Python<'_>, slot: usize) -> PyResult<Py<PyAny>> {
        self.kinds[slot].value_to_py(py, &self.values[slot])
    }
}
