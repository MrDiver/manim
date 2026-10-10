//! Signal values and the operations of their kinds, ported from
//! manimx/animation/kinds.py. Every operation computes what the Python
//! kind computes, in the same order, so both paths agree.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use glam::DQuat;

/// A ``SignalKind``. Kinds the core does not know keep their Python
/// object, and their values are Python objects the kind operates on.
pub enum Kind {
    Scalar,
    Factor,
    Rotation,
    Color,
    Colors,
    Scalars,
    Path,
    Array,
    Discrete,
    Python(Py<PyAny>),
}

/// A float array of one or two dimensions, row-major.
#[derive(Debug, PartialEq)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Array {
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        if self.shape.len() > 1 {
            self.shape[1]
        } else {
            1
        }
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Array {
        Array {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    fn zip(&self, other: &Array, f: impl Fn(f64, f64) -> f64) -> Array {
        Array {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }
}

#[derive(Clone)]
pub enum Value {
    Scalar(f64),
    Quat(DQuat),
    /// A COLOR: 3 or 4 channels.
    Tuple(Arc<Vec<f64>>),
    Array(Arc<Array>),
    Object(Arc<Py<PyAny>>),
}

fn type_error(kind: &str, what: &str) -> PyErr {
    PyTypeError::new_err(format!("{kind} signal got {what}"))
}

impl Value {
    pub fn scalar(&self) -> PyResult<f64> {
        match self {
            Value::Scalar(v) => Ok(*v),
            _ => Err(type_error("scalar", "a non-scalar value")),
        }
    }

    pub fn quat(&self) -> PyResult<DQuat> {
        match self {
            Value::Quat(q) => Ok(*q),
            _ => Err(type_error("rotation", "a non-quaternion value")),
        }
    }

    fn tuple(&self) -> PyResult<&Arc<Vec<f64>>> {
        match self {
            Value::Tuple(t) => Ok(t),
            _ => Err(type_error("color", "a non-color value")),
        }
    }

    pub fn array(&self) -> PyResult<&Arc<Array>> {
        match self {
            Value::Array(a) => Ok(a),
            _ => Err(type_error("array", "a non-array value")),
        }
    }

    fn object(&self) -> PyResult<&Arc<Py<PyAny>>> {
        match self {
            Value::Object(o) => Ok(o),
            _ => Err(type_error("python", "a native value")),
        }
    }

    /// The same object (not just an equal value): Python's ``is``.
    pub fn same(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Array(a), Value::Array(b)) => Arc::ptr_eq(a, b),
            (Value::Tuple(a), Value::Tuple(b)) => Arc::ptr_eq(a, b),
            (Value::Object(a), Value::Object(b)) => Arc::ptr_eq(a, b) || a.as_ptr() == b.as_ptr(),
            _ => false,
        }
    }
}

// -- conversions --------------------------------------------------------------

fn float_array<'py>(object: &Bound<'py, PyAny>) -> PyResult<Array> {
    let numpy = object.py().import("numpy")?;
    let converted = numpy.call_method1("ascontiguousarray", (object, "float64"))?;
    let array = converted.cast::<PyArrayDyn<f64>>()?;
    let shape = array.shape().to_vec();
    let data = array.to_vec()?;
    Ok(Array { shape, data })
}

impl Kind {
    pub fn from_code(code: &str, kind: Py<PyAny>) -> Kind {
        match code {
            "scalar" => Kind::Scalar,
            "factor" => Kind::Factor,
            "rotation" => Kind::Rotation,
            "color" => Kind::Color,
            "colors" => Kind::Colors,
            "scalars" => Kind::Scalars,
            "path" => Kind::Path,
            "array" => Kind::Array,
            "discrete" => Kind::Discrete,
            _ => Kind::Python(kind),
        }
    }

    /// A value from Python, as the kind's operations take it (not
    /// validated: relative changes may have other shapes than values).
    pub fn value_from_py(&self, object: &Bound<'_, PyAny>) -> PyResult<Value> {
        Ok(match self {
            Kind::Scalar | Kind::Factor => Value::Scalar(object.extract::<f64>()?),
            Kind::Rotation => {
                // numpy-quaternion components are (w, x, y, z).
                let c: Vec<f64> = object.getattr("components")?.extract()?;
                Value::Quat(DQuat::from_xyzw(c[1], c[2], c[3], c[0]))
            }
            Kind::Color => Value::Tuple(Arc::new(object.extract::<Vec<f64>>()?)),
            Kind::Colors | Kind::Scalars | Kind::Path | Kind::Array => {
                Value::Array(Arc::new(float_array(object)?))
            }
            Kind::Discrete | Kind::Python(_) => Value::Object(Arc::new(object.clone().unbind())),
        })
    }

    /// The value as the Python kind holds it (read-only arrays, quaternion
    /// objects, tuples).
    pub fn value_to_py(&self, py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
        Ok(match value {
            Value::Scalar(v) => v.into_pyobject(py)?.into_any().unbind(),
            Value::Quat(q) => py
                .import("quaternion")?
                .getattr("quaternion")?
                .call1((q.w, q.x, q.y, q.z))?
                .unbind(),
            Value::Tuple(t) => PyTuple::new(py, t.iter())?.into_any().unbind(),
            Value::Array(a) => {
                let array = a.data.clone().into_pyarray(py).reshape(a.shape.clone())?;
                let flags = PyDict::new(py);
                flags.set_item("write", false)?;
                array.call_method("setflags", (), Some(&flags))?;
                array.into_any().unbind()
            }
            Value::Object(o) => o.clone_ref(py),
        })
    }

    // -- operations ---------------------------------------------------------

    /// ``validate``: what a signal stores.
    pub fn validate(&self, py: Python<'_>, value: Value) -> PyResult<Value> {
        Ok(match (self, value) {
            (Kind::Rotation, Value::Quat(q)) => {
                if q.length_squared() == 0.0 {
                    return Err(PyValueError::new_err(
                        "rotation quaternion must not be zero",
                    ));
                }
                Value::Quat(q.normalize())
            }
            (Kind::Colors, Value::Array(a)) if a.shape.len() == 1 => {
                Value::Array(Arc::new(Array {
                    shape: vec![1, a.data.len()],
                    data: a.data.clone(),
                }))
            }
            (Kind::Python(kind), value) => {
                let object = self.value_to_py(py, &value)?;
                Value::Object(Arc::new(
                    kind.bind(py).call_method1("validate", (object,))?.unbind(),
                ))
            }
            (_, value) => value,
        })
    }

    pub fn interpolate(
        &self,
        py: Python<'_>,
        start: &Value,
        end: &Value,
        alpha: f64,
    ) -> PyResult<Value> {
        Ok(match self {
            Kind::Scalar | Kind::Factor => {
                let (s, e) = (start.scalar()?, end.scalar()?);
                Value::Scalar(s + (e - s) * alpha)
            }
            Kind::Rotation => {
                let (s, mut e) = (start.quat()?, end.quat()?);
                if s.dot(e) < 0.0 {
                    e = -e; // q and -q are the same rotation: take the short way
                }
                Value::Quat(s.slerp(e, alpha))
            }
            Kind::Color => {
                let (s, e) = (start.tuple()?, end.tuple()?);
                if s.len() != e.len() {
                    return Err(PyValueError::new_err(
                        "cannot interpolate colors with different channel counts",
                    ));
                }
                Value::Tuple(Arc::new(
                    s.iter()
                        .zip(e.iter())
                        .map(|(a, b)| a + (b - a) * alpha)
                        .collect(),
                ))
            }
            Kind::Colors | Kind::Scalars => {
                let (s, e) = matched(start.array()?, end.array()?);
                Value::Array(Arc::new(s.zip(&e, |a, b| a + (b - a) * alpha)))
            }
            Kind::Path | Kind::Array => {
                let (s, e) = (start.array()?, end.array()?);
                if s.shape != e.shape {
                    return Err(PyValueError::new_err(format!(
                        "cannot interpolate paths of shapes {:?} and {:?}; align them first",
                        s.shape, e.shape
                    )));
                }
                Value::Array(Arc::new(s.zip(e, |a, b| a + (b - a) * alpha)))
            }
            Kind::Discrete => {
                if alpha >= 1.0 {
                    end.clone()
                } else {
                    start.clone()
                }
            }
            Kind::Python(kind) => {
                let (s, e) = (start.object()?.clone_ref(py), end.object()?.clone_ref(py));
                Value::Object(Arc::new(
                    kind.bind(py)
                        .call_method1("interpolate", (s, e, alpha))?
                        .unbind(),
                ))
            }
        })
    }

    pub fn combine(&self, py: Python<'_>, value: &Value, change: &Value) -> PyResult<Value> {
        Ok(match self {
            Kind::Scalar => Value::Scalar(value.scalar()? + change.scalar()?),
            Kind::Factor => Value::Scalar(value.scalar()? * change.scalar()?),
            Kind::Rotation => Value::Quat(change.quat()? * value.quat()?),
            Kind::Color => {
                let (v, c) = (value.tuple()?, change.tuple()?);
                Value::Tuple(Arc::new(
                    v.iter()
                        .zip(c.iter())
                        .map(|(a, b)| (a + b).clamp(0.0, 1.0))
                        .collect(),
                ))
            }
            Kind::Colors => {
                let change = colors_shape(change.array()?);
                let (v, c) = matched(value.array()?, &change);
                Value::Array(Arc::new(v.zip(&c, |a, b| (a + b).clamp(0.0, 1.0))))
            }
            Kind::Scalars => {
                let (v, c) = matched(value.array()?, change.array()?);
                Value::Array(Arc::new(v.zip(&c, |a, b| a + b)))
            }
            Kind::Path | Kind::Array => {
                Value::Array(Arc::new(broadcast_add(value.array()?, change.array()?)?))
            }
            Kind::Discrete => {
                return Err(PyTypeError::new_err(
                    "discrete signals do not support relative changes",
                ))
            }
            Kind::Python(kind) => {
                let (v, c) = (
                    value.object()?.clone_ref(py),
                    change.object()?.clone_ref(py),
                );
                Value::Object(Arc::new(
                    kind.bind(py).call_method1("combine", (v, c))?.unbind(),
                ))
            }
        })
    }

    pub fn scale(&self, py: Python<'_>, change: &Value, alpha: f64) -> PyResult<Value> {
        Ok(match self {
            Kind::Scalar => Value::Scalar(change.scalar()? * alpha),
            Kind::Factor => {
                let c = change.scalar()?;
                if c <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "scale factors must be positive, got {c}"
                    )));
                }
                Value::Scalar(c.powf(alpha))
            }
            Kind::Rotation => self.interpolate(py, &Value::Quat(DQuat::IDENTITY), change, alpha)?,
            Kind::Color => Value::Tuple(Arc::new(
                change.tuple()?.iter().map(|c| c * alpha).collect(),
            )),
            Kind::Colors | Kind::Scalars | Kind::Path | Kind::Array => {
                Value::Array(Arc::new(change.array()?.map(|c| c * alpha)))
            }
            Kind::Discrete => {
                return Err(PyTypeError::new_err(
                    "discrete signals do not support relative changes",
                ))
            }
            Kind::Python(kind) => {
                let c = change.object()?.clone_ref(py);
                Value::Object(Arc::new(
                    kind.bind(py).call_method1("scale", (c, alpha))?.unbind(),
                ))
            }
        })
    }

    pub fn equal(&self, py: Python<'_>, a: &Value, b: &Value) -> PyResult<bool> {
        Ok(match (a, b) {
            (Value::Scalar(x), Value::Scalar(y)) => x == y,
            (Value::Quat(x), Value::Quat(y)) => x == y,
            (Value::Tuple(x), Value::Tuple(y)) => x == y,
            (Value::Array(x), Value::Array(y)) => Arc::ptr_eq(x, y) || x == y,
            (Value::Object(x), Value::Object(y)) => {
                if x.as_ptr() == y.as_ptr() {
                    return Ok(true);
                }
                let (x, y) = (x.bind(py), y.bind(py));
                match self {
                    Kind::Python(kind) => {
                        kind.bind(py).call_method1("equal", (x, y))?.is_truthy()?
                    }
                    // DiscreteKind.equal: ==, or identity for values that
                    // cannot say (numpy arrays).
                    _ => x.eq(y).unwrap_or_default(),
                }
            }
            _ => false,
        })
    }
}

/// ``_spread``: ``values`` (K rows) spread evenly over ``count`` rows by
/// linear interpolation (numpy's ``interp`` over two ``linspace``s).
fn spread(values: &Array, count: usize) -> Array {
    let k = values.rows();
    if k == count {
        return Array {
            shape: values.shape.clone(),
            data: values.data.clone(),
        };
    }
    let cols = values.cols();
    let mut shape = values.shape.clone();
    shape[0] = count;
    if k == 1 {
        let data = (0..count)
            .flat_map(|_| values.data.iter().copied())
            .collect();
        return Array { shape, data };
    }
    let source = linspace(k);
    let target = linspace(count);
    let mut data = vec![0.0; count * cols];
    for (row, &x) in target.iter().enumerate() {
        for col in 0..cols {
            data[row * cols + col] = interp(x, &source, |j| values.data[j * cols + col]);
        }
    }
    Array { shape, data }
}

/// ``np.linspace(0, 1, n)``.
fn linspace(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![0.0];
    }
    let step = 1.0 / (n - 1) as f64;
    let mut values: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
    values[n - 1] = 1.0;
    values
}

/// ``np.interp(x, xp, fp)`` for increasing ``xp`` covering ``x``.
fn interp(x: f64, xp: &[f64], fp: impl Fn(usize) -> f64) -> f64 {
    let last = xp.len() - 1;
    if x >= xp[last] {
        return fp(last);
    }
    if x <= xp[0] {
        return fp(0);
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if x == xp[j] {
        return fp(j);
    }
    let slope = (fp(j + 1) - fp(j)) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp(j)
}

/// ``_matched``: both lists spread to the longer length.
fn matched(a: &Array, b: &Array) -> (Array, Array) {
    let count = a.rows().max(b.rows());
    (spread(a, count), spread(b, count))
}

/// A COLORS change as ``ColorsKind.validate`` shapes it: one color is one row.
fn colors_shape(change: &Array) -> Array {
    if change.shape.len() == 1 {
        Array {
            shape: vec![1, change.data.len()],
            data: change.data.clone(),
        }
    } else {
        Array {
            shape: change.shape.clone(),
            data: change.data.clone(),
        }
    }
}

/// ``value + change`` with numpy broadcasting of a scalar, a row or a
/// whole array.
fn broadcast_add(value: &Array, change: &Array) -> PyResult<Array> {
    if change.data.len() == 1 {
        let c = change.data[0];
        return Ok(value.map(|v| v + c));
    }
    if change.shape == value.shape {
        return Ok(value.zip(change, |a, b| a + b));
    }
    let cols = value.cols();
    if change.shape.len() == 1 && change.data.len() == cols {
        let data = value
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| v + change.data[i % cols])
            .collect();
        return Ok(Array {
            shape: value.shape.clone(),
            data,
        });
    }
    Err(PyValueError::new_err(format!(
        "operands could not be broadcast together with shapes {:?} {:?}",
        value.shape, change.shape
    )))
}
