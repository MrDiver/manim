//! Animations called back in Python: those whose contributions have no
//! data form (`Animation.contributions(alpha, start_values)`). Each is
//! called once per evaluation, for all its targets at once; the entries
//! of its targets on their tracks read their own value from the result.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// One animation called back, with the start value of each of its
/// targets, in `targets()` order.
struct Callback {
    animation: Py<PyAny>,
    start_values: Py<PyTuple>,
}

/// Every callback of a scene, and what each returned in the current
/// evaluation.
#[derive(Default)]
pub struct Callbacks {
    callbacks: Vec<Callback>,
    /// Per callback, its contributions at the time evaluated now (None:
    /// not called yet in this evaluation).
    results: Vec<Option<Py<PyAny>>>,
}

impl Callbacks {
    /// Add an animation; returns its index.
    pub fn add(&mut self, animation: Py<PyAny>, start_values: Py<PyTuple>) -> usize {
        self.callbacks.push(Callback {
            animation,
            start_values,
        });
        self.results.push(None);
        self.callbacks.len() - 1
    }

    /// Forget the results: a new evaluation begins.
    pub fn clear(&mut self) {
        self.results.iter_mut().for_each(|result| *result = None);
    }

    /// Target `target`'s contribution of callback `index` at progress
    /// `alpha`, calling the animation if it was not called in this
    /// evaluation yet.
    pub fn contribution<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
        target: usize,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.results[index].is_none() {
            let callback = &self.callbacks[index];
            let values = callback
                .animation
                .bind(py)
                .call_method1("contributions", (alpha, callback.start_values.bind(py)))?;
            self.results[index] = Some(values.unbind());
        }
        self.results[index]
            .as_ref()
            .unwrap()
            .bind(py)
            .get_item(target)
    }
}
