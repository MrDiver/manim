//! Growable record arrays that mirror GPU buffers. Ports `Arena`
//! (manimx/renderer/webgpu/arena.py).
//!
//! The records live in a numpy byte array, so Python hands them to the
//! renderer as a view (`array[:used].view(DTYPE)`) without copying.

use bytemuck::Pod;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

pub struct Arena {
    /// Bytes per record.
    pub record_size: usize,
    data: Py<PyArray1<u8>>,
    capacity: usize,
    /// Records in use (live or garbage).
    pub used: usize,
    /// Records no longer referenced.
    pub garbage: usize,
    dirty: Vec<(usize, usize)>,
    reallocated: bool,
}

impl Arena {
    pub fn new(py: Python<'_>, record_size: usize) -> Arena {
        let capacity = 256;
        Arena {
            record_size,
            data: PyArray1::zeros(py, capacity * record_size, false).unbind(),
            capacity,
            used: 0,
            garbage: 0,
            dirty: Vec::new(),
            reallocated: true,
        }
    }

    /// Reserve `count` records at the end; returns the first index.
    pub fn allocate(&mut self, py: Python<'_>, count: usize) -> usize {
        let start = self.used;
        if start + count > self.capacity {
            let capacity = (2 * self.capacity).max(start + count);
            let grown = PyArray1::<u8>::zeros(py, capacity * self.record_size, false);
            // SAFETY: both arrays are owned here and nothing else writes them.
            unsafe {
                let old = self.data.bind(py).as_slice().expect("contiguous");
                grown.as_slice_mut().expect("contiguous")[..start * self.record_size]
                    .copy_from_slice(&old[..start * self.record_size]);
            }
            self.data = grown.unbind();
            self.capacity = capacity;
            self.reallocated = true;
        }
        self.used += count;
        start
    }

    /// The bytes of records `start..stop`, to write into (marked dirty).
    pub fn records_mut(&mut self, py: Python<'_>, start: usize, stop: usize) -> &mut [u8] {
        self.dirty.push((start, stop));
        let (first, last) = (start * self.record_size, stop * self.record_size);
        // SAFETY: the array is owned by this arena, so the slice stays valid
        // while the arena is borrowed (growing needs `&mut self`), and
        // Python only reads the array between packs.
        unsafe {
            let all = self.data.bind(py).as_slice_mut().expect("contiguous");
            std::slice::from_raw_parts_mut(all.as_mut_ptr().add(first), last - first)
        }
    }

    /// Record `index` as a `T` (a record struct of records.rs), to change
    /// in place (marked dirty).
    pub fn record_mut<T: Pod>(&mut self, py: Python<'_>, index: usize) -> &mut T {
        bytemuck::from_bytes_mut(self.records_mut(py, index, index + 1))
    }

    pub fn write(&mut self, py: Python<'_>, start: usize, records: &[u8]) {
        let stop = start + records.len() / self.record_size;
        self.records_mut(py, start, stop).copy_from_slice(records);
    }

    /// Write `records` over the old range if they fit exactly, else into a
    /// new range (the old one becomes garbage); returns the first index.
    pub fn place(
        &mut self,
        py: Python<'_>,
        records: &[u8],
        old_first: usize,
        old_count: usize,
    ) -> usize {
        let count = records.len() / self.record_size;
        if old_count == count && old_count > 0 {
            self.write(py, old_first, records);
            return old_first;
        }
        self.garbage += old_count;
        let first = self.allocate(py, count);
        self.write(py, first, records);
        first
    }

    /// The records in use, as a numpy byte array view.
    pub fn view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let end = self.used * self.record_size;
        self.data.bind(py).call_method1(
            "__getitem__",
            (pyo3::types::PySlice::new(py, 0, end as isize, 1),),
        )
    }

    /// Merged dirty ranges since the last call, or None for "upload
    /// everything" (after a reallocation).
    pub fn ranges(&mut self) -> Option<Vec<(usize, usize)>> {
        if self.reallocated {
            self.reallocated = false;
            self.dirty.clear();
            return None;
        }
        self.dirty.sort();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for &(start, stop) in &self.dirty {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(stop),
                _ => merged.push((start, stop)),
            }
        }
        self.dirty.clear();
        Some(merged)
    }

    pub fn too_much_garbage(&self, threshold: usize) -> bool {
        self.garbage > threshold.max(self.used / 2)
    }
}
