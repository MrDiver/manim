//! Growable record arrays that mirror GPU buffers: records are written in
//! place, and the ranges written since the last upload are remembered, so
//! only those are uploaded again.

use bytemuck::Pod;

pub struct Arena {
    /// Bytes per record (a multiple of 4, like every WGSL struct).
    pub record_size: usize,
    /// The records, kept in 4-byte words so every record struct of
    /// records.rs can be read and written in place (they hold only 4-byte
    /// fields).
    data: Vec<u32>,
    /// Records in use (live or garbage).
    pub used: usize,
    /// Records no longer referenced.
    pub garbage: usize,
    dirty: Vec<(usize, usize)>,
    reallocated: bool,
}

impl Arena {
    pub fn new(record_size: usize) -> Arena {
        assert!(
            record_size.is_multiple_of(4),
            "records are made of 4-byte fields"
        );
        Arena {
            record_size,
            data: vec![0; 256 * record_size / 4],
            used: 0,
            garbage: 0,
            dirty: Vec::new(),
            reallocated: true,
        }
    }

    fn capacity(&self) -> usize {
        4 * self.data.len() / self.record_size
    }

    /// Reserve `count` records at the end; returns the first index.
    pub fn allocate(&mut self, count: usize) -> usize {
        let start = self.used;
        if start + count > self.capacity() {
            let capacity = (2 * self.capacity()).max(start + count);
            self.data.resize(capacity * self.record_size / 4, 0);
            self.reallocated = true;
        }
        self.used += count;
        start
    }

    /// The bytes of records `start..stop`, to write into (marked dirty).
    pub fn records_mut(&mut self, start: usize, stop: usize) -> &mut [u8] {
        self.dirty.push((start, stop));
        let (first, last) = (start * self.record_size, stop * self.record_size);
        &mut bytemuck::cast_slice_mut(&mut self.data)[first..last]
    }

    /// Record `index` as a `T` (a record struct of records.rs), to change
    /// in place (marked dirty).
    pub fn record_mut<T: Pod>(&mut self, index: usize) -> &mut T {
        bytemuck::from_bytes_mut(self.records_mut(index, index + 1))
    }

    pub fn write(&mut self, start: usize, records: &[u8]) {
        let stop = start + records.len() / self.record_size;
        self.records_mut(start, stop).copy_from_slice(records);
    }

    /// Write `records` over the old range if they fit exactly, else into a
    /// new range (the old one becomes garbage); returns the first index.
    pub fn place(&mut self, records: &[u8], old_first: usize, old_count: usize) -> usize {
        let count = records.len() / self.record_size;
        if old_count == count && old_count > 0 {
            self.write(old_first, records);
            return old_first;
        }
        self.garbage += old_count;
        let first = self.allocate(count);
        self.write(first, records);
        first
    }

    /// The bytes of the records in use (for an upload).
    pub fn bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.data)[..self.used * self.record_size]
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
