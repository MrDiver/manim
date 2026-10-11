//! Optional GPU timing per render pass with timestamp queries: passes
//! with the same name are summed; reading waits for the GPU, so it is meant for
//! benchmarks, not for every frame.

use std::collections::HashMap;

/// Timestamps, i.e. 256 passes per read.
const CAPACITY: u32 = 512;

pub struct GpuProfiler {
    queries: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
    names: Vec<&'static str>,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, enabled: bool) -> GpuProfiler {
        let queries = enabled.then(|| {
            let set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profiler"),
                ty: wgpu::QueryType::Timestamp,
                count: CAPACITY,
            });
            let size = CAPACITY as u64 * 8;
            let buffer = |usage| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("profiler"),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let resolve = buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
            let readback = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            (set, resolve, readback)
        });
        GpuProfiler {
            queries,
            names: Vec::new(),
        }
    }

    /// `timestamp_writes` for a render pass, or None.
    pub fn pass_timing(
        &mut self,
        name: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (set, _, _) = self.queries.as_ref()?;
        if 2 * self.names.len() as u32 + 2 > CAPACITY {
            return None;
        }
        let index = 2 * self.names.len() as u32;
        self.names.push(name);
        Some(wgpu::RenderPassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        })
    }

    /// Milliseconds per pass name since the last read (waits for the GPU).
    pub fn read(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> HashMap<String, f64> {
        let mut timings = HashMap::new();
        let (Some((set, resolve, readback)), false) =
            (self.queries.as_ref(), self.names.is_empty())
        else {
            self.names.clear();
            return timings;
        };
        let count = 2 * self.names.len() as u32;
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(set, 0..count, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, count as u64 * 8);
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..count as u64 * 8);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let period = queue.get_timestamp_period() as f64;
        if let Ok(data) = slice.get_mapped_range() {
            let stamps: &[u64] = bytemuck::cast_slice(&data);
            for (index, name) in self.names.iter().enumerate() {
                let ticks = stamps[2 * index + 1].saturating_sub(stamps[2 * index]) as f64;
                *timings.entry(name.to_string()).or_insert(0.0) += ticks * period / 1e6;
                // ns -> ms
            }
        }
        readback.unmap();
        self.names.clear();
        timings
    }
}
