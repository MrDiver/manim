//! A GPU buffer reused across frames that grows when needed.

const MIN_SIZE: u64 = 256;

/// Records to upload: their bytes, the bytes of one record, and the record
/// ranges written since the last upload (None: upload all of them).
pub struct Records<'a> {
    pub bytes: &'a [u8],
    pub record_size: usize,
    pub written: Option<Vec<(usize, usize)>>,
}

pub struct GrowableBuffer {
    pub buffer: wgpu::Buffer,
    label: &'static str,
    usage: wgpu::BufferUsages,
    /// The size it never grows past by doubling (what one binding may
    /// see); data larger than it still gets a buffer of its size.
    limit: u64,
}

impl GrowableBuffer {
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
    ) -> GrowableBuffer {
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        GrowableBuffer {
            buffer: allocate(device, label, usage, MIN_SIZE),
            label,
            usage,
            limit: u64::MAX,
        }
    }

    /// A buffer that grows by doubling up to `limit` bytes at most.
    pub fn with_limit(device: &wgpu::Device, label: &'static str, usage: wgpu::BufferUsages, limit: u64) -> GrowableBuffer {
        GrowableBuffer {
            limit,
            ..GrowableBuffer::new(device, label, usage)
        }
    }

    /// Write `data` to the start of the buffer; returns whether it had to
    /// grow (bind groups that reference it must then be made again).
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        let size = MIN_SIZE.max(data.len() as u64);
        let grown = size > self.buffer.size();
        if grown {
            // Grow geometrically, so a slowly growing scene reallocates
            // rarely, but not past the limit.
            self.buffer = allocate(
                device,
                self.label,
                self.usage,
                (2 * self.buffer.size()).min(self.limit).max(size),
            );
        }
        if !data.is_empty() {
            queue.write_buffer(&self.buffer, 0, data);
        }
        grown
    }

    /// Upload only the record ranges `[start, stop)` of `data`, whose other
    /// records the buffer holds already; everything when `ranges` is None
    /// or the buffer is too small. Returns whether it grew.
    pub fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        record_size: usize,
        ranges: Option<&[(usize, usize)]>,
    ) -> bool {
        let Some(ranges) = ranges else {
            return self.upload(device, queue, data);
        };
        if data.len() as u64 > self.buffer.size() {
            return self.upload(device, queue, data);
        }
        for &(start, stop) in ranges {
            if stop > start {
                queue.write_buffer(
                    &self.buffer,
                    (start * record_size) as u64,
                    &data[start * record_size..stop * record_size],
                );
            }
        }
        false
    }

    /// `write` for `Records`.
    pub fn write_records(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &Records,
    ) -> bool {
        self.write(
            device,
            queue,
            records.bytes,
            records.record_size,
            records.written.as_deref(),
        )
    }

    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }
}

fn allocate(
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    size: u64,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.div_ceil(4) * 4, // buffer sizes must be multiples of 4
        usage,
        mapped_at_creation: false,
    })
}
