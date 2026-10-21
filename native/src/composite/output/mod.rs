//! Output: the composited frame finished for where it goes and read back.
//!
//! The frame (linear light, premultiplied) is turned into RGBA (sRGB
//! values, straight alpha) or NV12, the Y'CbCr 4:2:0 planes a video
//! encoder takes (`PixelFormat`; converted on the GPU, it reads back 1.5
//! bytes per pixel instead of 4), copied into a staging buffer and
//! mapped while the next frames are prepared. Up to `in_flight` frames
//! are on their way; they come back in submission order.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gpu::device::{Gpu, COLOR_FORMAT};
use crate::gpu::fullscreen::{Blend, Fullscreen, TexturePool};

/// Bytes per row of texture-to-buffer copies must be multiples of this.
const ROW_ALIGNMENT: u32 = 256;
/// The planes of NV12: luma per pixel, (Cb, Cr) per 2x2 pixels.
const LUMA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const CHROMA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

/// How the frames are read back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    /// Rows of RGBA bytes: sRGB values, straight alpha.
    Rgba,
    /// NV12: rows of BT.709 limited-range luma, one byte per pixel, then
    /// half as many rows of (Cb, Cr) byte pairs, one pair per 2x2 pixels.
    /// Width and height must be even.
    Nv12,
}

impl PixelFormat {
    /// Bytes per row and rows of a frame `size` pixels large.
    pub fn layout(self, size: (u32, u32)) -> (u32, u32) {
        match self {
            PixelFormat::Rgba => (4 * size.0, size.1),
            PixelFormat::Nv12 => (size.0, size.1 + size.1 / 2),
        }
    }
}

/// A frame whose copy into `staging` is on its way.
struct InFlight {
    index: u64,
    staging: wgpu::Buffer,
    /// The submission that copies the frame into `staging`.
    submission: wgpu::SubmissionIndex,
    mapped: Arc<AtomicBool>,
}

pub struct Output {
    pixel_format: PixelFormat,
    size: (u32, u32),
    /// Bytes per row in the staging buffers (aligned for copies).
    row_bytes: u32,
    free: Vec<wgpu::Buffer>,
    pending: VecDeque<InFlight>,
    /// Frames read back but not yet taken, `(index, pixels)`.
    ready: VecDeque<(u64, Vec<u8>)>,
    fullscreen: Fullscreen,
    textures: TexturePool,
}

impl Output {
    pub fn new(
        gpu: &Gpu,
        size: (u32, u32),
        pixel_format: PixelFormat,
        in_flight: usize,
    ) -> Result<Output> {
        let (width, height) = size;
        if pixel_format == PixelFormat::Nv12 && (width % 2 == 1 || height % 2 == 1) {
            return Err(Error::Invalid(format!(
                "NV12 needs an even width and height, not {width}x{height}"
            )));
        }
        let (row_length, rows) = pixel_format.layout(size);
        let row_bytes = row_length.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        let free = (0..in_flight)
            .map(|index| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("readback {index}")),
                    size: (row_bytes * rows) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(Output {
            pixel_format,
            size,
            row_bytes,
            free,
            pending: VecDeque::new(),
            ready: VecDeque::new(),
            fullscreen: Fullscreen::new(&gpu.device),
            textures: TexturePool::default(),
        })
    }

    /// Finish frame `index` from `frame` (linear, premultiplied): convert
    /// it to the pixel format, copy it into a staging buffer, submit
    /// `encoder` and map the buffer. Waits for the oldest frame when every
    /// staging buffer is in use.
    pub fn send(
        &mut self,
        gpu: &Gpu,
        mut encoder: wgpu::CommandEncoder,
        frame: &wgpu::Texture,
        index: u64,
    ) -> Result<()> {
        if self.free.is_empty() {
            self.finish_oldest(gpu)?;
        }
        let staging = self.free.pop().unwrap();
        let frame_view = frame.create_view(&Default::default());
        let resources = [wgpu::BindingResource::TextureView(&frame_view)];
        let (width, height) = self.size;
        let planes: Vec<(&'static str, &'static str, (u32, u32), wgpu::TextureFormat)> =
            match self.pixel_format {
                PixelFormat::Rgba => vec![("output", "fs_finalize", self.size, COLOR_FORMAT)],
                PixelFormat::Nv12 => vec![
                    ("luma", "fs_luma", self.size, LUMA_FORMAT),
                    (
                        "chroma",
                        "fs_chroma",
                        (width / 2, height / 2),
                        CHROMA_FORMAT,
                    ),
                ],
            };
        // The planes one after the other, each row `row_bytes` long.
        let mut offset = 0;
        for (name, entry_point, size, format) in planes {
            let plane = self.textures.get(
                &gpu.device,
                name,
                size,
                format,
                1,
                wgpu::TextureUsages::COPY_SRC,
            );
            self.fullscreen.run(
                gpu,
                &mut encoder,
                "finalize.wgsl",
                entry_point,
                &plane.create_view(&Default::default()),
                format,
                &resources,
                Blend::Replace,
                true,
            )?;
            encoder.copy_texture_to_buffer(
                plane.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset,
                        bytes_per_row: Some(self.row_bytes),
                        rows_per_image: Some(plane.height()),
                    },
                },
                plane.size(),
            );
            offset += (self.row_bytes * plane.height()) as u64;
        }
        let submission = gpu.queue.submit([encoder.finish()]);
        let mapped = Arc::new(AtomicBool::new(false));
        let signal = mapped.clone();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    signal.store(true, Ordering::Release);
                }
            });
        self.pending.push_back(InFlight {
            index,
            staging,
            submission,
            mapped,
        });
        Ok(())
    }

    /// Frames finished so far, `(index, pixels)` in submission order;
    /// with `wait`, waits for every frame on its way.
    pub fn finished(&mut self, gpu: &Gpu, wait: bool) -> Result<Vec<(u64, Vec<u8>)>> {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        while let Some(front) = self.pending.front() {
            if !wait && !front.mapped.load(Ordering::Acquire) {
                break;
            }
            self.finish_oldest(gpu)?;
        }
        Ok(self.ready.drain(..).collect())
    }

    /// Read the oldest frame on its way back into `ready`, without the
    /// row padding (waits for it).
    fn finish_oldest(&mut self, gpu: &Gpu) -> Result<()> {
        let frame = self
            .pending
            .pop_front()
            .ok_or_else(|| Error::Invalid("no frame in flight".into()))?;
        // Wait for this frame's copy only; later frames keep running.
        while !frame.mapped.load(Ordering::Acquire) {
            let wait = wgpu::PollType::Wait {
                submission_index: Some(frame.submission.clone()),
                timeout: None,
            };
            gpu.device
                .poll(wait)
                .map_err(|error| Error::Gpu(error.to_string()))?;
        }
        let (row_length, rows) = self.pixel_format.layout(self.size);
        let (row_length, rows) = (row_length as usize, rows as usize);
        let row_bytes = self.row_bytes as usize;
        let image = {
            let data = frame
                .staging
                .slice(..)
                .get_mapped_range()
                .map_err(|error| Error::Gpu(error.to_string()))?;
            // Zeroed first: the new pages are faulted in by a plain
            // memset, not one by one in the middle of the slow reads from
            // mapped memory (measured: 1.6 against 1.1 ms per 1080p frame).
            let mut image = vec![0u8; row_length * rows];
            for (row, target) in data
                .chunks_exact(row_bytes)
                .zip(image.chunks_exact_mut(row_length))
            {
                target.copy_from_slice(&row[..row_length]);
            }
            image
        };
        frame.staging.unmap();
        self.free.push(frame.staging);
        self.ready.push_back((frame.index, image));
        Ok(())
    }
}
