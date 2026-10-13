//! `FrameRenderer`: the whole render chain for frames of render object
//! trees. Python drives it through python/frame_renderer.rs.
//!
//! A frame is a list of sub-frames (time weight and views; motion blur
//! averages several), each view a render object tree and its
//! `ViewSettings`. The renderer packs, renders, composites and finalizes
//! them, then copies the output into a staging buffer that is mapped
//! while the next frames are prepared. Up to `in_flight` frames are on
//! the GPU at once; finished frames come back in submission order.
//!
//! The output is RGBA (sRGB values, straight alpha) or NV12, the Y'CbCr
//! 4:2:0 planes a video encoder takes (`PixelFormat`): converted on the
//! GPU, it reads back 1.5 bytes per pixel instead of 4.
//!
//! A frame that shows exactly what the frame before showed is not
//! rendered: equal trees (`RNode::fingerprint`) seen with equal settings
//! give equal pictures.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::render::composer::{Composer, CompositeTarget};
use crate::render::context::{Gpu, CHROMA_FORMAT, COLOR_FORMAT, HDR_FORMAT, LUMA_FORMAT};
use crate::render::effects::depth_of_field::DepthOfField;
use crate::render::effects::Effect;
use crate::render::error::{Error, Result};
use crate::render::fullscreen::Blend;
use crate::render::inspect::TextureKind;
use crate::render::packing::packer::{Packer, Packet};
use crate::render::packing::records::Frame;
use crate::render::packing::vectors::VectorDraws;
use crate::render::profiler::StageClock;
use crate::render::renderer::{Renderer, TargetSettings};
use crate::render::settings::ViewSettings;
use crate::robjects::RNode;

/// Bytes per row of texture-to-buffer copies must be multiples of this.
const ROW_ALIGNMENT: u32 = 256;

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

/// One scene seen from one camera.
pub struct View {
    pub root: Arc<RNode>,
    pub settings: ViewSettings,
}

/// Views drawn over each other (lowest first), weighted into the frame.
pub struct SubFrame {
    pub weight: f32,
    pub views: Vec<View>,
}

/// What stays the same for every frame of a renderer.
pub struct RendererOptions {
    pub width: u32,
    pub height: u32,
    /// MSAA samples per pixel.
    pub samples: u32,
    pub shadow_resolution: u32,
    /// Straight RGBA, sRGB.
    pub background: [f32; 4],
    /// The `ToneMapOperator` value.
    pub tone_map: u32,
    /// The image effects, in chain order.
    pub effects: Vec<Effect>,
    pub in_flight: usize,
    /// Measure GPU time per pass.
    pub profile: bool,
    pub pixel_format: PixelFormat,
}

/// A frame whose copy into `staging` is on its way.
struct InFlight {
    index: u64,
    staging: wgpu::Buffer,
    /// The submission that copies the frame into `staging`.
    submission: wgpu::SubmissionIndex,
    mapped: Arc<AtomicBool>,
}

pub struct FrameRenderer {
    gpu: Arc<Gpu>,
    packer: Packer,
    renderer: Renderer,
    composer: Composer,
    settings: TargetSettings,
    background: [f32; 4],
    pixel_format: PixelFormat,
    /// Bytes per row in the staging buffers (aligned for copies).
    row_bytes: u32,
    free: Vec<wgpu::Buffer>,
    pending: VecDeque<InFlight>,
    /// Frames read back but not yet taken, `(index, pixels)`.
    ready: VecDeque<(u64, Vec<u8>)>,
    /// The key of the last rendered frame (None: never held).
    last_key: Option<Vec<u64>>,
    clock: StageClock,
}

impl FrameRenderer {
    pub fn new(options: RendererOptions) -> Result<FrameRenderer> {
        if options.in_flight < 1 {
            return Err(Error::Invalid(
                "a pipeline needs at least one frame in flight".into(),
            ));
        }
        let gpu = Gpu::shared()?;
        let (width, height) = (options.width, options.height);
        if options.pixel_format == PixelFormat::Nv12 && (width % 2 == 1 || height % 2 == 1) {
            return Err(Error::Invalid(format!(
                "NV12 needs an even width and height, not {width}x{height}"
            )));
        }
        let (row_length, rows) = options.pixel_format.layout((width, height));
        let row_bytes = row_length.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        let free = (0..options.in_flight)
            .map(|index| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("readback {index}")),
                    size: (row_bytes * rows) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(FrameRenderer {
            packer: Packer::new(),
            renderer: Renderer::new(&gpu, options.profile)?,
            composer: Composer::new(&gpu, options.effects, options.tone_map),
            gpu,
            settings: TargetSettings {
                width,
                height,
                samples: options.samples,
                shadow_resolution: options.shadow_resolution,
            },
            background: options.background,
            pixel_format: options.pixel_format,
            row_bytes,
            free,
            pending: VecDeque::new(),
            ready: VecDeque::new(),
            last_key: None,
            clock: StageClock::default(),
        })
    }

    /// Render frame `index`. Returns true when the frame shows exactly
    /// what the frame before showed: then nothing is rendered or read
    /// back for it.
    pub fn render(&mut self, index: u64, sub_frames: &[SubFrame]) -> Result<bool> {
        let start = Instant::now();
        let held = self.render_frame(index, sub_frames);
        self.clock.add("render (all)", start);
        held
    }

    fn render_frame(&mut self, index: u64, sub_frames: &[SubFrame]) -> Result<bool> {
        let key = self.frame_key(sub_frames);
        if key.is_some() && key == self.last_key {
            return Ok(true);
        }
        self.last_key = key;
        let size = (self.settings.width, self.settings.height);
        let device = self.gpu.device.clone();
        let accumulator = self.composer.textures.get(
            &device,
            "accumulator",
            size,
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::empty(),
        );
        let accumulator_view = accumulator.create_view(&Default::default());
        for (position, sub_frame) in sub_frames.iter().enumerate() {
            let first = position == 0;
            let weight = sub_frame.weight;
            if sub_frame.views.len() <= 1 {
                // One submission per sub-frame: the renderer's passes and
                // the composite share an encoder. Without a view, the
                // background alone.
                let mut encoder = device.create_command_encoder(&Default::default());
                let target = CompositeTarget {
                    view: &accumulator_view,
                    weight,
                    clear: first,
                    background: true,
                    blend: Blend::Additive,
                };
                match sub_frame.views.first() {
                    Some(view) => self.render_view(&mut encoder, view, target)?,
                    None => self.render_empty(&mut encoder, target)?,
                }
                self.submit(encoder);
                continue;
            }
            // Several views: each is rendered and submitted on its own (the
            // renderer's buffers hold one view at a time) and put over the
            // stack, which is then added to the accumulator.
            let stack = self.composer.textures.get(
                &device,
                "view stack",
                size,
                HDR_FORMAT,
                1,
                wgpu::TextureUsages::empty(),
            );
            let stack_view = stack.create_view(&Default::default());
            for (number, view) in sub_frame.views.iter().enumerate() {
                let mut encoder = device.create_command_encoder(&Default::default());
                let target = CompositeTarget {
                    view: &stack_view,
                    weight: 1.0,
                    clear: number == 0,
                    background: number == 0,
                    blend: Blend::Over,
                };
                self.render_view(&mut encoder, view, target)?;
                self.submit(encoder);
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            self.composer.accumulate(
                &self.gpu,
                &mut encoder,
                &stack,
                &accumulator_view,
                weight,
                first,
                &mut self.renderer.profiler,
            )?;
            self.submit(encoder);
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        let frame_uniform = self.renderer.frame_uniform().clone();
        let frame =
            self.composer
                .frame_effects(&self.gpu, &mut encoder, &accumulator, &frame_uniform)?;
        if self.free.is_empty() {
            self.finish_oldest()?;
        }
        let staging = self.free.pop().unwrap();
        let planes = match self.pixel_format {
            PixelFormat::Rgba => {
                let output = self.texture("output", size, COLOR_FORMAT);
                self.composer.finalize(
                    &self.gpu,
                    &mut encoder,
                    &frame,
                    &output.create_view(&Default::default()),
                    &mut self.renderer.profiler,
                )?;
                vec![output]
            }
            PixelFormat::Nv12 => {
                let luma = self.texture("luma", size, LUMA_FORMAT);
                let chroma = self.texture("chroma", (size.0 / 2, size.1 / 2), CHROMA_FORMAT);
                self.composer.finalize_nv12(
                    &self.gpu,
                    &mut encoder,
                    &frame,
                    &luma.create_view(&Default::default()),
                    &chroma.create_view(&Default::default()),
                    &mut self.renderer.profiler,
                )?;
                vec![luma, chroma]
            }
        };
        // The planes one after the other, each row `row_bytes` long.
        let mut offset = 0;
        for plane in planes {
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
        let start = Instant::now();
        let submission = self.gpu.queue.submit([encoder.finish()]);
        self.clock.add("submit", start);
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
        Ok(false)
    }

    /// Frames finished so far, `(index, pixels)` in submission order;
    /// with `wait`, waits for every frame on the GPU.
    pub fn finished(&mut self, wait: bool) -> Result<Vec<(u64, Vec<u8>)>> {
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        while let Some(front) = self.pending.front() {
            if !wait && !front.mapped.load(Ordering::Acquire) {
                break;
            }
            self.finish_oldest()?;
        }
        Ok(self.ready.drain(..).collect())
    }

    /// A texture of the last frame as RGBA floats, to look at how it was
    /// made (see inspect.rs): `(width, height, texels)`.
    pub fn inspect(&mut self, name: &str, layer: u32) -> Result<(u32, u32, Vec<f32>)> {
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let (view, kind, size) = match self.renderer.inspected(name) {
            Some(found) => found,
            None => {
                let texture = self
                    .composer
                    .textures
                    .find(name)
                    .ok_or_else(|| Error::Invalid(format!("no texture named {name:?}")))?;
                let size = (texture.width(), texture.height());
                (
                    texture.create_view(&Default::default()),
                    TextureKind::Float,
                    size,
                )
            }
        };
        let texels = crate::render::inspect::read(&self.gpu, &view, kind, size, layer)?;
        Ok((size.0, size.1, texels))
    }

    /// How many shadow map layers there are.
    pub fn shadow_layers(&self) -> u32 {
        self.renderer.shadow_layers()
    }

    /// GPU milliseconds per pass name since the last call (with `profile`).
    pub fn gpu_timings(&mut self) -> Vec<(String, f64)> {
        self.renderer
            .profiler
            .read(&self.gpu.device, &self.gpu.queue)
            .into_iter()
            .map(|(name, ms)| (name.to_string(), ms))
            .collect()
    }

    /// Equal keys promise equal frames: per sub-frame its weight and per
    /// view the tree's and the settings' fingerprints. None with a shader
    /// effect or a surface shader in a view, which may read the time.
    fn frame_key(&self, sub_frames: &[SubFrame]) -> Option<Vec<u64>> {
        if self.composer.has_shader_effects() {
            return None;
        }
        let views = || sub_frames.iter().flat_map(|sub_frame| &sub_frame.views);
        if views().any(|view| view.root.uses_time()) {
            return None;
        }
        let mut key = Vec::new();
        for sub_frame in sub_frames {
            key.extend([
                sub_frame.weight.to_bits() as u64,
                sub_frame.views.len() as u64,
            ]);
            for view in &sub_frame.views {
                view.root.fingerprint(&mut key);
                view.settings.fingerprint(&mut key);
            }
        }
        Some(key)
    }

    /// Render an empty scene and composite it into `target`.
    fn render_empty(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: CompositeTarget,
    ) -> Result<()> {
        let packet = Packet {
            draws: VectorDraws::default(),
            scope_parents: Vec::new(),
            meshes: None,
        };
        let frame: Frame = bytemuck::Zeroable::zeroed();
        let targets = self.renderer.render_layers(
            &self.gpu,
            &mut self.packer,
            &packet,
            &frame,
            self.settings,
            encoder,
        )?;
        let (lit, vector) = (targets.lit.clone(), targets.vector.clone());
        self.composer.composite(
            &self.gpu,
            encoder,
            &lit,
            &vector,
            target,
            self.background,
            &mut self.renderer.profiler,
            None,
        )
    }

    /// Pack and render one view and composite it into `target`. The
    /// pictures of its camera views are rendered and submitted first.
    fn render_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &View,
        target: CompositeTarget,
    ) -> Result<()> {
        let camera_views = &view.settings.camera_views;
        if !camera_views.is_empty() {
            let size = (self.settings.width, self.settings.height);
            self.renderer
                .ensure_camera_views(&self.gpu, camera_views.len(), size);
            for (layer, camera_view) in camera_views.iter().enumerate() {
                // Each is submitted on its own: the renderer's buffers hold
                // one render at a time.
                let mut picture_encoder =
                    self.gpu.device.create_command_encoder(&Default::default());
                let layer_view = self.renderer.camera_view_layer(layer);
                let picture = CompositeTarget {
                    view: &layer_view,
                    weight: 1.0,
                    clear: true,
                    background: true,
                    blend: Blend::Replace,
                };
                let settings = view.settings.picture(camera_view);
                self.render_layers_into(&mut picture_encoder, &view.root, &settings, picture)?;
                self.submit(picture_encoder);
            }
        }
        self.render_layers_into(encoder, &view.root, &view.settings, target)
    }

    /// Pack and render the tree under `root` seen with `settings`, and
    /// composite the result into `target`.
    fn render_layers_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        root: &RNode,
        settings: &ViewSettings,
        target: CompositeTarget,
    ) -> Result<()> {
        let size = (self.settings.width, self.settings.height);
        let start = Instant::now();
        let packet = self.packer.pack_frame(root, &settings.pack_settings())?;
        self.clock.add("pack", start);
        let light_count = packet
            .meshes
            .as_ref()
            .map_or(0, |meshes| meshes.lights.len() as u32);
        let frame = settings.camera_frame(size, light_count);
        let start = Instant::now();
        let targets = self.renderer.render_layers(
            &self.gpu,
            &mut self.packer,
            &packet,
            &frame,
            self.settings,
            encoder,
        )?;
        self.clock.add("draw (upload, record)", start);
        let (lit, vector) = (targets.lit.clone(), targets.vector.clone());
        let depths = (
            targets.depth_view.clone(),
            targets.vector_depth_view.clone(),
        );
        let frame_uniform = self.renderer.frame_uniform().clone();
        let lit = self.composer.lit_effects(
            &self.gpu,
            encoder,
            &lit,
            &frame_uniform,
            &mut self.renderer.profiler,
        )?;
        let camera = &settings.camera;
        let depth_of_field = (camera.aperture > 0.0).then_some(DepthOfField {
            focus_distance: camera.focus_distance as f32,
            aperture: camera.aperture as f32,
            mesh_depth: &depths.0,
            vector_depth: &depths.1,
            samples: self.settings.samples,
            frame_uniform: &frame_uniform,
            size,
        });
        self.composer.composite(
            &self.gpu,
            encoder,
            &lit,
            &vector,
            target,
            self.background,
            &mut self.renderer.profiler,
            depth_of_field,
        )
    }

    /// CPU milliseconds per stage since the last call.
    pub fn stage_timings(&mut self) -> Vec<(&'static str, f64)> {
        self.clock.read()
    }

    fn submit(&mut self, encoder: wgpu::CommandEncoder) {
        let start = Instant::now();
        self.gpu.queue.submit([encoder.finish()]);
        self.clock.add("submit", start);
    }

    /// A texture of the composer's pool that frames are copied out of.
    fn texture(
        &mut self,
        name: &'static str,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        self.composer.textures.get(
            &self.gpu.device,
            name,
            size,
            format,
            1,
            wgpu::TextureUsages::COPY_SRC,
        )
    }

    /// Read the oldest frame on the GPU back into `ready`, without the row
    /// padding (waits for it).
    fn finish_oldest(&mut self) -> Result<()> {
        let frame = self
            .pending
            .pop_front()
            .ok_or_else(|| Error::Invalid("no frame in flight".into()))?;
        // Wait for this frame's copy only; later frames keep running.
        let start = Instant::now();
        while !frame.mapped.load(Ordering::Acquire) {
            let wait = wgpu::PollType::Wait {
                submission_index: Some(frame.submission.clone()),
                timeout: None,
            };
            self.gpu
                .device
                .poll(wait)
                .map_err(|error| Error::Gpu(error.to_string()))?;
        }
        let size = (self.settings.width, self.settings.height);
        let (row_length, rows) = self.pixel_format.layout(size);
        let (row_length, rows) = (row_length as usize, rows as usize);
        self.clock.add("read-back wait", start);
        let start = Instant::now();
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
        self.clock.add("read-back copy", start);
        Ok(())
    }
}
