//! `RenderPipeline`: frames of scenes, made by the compositor from the
//! layers the renderer renders. Python drives it through
//! python/renderer.rs.
//!
//! A frame is a list of sub-frames (time weight and views; motion blur
//! averages several), each view a scene's changes (an `Update`, flat/)
//! and its `ViewSettings`. For each view the pipeline sends the changes
//! to the renderer (render/), has it render the pictures of the view's
//! camera views and composites them, then has it render the view and
//! composites the layers into the frame (`Compositor`); `Output`
//! finishes the frame and reads it back. Today the stages run one after
//! the other on the caller's thread, while up to `in_flight` frames are
//! on the GPU.
//!
//! A frame that shows exactly what the frame before showed is not
//! rendered: equal scenes (equal revisions) seen with equal settings give
//! equal pictures.

use std::sync::Arc;
use std::time::Instant;

use crate::composite::effects::depth_of_field::DepthOfField;
use crate::composite::effects::Effect;
use crate::composite::output::{Output, PixelFormat};
use crate::composite::pictures::CameraViewTexture;
use crate::composite::{CompositeTarget, Compositor};
use crate::error::{Error, Result};
use crate::flat::Update;
use crate::gpu::device::{Gpu, HDR_FORMAT};
use crate::gpu::fullscreen::Blend;
use crate::gpu::profiler::{GpuProfiler, StageClock};
use crate::render::draw::TargetSettings;
use crate::render::renderer::{Layers, Renderer};
use crate::render::settings::ViewSettings;

/// One scene seen from one camera, and the settings to see it with.
pub struct View {
    pub source: ViewSource,
    pub settings: ViewSettings,
}

pub enum ViewSource {
    /// For the core's renderer: the scene's changes since it was last
    /// sent (flat/).
    Scene(Update),
    /// Rendered by another renderer (Cycles): the lit layer as RGBA half
    /// floats, linear and premultiplied, top row first, at the frame's
    /// size. The compositor treats it like the core's lit layer (lit-stage
    /// effects, tone mapping); there is no vector layer.
    Picture(Vec<u8>),
}

/// Views drawn over each other (lowest first), weighted into the frame.
pub struct SubFrame {
    pub weight: f32,
    pub views: Vec<View>,
}

/// MSAA samples per pixel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Samples {
    /// Exactly this many; the GPU must support it (`Gpu::sample_counts`).
    Count(u32),
    /// The most the GPU supports.
    Most,
}

impl Samples {
    /// The sample count on `gpu`.
    fn resolve(self, gpu: &Gpu) -> Result<u32> {
        let supported = &gpu.sample_counts;
        match self {
            Samples::Most => Ok(*supported.last().expect("1 is always supported")),
            Samples::Count(count) if supported.contains(&count) => Ok(count),
            Samples::Count(count) => Err(Error::Unsupported(format!(
                "{count} MSAA samples per pixel; this GPU renders with {supported:?}"
            ))),
        }
    }
}

/// What stays the same for every frame of a pipeline.
pub struct PipelineOptions {
    pub width: u32,
    pub height: u32,
    /// MSAA samples per pixel.
    pub samples: Samples,
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
    /// The largest page of vector records, in bytes per buffer
    /// (pack/vectors.rs); None: the most the device binds at once.
    pub page_bytes: Option<u64>,
    /// Which GPU renders: a dedicated one or an integrated one
    /// (`Gpu::shared`).
    pub gpu: wgpu::PowerPreference,
}

pub struct RenderPipeline {
    gpu: Arc<Gpu>,
    renderer: Renderer,
    compositor: Compositor,
    /// The pictures of the camera views of the view being made.
    pictures: CameraViewTexture,
    output: Output,
    settings: TargetSettings,
    background: [f32; 4],
    /// The `Frame` uniform of the last render (frame-stage effects read
    /// it).
    last_frame: Option<wgpu::Buffer>,
    /// GPU time per pass (with `PipelineOptions::profile`).
    profiler: GpuProfiler,
    /// CPU time per stage.
    clock: StageClock,
    /// The key of the last rendered frame (None: never held).
    last_key: Option<Vec<u64>>,
}

impl RenderPipeline {
    pub fn new(options: PipelineOptions) -> Result<RenderPipeline> {
        if options.in_flight < 1 {
            return Err(Error::Invalid(
                "a pipeline needs at least one frame in flight".into(),
            ));
        }
        let gpu = Gpu::shared(options.gpu)?;
        let size = (options.width, options.height);
        let limits = gpu.device.limits();
        let page_bytes = options.page_bytes.unwrap_or_else(|| {
            limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size)
        });
        let settings = TargetSettings {
            width: size.0,
            height: size.1,
            samples: options.samples.resolve(&gpu)?,
            shadow_resolution: options.shadow_resolution,
        };
        Ok(RenderPipeline {
            renderer: Renderer::new(&gpu, settings, page_bytes)?,
            compositor: Compositor::new(&gpu, options.effects, options.tone_map),
            pictures: CameraViewTexture::new(&gpu.device),
            output: Output::new(&gpu, size, options.pixel_format, options.in_flight)?,
            profiler: GpuProfiler::new(&gpu.device, options.profile && gpu.timestamps),
            gpu,
            settings,
            background: options.background,
            last_frame: None,
            clock: StageClock::default(),
            last_key: None,
        })
    }

    /// Render frame `index`. Returns true when the frame shows exactly
    /// what the frame before showed: then nothing is rendered or read
    /// back for it.
    pub fn render(&mut self, index: u64, sub_frames: Vec<SubFrame>) -> Result<bool> {
        let start = Instant::now();
        let held = self.render_frame(index, sub_frames);
        self.clock.add("render (all)", start);
        held
    }

    /// Drop everything kept for `scene` (it is sent whole again if it
    /// comes back).
    pub fn forget(&mut self, scene: u64) {
        self.renderer.forget(scene);
    }

    fn render_frame(&mut self, index: u64, mut sub_frames: Vec<SubFrame>) -> Result<bool> {
        let key = self.frame_key(&sub_frames);
        if key.is_some() && key == self.last_key {
            // Equal revisions: none of the updates changes anything.
            debug_assert!(sub_frames
                .iter()
                .flat_map(|sub_frame| &sub_frame.views)
                .all(|view| matches!(&view.source, ViewSource::Scene(update) if update.changes.is_empty())));
            return Ok(true);
        }
        // Set once the frame is rendered: a scene sent for the first time
        // has a key only once it is kept.
        self.last_key = None;
        let size = (self.settings.width, self.settings.height);
        let device = self.gpu.device.clone();
        let accumulator = self.compositor.textures.get(
            &device,
            "accumulator",
            size,
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::empty(),
        );
        let accumulator_view = accumulator.create_view(&Default::default());
        for (position, sub_frame) in sub_frames.iter_mut().enumerate() {
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
                match sub_frame.views.first_mut() {
                    Some(view) => self.composite_view(&mut encoder, view, target)?,
                    None => self.composite_empty(&mut encoder, target)?,
                }
                self.submit(encoder);
                continue;
            }
            // Several views: each is rendered and submitted on its own (the
            // renderer's layers hold one view at a time) and put over the
            // stack, which is then added to the accumulator.
            let stack = self.compositor.textures.get(
                &device,
                "view stack",
                size,
                HDR_FORMAT,
                1,
                wgpu::TextureUsages::empty(),
            );
            let stack_view = stack.create_view(&Default::default());
            for (number, view) in sub_frame.views.iter_mut().enumerate() {
                let mut encoder = device.create_command_encoder(&Default::default());
                let target = CompositeTarget {
                    view: &stack_view,
                    weight: 1.0,
                    clear: number == 0,
                    background: number == 0,
                    blend: Blend::Over,
                };
                self.composite_view(&mut encoder, view, target)?;
                self.submit(encoder);
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            self.compositor.accumulate(
                &self.gpu,
                &mut encoder,
                &stack,
                &accumulator_view,
                weight,
                first,
                &mut self.profiler,
            )?;
            self.submit(encoder);
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        let frame_uniform = self
            .last_frame
            .clone()
            .ok_or_else(|| Error::Invalid("a frame needs at least one sub-frame".into()))?;
        let frame =
            self.compositor
                .frame_effects(&self.gpu, &mut encoder, &accumulator, &frame_uniform)?;
        self.output.send(
            &self.gpu,
            encoder,
            &frame,
            index,
            &mut self.profiler,
            &mut self.clock,
        )?;
        self.last_key = self.frame_key(&sub_frames);
        Ok(false)
    }

    /// Frames finished so far, `(index, pixels)` in submission order;
    /// with `wait`, waits for every frame on the GPU.
    pub fn finished(&mut self, wait: bool) -> Result<Vec<(u64, Vec<u8>)>> {
        self.output.finished(&self.gpu, wait, &mut self.clock)
    }

    /// GPU milliseconds per pass name since the last call (with `profile`).
    pub fn gpu_timings(&mut self) -> Vec<(String, f64)> {
        self.profiler
            .read(&self.gpu.device, &self.gpu.queue)
            .into_iter()
            .map(|(name, ms)| (name.to_string(), ms))
            .collect()
    }

    /// Equal keys promise equal frames: per sub-frame its weight and per
    /// view its scene, the revision its update brings it to and the
    /// settings' fingerprint. None with a shader effect, a surface shader
    /// in a view (either may read the time), a scene not kept yet or a
    /// picture of another renderer.
    ///
    /// Read before any update is applied: a scene whose revisions all
    /// stay the same has no changes, so what it keeps is current.
    fn frame_key(&self, sub_frames: &[SubFrame]) -> Option<Vec<u64>> {
        if self.compositor.has_shader_effects() {
            return None;
        }
        let mut key = Vec::new();
        for sub_frame in sub_frames {
            key.extend([
                sub_frame.weight.to_bits() as u64,
                sub_frame.views.len() as u64,
            ]);
            for view in &sub_frame.views {
                let ViewSource::Scene(update) = &view.source else {
                    return None;
                };
                let (_, uses_time) = self.renderer.state(update.scene)?;
                if uses_time {
                    return None;
                }
                key.extend([update.scene, update.revision]);
                view.settings.fingerprint(&mut key);
            }
        }
        Some(key)
    }

    /// Composite a picture of another renderer (`ViewSource::Picture`)
    /// into `target` as the lit layer, with an empty vector layer. Depth
    /// of field needs depths, which a picture does not bring: its
    /// renderer blurs it itself.
    fn composite_picture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pixels: &[u8],
        target: CompositeTarget,
    ) -> Result<()> {
        let size = (self.settings.width, self.settings.height);
        let expected = size.0 as usize * size.1 as usize * 8;
        if pixels.len() != expected {
            return Err(Error::Invalid(format!(
                "a picture of {}x{} pixels needs {expected} bytes of RGBA half floats, got {}",
                size.0,
                size.1,
                pixels.len()
            )));
        }
        // The empty scene's layers: a cleared vector layer and the frame
        // uniform the effects read.
        let empty = self.renderer.render_empty(
            &self.gpu,
            encoder,
            self.pictures.pictures(),
            &mut self.profiler,
        )?;
        let lit = self.compositor.textures.get(
            &self.gpu.device,
            "picture",
            size,
            HDR_FORMAT,
            1,
            wgpu::TextureUsages::COPY_DST,
        );
        self.gpu.queue.write_texture(
            lit.as_image_copy(),
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0 * 8),
                rows_per_image: Some(size.1),
            },
            lit.size(),
        );
        let lit = self.compositor.lit_effects(
            &self.gpu,
            encoder,
            &lit,
            &empty.frame,
            &mut self.profiler,
        )?;
        self.compositor.composite(
            &self.gpu,
            encoder,
            &lit,
            &empty.vector,
            target,
            self.background,
            &mut self.profiler,
            None,
        )?;
        self.last_frame = Some(empty.frame);
        Ok(())
    }

    /// Render an empty scene and composite it into `target`.
    fn composite_empty(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: CompositeTarget,
    ) -> Result<()> {
        let layers = self.renderer.render_empty(
            &self.gpu,
            encoder,
            self.pictures.pictures(),
            &mut self.profiler,
        )?;
        self.compositor.composite(
            &self.gpu,
            encoder,
            &layers.lit,
            &layers.vector,
            target,
            self.background,
            &mut self.profiler,
            None,
        )?;
        self.last_frame = Some(layers.frame);
        Ok(())
    }

    /// Send one view's update to the renderer, then render and composite
    /// the view into `target`. The pictures of its camera views are
    /// rendered, composited and submitted first.
    fn composite_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &mut View,
        target: CompositeTarget,
    ) -> Result<()> {
        let update = match &mut view.source {
            ViewSource::Scene(update) => update,
            ViewSource::Picture(pixels) => return self.composite_picture(encoder, pixels, target),
        };
        let scene = update.scene;
        self.renderer.apply(Update {
            scene,
            revision: update.revision,
            changes: std::mem::take(&mut update.changes),
        })?;
        let camera_views = &view.settings.camera_views;
        if !camera_views.is_empty() {
            let size = (self.settings.width, self.settings.height);
            self.pictures
                .ensure(&self.gpu.device, camera_views.len(), size);
            for (layer, camera_view) in camera_views.iter().enumerate() {
                // Each is submitted on its own: the renderer's layers hold
                // one view at a time.
                let mut picture_encoder =
                    self.gpu.device.create_command_encoder(&Default::default());
                let layer_view = self.pictures.layer(layer).clone();
                let picture = CompositeTarget {
                    view: &layer_view,
                    weight: 1.0,
                    clear: true,
                    background: true,
                    blend: Blend::Replace,
                };
                let settings = view.settings.picture(camera_view);
                self.composite_scene(&mut picture_encoder, scene, &settings, picture)?;
                self.submit(picture_encoder);
            }
        }
        self.composite_scene(encoder, scene, &view.settings, target)
    }

    /// Have the renderer render `scene` seen with `settings`, and
    /// composite its layers into `target`.
    fn composite_scene(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: u64,
        settings: &ViewSettings,
        target: CompositeTarget,
    ) -> Result<()> {
        let size = (self.settings.width, self.settings.height);
        let Layers {
            lit,
            vector,
            mesh_depth,
            vector_depth,
            frame,
        } = self.renderer.render(
            &self.gpu,
            encoder,
            scene,
            settings,
            self.pictures.pictures(),
            &mut self.profiler,
            &mut self.clock,
        )?;
        let lit =
            self.compositor
                .lit_effects(&self.gpu, encoder, &lit, &frame, &mut self.profiler)?;
        let camera = &settings.camera;
        let depth_of_field = (camera.aperture > 0.0).then_some(DepthOfField {
            focus_distance: camera.focus_distance as f32,
            aperture: camera.aperture as f32,
            mesh_depth: &mesh_depth,
            vector_depth: &vector_depth,
            samples: self.settings.samples,
            frame_uniform: &frame,
            size,
        });
        self.compositor.composite(
            &self.gpu,
            encoder,
            &lit,
            &vector,
            target,
            self.background,
            &mut self.profiler,
            depth_of_field,
        )?;
        self.last_frame = Some(frame);
        Ok(())
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
}
