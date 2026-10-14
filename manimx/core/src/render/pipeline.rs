//! `RenderPipeline`: frames of render object trees through the stages
//! Pack, Draw, Composite and Output (core/README.md). Python drives it
//! through python/render_pipeline.rs.
//!
//! A frame is a list of sub-frames (time weight and views; motion blur
//! averages several), each view a render object tree and its
//! `ViewSettings`. For each view the pipeline packs the tree (`Packer`),
//! draws the packet into layers (`Drawer`) and composites them into the
//! frame (`Compositor`); `Output` finishes the frame and reads it back.
//! Today the stages run one after the other on the caller's thread, while
//! up to `in_flight` frames are on the GPU.
//!
//! A frame that shows exactly what the frame before showed is not
//! rendered: equal trees (`RNode::fingerprint`) seen with equal settings
//! give equal pictures.

use std::sync::Arc;
use std::time::Instant;

use crate::render::composite::effects::depth_of_field::DepthOfField;
use crate::render::composite::effects::Effect;
use crate::render::composite::{CompositeTarget, Compositor};
use crate::render::draw::{Drawer, TargetSettings};
use crate::render::error::{Error, Result};
use crate::render::gpu::device::{Gpu, HDR_FORMAT};
use crate::render::gpu::fullscreen::Blend;
use crate::render::gpu::inspect::TextureKind;
use crate::render::gpu::profiler::{GpuProfiler, StageClock};
use crate::render::output::{Output, PixelFormat};
use crate::render::pack::packer::{Packer, Packet};
use crate::render::pack::records::Frame;
use crate::render::pack::vectors::VectorDraws;
use crate::render::settings::ViewSettings;
use crate::robjects::RNode;

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

/// What stays the same for every frame of a pipeline.
pub struct PipelineOptions {
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
    /// The largest page of vector records, in bytes per buffer
    /// (pack/vectors.rs); None: the most the device binds at once.
    pub page_bytes: Option<u64>,
}

pub struct RenderPipeline {
    gpu: Arc<Gpu>,
    packer: Packer,
    drawer: Drawer,
    compositor: Compositor,
    output: Output,
    settings: TargetSettings,
    background: [f32; 4],
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
        let gpu = Gpu::shared()?;
        let size = (options.width, options.height);
        let limits = gpu.device.limits();
        let page_bytes = options.page_bytes.unwrap_or_else(|| {
            limits.max_storage_buffer_binding_size.min(limits.max_buffer_size)
        });
        Ok(RenderPipeline {
            packer: Packer::with_page_bytes(page_bytes as usize),
            drawer: Drawer::new(&gpu, page_bytes)?,
            compositor: Compositor::new(&gpu, options.effects, options.tone_map),
            output: Output::new(&gpu, size, options.pixel_format, options.in_flight)?,
            profiler: GpuProfiler::new(&gpu.device, options.profile && gpu.timestamps),
            gpu,
            settings: TargetSettings {
                width: size.0,
                height: size.1,
                samples: options.samples,
                shadow_resolution: options.shadow_resolution,
            },
            background: options.background,
            clock: StageClock::default(),
            last_key: None,
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
        let accumulator = self.compositor.textures.get(
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
                // One submission per sub-frame: the drawer's passes and
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
                    Some(view) => self.composite_view(&mut encoder, view, target)?,
                    None => self.composite_empty(&mut encoder, target)?,
                }
                self.submit(encoder);
                continue;
            }
            // Several views: each is drawn and submitted on its own (the
            // drawer's buffers hold one view at a time) and put over the
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
            for (number, view) in sub_frame.views.iter().enumerate() {
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
        let frame_uniform = self.drawer.frame_uniform().clone();
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
        Ok(false)
    }

    /// Frames finished so far, `(index, pixels)` in submission order;
    /// with `wait`, waits for every frame on the GPU.
    pub fn finished(&mut self, wait: bool) -> Result<Vec<(u64, Vec<u8>)>> {
        self.output.finished(&self.gpu, wait, &mut self.clock)
    }

    /// A texture of the last frame as RGBA floats, to look at how it was
    /// made (see gpu/inspect.rs): `(width, height, texels)`.
    pub fn inspect(&mut self, name: &str, layer: u32) -> Result<(u32, u32, Vec<f32>)> {
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let (view, kind, size) = match self.drawer.inspected(name) {
            Some(found) => found,
            None => {
                let texture = self
                    .compositor
                    .textures
                    .find(name)
                    .or_else(|| self.output.texture(name))
                    .ok_or_else(|| Error::Invalid(format!("no texture named {name:?}")))?;
                let size = (texture.width(), texture.height());
                (
                    texture.create_view(&Default::default()),
                    TextureKind::Float,
                    size,
                )
            }
        };
        let texels = crate::render::gpu::inspect::read(&self.gpu, &view, kind, size, layer)?;
        Ok((size.0, size.1, texels))
    }

    /// How many shadow map layers there are.
    pub fn shadow_layers(&self) -> u32 {
        self.drawer.shadow_layers()
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
    /// view the tree's and the settings' fingerprints. None with a shader
    /// effect or a surface shader in a view, which may read the time.
    fn frame_key(&self, sub_frames: &[SubFrame]) -> Option<Vec<u64>> {
        if self.compositor.has_shader_effects() {
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

    /// Draw an empty scene and composite it into `target`.
    fn composite_empty(
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
        let targets = self.drawer.draw(
            &self.gpu,
            &mut self.packer,
            &packet,
            &frame,
            self.settings,
            encoder,
            &mut self.profiler,
        )?;
        let (lit, vector) = (targets.lit.clone(), targets.vector.clone());
        self.compositor.composite(
            &self.gpu,
            encoder,
            &lit,
            &vector,
            target,
            self.background,
            &mut self.profiler,
            None,
        )
    }

    /// Pack, draw and composite one view into `target`. The pictures of
    /// its camera views are made and submitted first.
    fn composite_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &View,
        target: CompositeTarget,
    ) -> Result<()> {
        let camera_views = &view.settings.camera_views;
        if !camera_views.is_empty() {
            let size = (self.settings.width, self.settings.height);
            self.drawer
                .ensure_camera_views(&self.gpu, camera_views.len(), size);
            for (layer, camera_view) in camera_views.iter().enumerate() {
                // Each is submitted on its own: the drawer's buffers hold
                // one view at a time.
                let mut picture_encoder =
                    self.gpu.device.create_command_encoder(&Default::default());
                let layer_view = self.drawer.camera_view_layer(layer);
                let picture = CompositeTarget {
                    view: &layer_view,
                    weight: 1.0,
                    clear: true,
                    background: true,
                    blend: Blend::Replace,
                };
                let settings = view.settings.picture(camera_view);
                self.composite_tree(&mut picture_encoder, &view.root, &settings, picture)?;
                self.submit(picture_encoder);
            }
        }
        self.composite_tree(encoder, &view.root, &view.settings, target)
    }

    /// Pack and draw the tree under `root` seen with `settings`, and
    /// composite the layers into `target`.
    fn composite_tree(
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
        let targets = self.drawer.draw(
            &self.gpu,
            &mut self.packer,
            &packet,
            &frame,
            self.settings,
            encoder,
            &mut self.profiler,
        )?;
        self.clock.add("draw (upload, record)", start);
        let (lit, vector) = (targets.lit.clone(), targets.vector.clone());
        let depths = (
            targets.depth_view.clone(),
            targets.vector_depth_view.clone(),
        );
        let frame_uniform = self.drawer.frame_uniform().clone();
        let lit = self.compositor.lit_effects(
            &self.gpu,
            encoder,
            &lit,
            &frame_uniform,
            &mut self.profiler,
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
        self.compositor.composite(
            &self.gpu,
            encoder,
            &lit,
            &vector,
            target,
            self.background,
            &mut self.profiler,
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
}
