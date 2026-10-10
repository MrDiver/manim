//! `FrameRenderer`: the core's whole render chain behind one Python class.
//! The counterpart of `FrameComposer` plus `FramePipeline`
//! (manimx/renderer/webgpu/composer.py, manimx/drivers/pipeline.py) for
//! frames of the core (PlannedFrames), driven by `CoreFramePipeline`
//! (manimx/drivers/core_pipeline.py).
//!
//! Per frame, Python hands over the sub-frames (time weight and views, each
//! view a scene graph, its frame state and the camera's Frame record);
//! the renderer packs, renders, composites and finalizes them, then copies
//! the output into a staging buffer that is mapped while the next frames
//! are prepared. Up to `in_flight` frames are on the GPU at once; finished
//! frames come back in submission order.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use super::composer::{Blend, Bloom, Composer, CompositeTarget, Effect, ShaderEffect};
use super::context::{Gpu, COLOR_FORMAT, HDR_FORMAT};
use super::renderer::{Renderer, TargetSettings};
use crate::evaluator::FrameState;
use crate::graph::SceneGraph;
use crate::packer::{parse_settings, Packer, Packet};
use crate::records::Frame;
use crate::vectors::VectorDraws;

/// Bytes per row of texture-to-buffer copies must be multiples of this.
const ROW_ALIGNMENT: u32 = 256;

/// A frame whose copy into `staging` is on its way.
struct InFlight {
    index: u64,
    staging: wgpu::Buffer,
    /// The submission that copies the frame into `staging`.
    submission: wgpu::SubmissionIndex,
    mapped: Arc<AtomicBool>,
}

#[pyclass(module = "manimx._core")]
pub struct FrameRenderer {
    gpu: Arc<Gpu>,
    packer: Packer,
    renderer: Renderer,
    composer: Composer,
    settings: TargetSettings,
    background: [f32; 4],
    row_bytes: u32,
    free: Vec<wgpu::Buffer>,
    pending: VecDeque<InFlight>,
    /// Frames read back but not yet handed to Python, `(index, RGBA bytes)`.
    ready: VecDeque<(u64, Py<PyBytes>)>,
    /// The key of the last rendered frame (None: never held).
    last_key: Option<Vec<u8>>,
}

#[pymethods]
impl FrameRenderer {
    /// `effects` is a list of `("bloom", threshold, knee, intensity,
    /// radius, levels)` and `("shader", code, stage, values)`, in chain
    /// order; see core_pipeline.py.
    #[new]
    #[pyo3(signature = (shader_folder, width, height, samples, shadow_resolution, background, tone_map, effects, in_flight, profile=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        shader_folder: PathBuf,
        width: u32,
        height: u32,
        samples: u32,
        shadow_resolution: u32,
        background: (f32, f32, f32, f32),
        tone_map: u32,
        effects: Vec<Bound<'_, PyTuple>>,
        in_flight: usize,
        profile: bool,
    ) -> PyResult<Self> {
        if in_flight < 1 {
            return Err(PyValueError::new_err(
                "a pipeline needs at least one frame in flight",
            ));
        }
        let gpu = Gpu::shared(shader_folder)?;
        let effects = effects
            .iter()
            .map(parse_effect)
            .collect::<PyResult<Vec<_>>>()?;
        let row_bytes = (4 * width).div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        let free = (0..in_flight)
            .map(|index| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("readback {index}")),
                    size: (row_bytes * height) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(FrameRenderer {
            packer: Packer::new(py)?,
            renderer: Renderer::new(py, &gpu, profile)?,
            composer: Composer::new(&gpu, effects, tone_map),
            gpu,
            settings: TargetSettings {
                width,
                height,
                samples,
                shadow_resolution,
            },
            background: [background.0, background.1, background.2, background.3],
            row_bytes,
            free,
            pending: VecDeque::new(),
            ready: VecDeque::new(),
            last_key: None,
        })
    }

    /// Render frame `index` from `sub_frames`: a list of `(weight, views)`,
    /// each view `(graph, state, key, config, frame, camera_views)` with
    /// the `config_values()` tuple, the camera's Frame record (bytes) and
    /// a `(config, frame)` pair per camera view of the view.
    /// Returns True when the frame shows exactly what the frame before
    /// showed: then nothing is rendered or read back for it.
    fn render(
        &mut self,
        py: Python<'_>,
        index: u64,
        sub_frames: Vec<(f32, Vec<Bound<'_, PyTuple>>)>,
    ) -> PyResult<bool> {
        let key = self.frame_key(&sub_frames)?;
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
        for (position, (weight, views)) in sub_frames.iter().enumerate() {
            let first = position == 0;
            if views.is_empty() {
                // Nothing active: the background alone.
                let mut encoder = device.create_command_encoder(&Default::default());
                let target = CompositeTarget {
                    view: &accumulator_view,
                    weight: *weight,
                    clear: first,
                    background: true,
                    blend: Blend::Additive,
                };
                self.render_empty(py, &mut encoder, target)?;
                self.gpu.queue.submit([encoder.finish()]);
                continue;
            }
            if views.len() == 1 {
                // One submission per sub-frame: the renderer's passes and
                // the composite share an encoder.
                let mut encoder = device.create_command_encoder(&Default::default());
                let target = CompositeTarget {
                    view: &accumulator_view,
                    weight: *weight,
                    clear: first,
                    background: true,
                    blend: Blend::Additive,
                };
                self.render_view(py, &mut encoder, &views[0], target)?;
                self.gpu.queue.submit([encoder.finish()]);
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
            for (number, view) in views.iter().enumerate() {
                let mut encoder = device.create_command_encoder(&Default::default());
                let target = CompositeTarget {
                    view: &stack_view,
                    weight: 1.0,
                    clear: number == 0,
                    background: number == 0,
                    blend: Blend::Over,
                };
                self.render_view(py, &mut encoder, view, target)?;
                self.gpu.queue.submit([encoder.finish()]);
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            self.composer.accumulate(
                &self.gpu,
                &mut encoder,
                &stack,
                &accumulator_view,
                *weight,
                first,
                &mut self.renderer.profiler,
            )?;
            self.gpu.queue.submit([encoder.finish()]);
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        let frame_uniform = self.renderer.frame_uniform().clone();
        let frame =
            self.composer
                .frame_effects(&self.gpu, &mut encoder, &accumulator, &frame_uniform)?;
        let output = self.composer.textures.get(
            &device,
            "output",
            size,
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::COPY_SRC,
        );
        self.composer.finalize(
            &self.gpu,
            &mut encoder,
            &frame,
            &output.create_view(&Default::default()),
            &mut self.renderer.profiler,
        )?;
        if self.free.is_empty() {
            self.finish_oldest(py)?;
        }
        let staging = self.free.pop().unwrap();
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.row_bytes),
                    rows_per_image: Some(size.1),
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        let submission = self.gpu.queue.submit([encoder.finish()]);
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

    /// Frames finished so far, `(index, RGBA bytes)` in submission order;
    /// with `wait`, waits for every frame on the GPU.
    #[pyo3(signature = (wait=false))]
    fn finished<'py>(&mut self, py: Python<'py>, wait: bool) -> PyResult<Bound<'py, PyList>> {
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        while let Some(front) = self.pending.front() {
            if !wait && !front.mapped.load(Ordering::Acquire) {
                break;
            }
            self.finish_oldest(py)?;
        }
        let done = PyList::empty(py);
        for (index, image) in self.ready.drain(..) {
            done.append((index, image))?;
        }
        Ok(done)
    }

    /// GPU milliseconds per pass name since the last call (with `profile`).
    fn gpu_timings<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let timings = PyDict::new(py);
        for (name, ms) in self
            .renderer
            .profiler
            .read(&self.gpu.device, &self.gpu.queue)
        {
            timings.set_item(name, ms)?;
        }
        Ok(timings)
    }
}

impl FrameRenderer {
    /// Equal keys promise equal frames: every view's graph, state key and
    /// Frame record (without the time). None with a shader effect, which
    /// may read the time.
    fn frame_key(
        &self,
        sub_frames: &[(f32, Vec<Bound<'_, PyTuple>>)],
    ) -> PyResult<Option<Vec<u8>>> {
        if self.composer.has_shader_effects() {
            return Ok(None);
        }
        let mut key = Vec::new();
        for (weight, views) in sub_frames {
            key.extend(weight.to_le_bytes());
            key.extend((views.len() as u64).to_le_bytes());
            for view in views {
                key.extend((view.get_item(0)?.as_ptr() as usize).to_le_bytes());
                key.extend(view.get_item(2)?.cast::<PyBytes>()?.as_bytes());
                let mut frame = frame_record(&view.get_item(4)?)?;
                frame.time = 0.0;
                key.extend(bytemuck::bytes_of(&frame));
                // The Frame record holds the camera and every config value a
                // pack reads, except whether the default light is on.
                key.push(parse_settings(&view.get_item(3)?)?.default_lights as u8);
                for camera_view in view.get_item(5)?.cast::<PyList>()?.iter() {
                    let mut frame = frame_record(&camera_view.get_item(1)?)?;
                    frame.time = 0.0;
                    key.extend(bytemuck::bytes_of(&frame));
                }
            }
        }
        Ok(Some(key))
    }

    /// Render an empty scene and composite it into `target`.
    fn render_empty(
        &mut self,
        py: Python<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target: CompositeTarget,
    ) -> PyResult<()> {
        let packet = Packet {
            draws: VectorDraws::default(),
            scope_parents: Vec::new(),
            meshes: None,
        };
        let frame: Frame = bytemuck::Zeroable::zeroed();
        let targets = self.renderer.render_layers(
            py,
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
        )
    }

    /// Pack and render one view and composite it into `target`. The
    /// pictures of its camera views are rendered and submitted first.
    fn render_view(
        &mut self,
        py: Python<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &Bound<'_, PyTuple>,
        target: CompositeTarget,
    ) -> PyResult<()> {
        let camera_views = view.get_item(5)?;
        let camera_views = camera_views.cast::<PyList>()?;
        if !camera_views.is_empty() {
            let size = (self.settings.width, self.settings.height);
            self.renderer
                .ensure_camera_views(&self.gpu, camera_views.len(), size);
            for (layer, camera_view) in camera_views.iter().enumerate() {
                // Each is submitted on its own: the renderer's buffers hold
                // one render at a time.
                let mut picture_encoder = self.gpu.device.create_command_encoder(&Default::default());
                let layer_view = self.renderer.camera_view_layer(layer);
                let picture = CompositeTarget {
                    view: &layer_view,
                    weight: 1.0,
                    clear: true,
                    background: true,
                    blend: Blend::Replace,
                };
                self.render_layers_into(
                    py,
                    &mut picture_encoder,
                    view,
                    &camera_view.get_item(0)?,
                    &camera_view.get_item(1)?,
                    picture,
                )?;
                self.gpu.queue.submit([picture_encoder.finish()]);
            }
        }
        self.render_layers_into(py, encoder, view, &view.get_item(3)?, &view.get_item(4)?, target)
    }

    /// Pack and render the graph and state of `view`, seen with `config`
    /// (a `config_values()` tuple) and `frame` (a Frame record), and
    /// composite the result into `target`.
    fn render_layers_into(
        &mut self,
        py: Python<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &Bound<'_, PyTuple>,
        config: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
        target: CompositeTarget,
    ) -> PyResult<()> {
        let graph = view.get_item(0)?;
        let graph = graph.cast::<SceneGraph>()?.get();
        let state = view.get_item(1)?;
        let state = state.cast::<FrameState>()?.get();
        let settings = parse_settings(config)?;
        let mut frame = frame_record(frame)?;
        let packet = self.packer.pack_frame(py, graph, state, &settings)?;
        frame.light_count = packet
            .meshes
            .as_ref()
            .map_or(0, |meshes| meshes.lights.len() as u32);
        let targets = self.renderer.render_layers(
            py,
            &self.gpu,
            &mut self.packer,
            &packet,
            &frame,
            self.settings,
            encoder,
        )?;
        let (lit, vector) = (targets.lit.clone(), targets.vector.clone());
        let frame_uniform = self.renderer.frame_uniform().clone();
        let lit = self.composer.lit_effects(
            &self.gpu,
            encoder,
            &lit,
            &frame_uniform,
            &mut self.renderer.profiler,
        )?;
        self.composer.composite(
            &self.gpu,
            encoder,
            &lit,
            &vector,
            target,
            self.background,
            &mut self.renderer.profiler,
        )
    }

    /// Read the oldest frame on the GPU back into `ready`, without the row
    /// padding (waits for it).
    fn finish_oldest(&mut self, py: Python<'_>) -> PyResult<()> {
        let frame = self
            .pending
            .pop_front()
            .ok_or_else(|| PyRuntimeError::new_err("no frame in flight"))?;
        // Wait for this frame's copy only; later frames keep running.
        while !frame.mapped.load(Ordering::Acquire) {
            let wait = wgpu::PollType::Wait {
                submission_index: Some(frame.submission.clone()),
                timeout: None,
            };
            self.gpu
                .device
                .poll(wait)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
        let (width, height) = (self.settings.width as usize, self.settings.height as usize);
        let row_bytes = self.row_bytes as usize;
        let image = {
            let data = frame
                .staging
                .slice(..)
                .get_mapped_range()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            // One copy, straight from the mapped memory into the Python
            // object, dropping the row padding.
            PyBytes::new_with(py, 4 * width * height, |image| {
                for (row, target) in data
                    .chunks_exact(row_bytes)
                    .zip(image.chunks_exact_mut(4 * width))
                {
                    target.copy_from_slice(&row[..4 * width]);
                }
                Ok(())
            })?
            .unbind()
        };
        frame.staging.unmap();
        self.free.push(frame.staging);
        self.ready.push_back((frame.index, image));
        Ok(())
    }
}

fn frame_record(bytes: &Bound<'_, PyAny>) -> PyResult<Frame> {
    let bytes = bytes.cast::<PyBytes>()?.as_bytes();
    if bytes.len() != size_of::<Frame>() {
        return Err(PyValueError::new_err("a Frame record has 400 bytes"));
    }
    Ok(bytemuck::pod_read_unaligned(bytes))
}

fn parse_effect(effect: &Bound<'_, PyTuple>) -> PyResult<Effect> {
    let kind: String = effect.get_item(0)?.extract()?;
    match kind.as_str() {
        "bloom" => Ok(Effect::Bloom(Bloom {
            threshold: effect.get_item(1)?.extract()?,
            knee: effect.get_item(2)?.extract()?,
            intensity: effect.get_item(3)?.extract()?,
            radius: effect.get_item(4)?.extract()?,
            levels: effect.get_item(5)?.extract()?,
        })),
        "shader" => Ok(Effect::Shader(ShaderEffect {
            code: effect.get_item(1)?.extract()?,
            frame_stage: effect.get_item(2)?.extract::<String>()? == "frame",
            values: effect.get_item(3)?.extract()?,
        })),
        _ => Err(PyValueError::new_err(format!("unknown effect {kind:?}"))),
    }
}
