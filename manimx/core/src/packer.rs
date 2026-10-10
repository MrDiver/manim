//! The packer: a scene graph and a frame state in, the records of a
//! `FramePacket` out. The core's counterpart of `SceneFlattener` plus
//! `FramePacker` (manimx/renderer/webgpu/packer.py); Python wraps the
//! result in a `FramePacket` (manimx/renderer/webgpu/core_packer.py), and
//! `WGPURenderer` renders it unchanged.
//!
//! One packer lives as long as the render service: its arenas mirror the
//! renderer's buffers, and its caches keep what Python built per version.

use std::collections::HashMap;

use glam::{DMat4, DVec3};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::callbacks::Callbacks;
use crate::evaluator::FrameState;
use crate::graph::{NodeData, SceneGraph, Values};
use crate::lights::{self, LightColors};
use crate::meshes::MeshPacker;
use crate::records::Light;
use crate::shadows::{self, Bounds, FrameSettings, ShadowLayers};
use crate::vectors::VectorPacker;

/// Packs after which per-node data unused since is dropped (`CACHE_MAX_AGE`).
const CACHE_MAX_AGE: u64 = 1024;

#[pyclass(module = "manimx._core")]
pub struct Packer {
    callbacks: Callbacks,
    vectors: VectorPacker,
    meshes: MeshPacker,
    light_colors: LightColors,
    bounds: Bounds,
    shadow_layers: ShadowLayers,
    generation: u64,
    /// The pack in which each node was last drawn.
    last_used: HashMap<u64, u64>,
}

/// What a pack needs from the `RenderConfig` (see `config_values` in
/// core_packer.py).
struct PackSettings {
    frame: FrameSettings,
    default_lights: bool,
    camera_position: DVec3,
    /// The camera's world matrix.
    camera: DMat4,
}

#[pymethods]
impl Packer {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Packer {
            callbacks: Callbacks::load(py)?,
            vectors: VectorPacker::new(py),
            meshes: MeshPacker::new(py),
            light_colors: LightColors::default(),
            bounds: Bounds::default(),
            shadow_layers: ShadowLayers::default(),
            generation: 0,
            last_used: HashMap::new(),
        })
    }

    /// Pack the scene `graph` shows with the values of `state`. `config`
    /// is the tuple of `config_values()` in core_packer.py. Returns a dict
    /// of numpy views and bytes that core_packer.py turns into a packet.
    fn pack<'py>(
        &mut self,
        py: Python<'py>,
        graph: &SceneGraph,
        state: &FrameState,
        config: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let settings = parse_settings(config)?;
        self.generation += 1;
        if self.generation.is_multiple_of(CACHE_MAX_AGE) {
            self.sweep();
        }
        let values = Values { state, py };
        let scene = graph.flatten(&values)?;
        for item in scene
            .items
            .iter()
            .chain(scene.clips.iter().flat_map(|scope| scope.mask.iter()))
        {
            self.last_used
                .insert(graph.nodes[item.node].uid, self.generation);
        }

        let draws = self.vectors.pack(
            py,
            graph,
            &values,
            &scene,
            settings.camera_position.to_array(),
            &self.callbacks,
            self.generation,
        )?;
        let result = PyDict::new(py);
        let vectors = PyDict::new(py);
        let dirty = PyDict::new(py);
        for (name, arena) in self.vectors.arenas_mut() {
            vectors.set_item(name, arena.view(py)?)?;
            dirty.set_item(name, arena.ranges())?;
        }
        vectors.set_item("opaque_draws", PyBytes::new(py, &draws.opaque))?;
        vectors.set_item("transparent_draws", PyBytes::new(py, &draws.transparent))?;
        vectors.set_item(
            "mask_draws",
            PyList::new(py, draws.masks.iter().map(|m| PyBytes::new(py, m)))?,
        )?;
        vectors.set_item(
            "shadow_draws",
            shadow_list(
                py,
                draws
                    .shadows
                    .iter()
                    .map(|(scope, d)| (*scope, d.as_slice())),
            )?,
        )?;
        vectors.set_item("dirty", dirty)?;
        result.set_item("vectors", vectors)?;
        result.set_item(
            "scope_parents",
            scene
                .clips
                .iter()
                .map(|scope| scope.parent)
                .collect::<Vec<_>>(),
        )?;
        result.set_item("light_count", 0)?;
        result.set_item("meshes", py.None())?;
        result.set_item("shadows", py.None())?;

        // No mesh in the frame: no lights, shadows or 3D passes (as in the
        // reference, any Shape3D counts, even one without triangles).
        if !scene
            .items
            .iter()
            .any(|item| matches!(graph.nodes[item.node].data, NodeData::Mesh(_)))
        {
            return Ok(result);
        }
        if self.meshes.too_much_garbage() {
            self.meshes.repack(py);
        }
        let output = self
            .meshes
            .pack(py, graph, &values, &scene, &self.callbacks)?;
        let sources = lights::light_sources(
            py,
            graph,
            &values,
            &scene,
            &mut self.light_colors,
            &self.callbacks,
            settings.default_lights,
            &settings.camera,
        )?;
        let mut rows: Vec<Light> = sources.iter().map(|light| light.record()).collect();
        let mut shadow_views = shadows::build_shadow_views(
            py,
            graph,
            &values,
            &scene,
            &sources,
            &mut rows,
            &mut self.bounds,
            &self.meshes,
            &self.callbacks,
            &settings.frame,
        )?;
        self.shadow_layers
            .mark(graph, &values, &scene, &mut shadow_views)?;

        let meshes = PyDict::new(py);
        meshes.set_item(
            "objects",
            PyBytes::new(py, bytemuck::cast_slice(&output.objects)),
        )?;
        meshes.set_item(
            "materials",
            PyBytes::new(py, bytemuck::cast_slice(&output.materials)),
        )?;
        meshes.set_item("lights", PyBytes::new(py, bytemuck::cast_slice(&rows)))?;
        meshes.set_item("vertices", self.meshes.vertices.view(py)?)?;
        meshes.set_item("indices", self.meshes.indices.view(py)?)?;
        let groups = PyList::empty(py);
        for group in &output.groups {
            let textures =
                pyo3::types::PyTuple::new(py, group.textures.iter().map(|t| t.bind(py)))?;
            groups.append((
                textures,
                PyBytes::new(py, bytemuck::cast_slice(&group.draws)),
            ))?;
        }
        meshes.set_item("groups", groups)?;
        meshes.set_item(
            "shadow_draws",
            shadow_list(
                py,
                output
                    .shadow_draws
                    .iter()
                    .map(|(scope, d)| (*scope, bytemuck::cast_slice(d))),
            )?,
        )?;
        let mesh_dirty = PyDict::new(py);
        mesh_dirty.set_item("vertices", self.meshes.vertices.ranges())?;
        mesh_dirty.set_item("indices", self.meshes.indices.ranges())?;
        meshes.set_item("dirty", mesh_dirty)?;
        result.set_item("meshes", meshes)?;
        result.set_item("light_count", rows.len())?;

        let shadows = PyDict::new(py);
        shadows.set_item(
            "frames",
            PyBytes::new(py, bytemuck::cast_slice(&shadow_views.frames)),
        )?;
        shadows.set_item(
            "views",
            PyBytes::new(py, bytemuck::cast_slice(&shadow_views.views)),
        )?;
        shadows.set_item("scopes", shadow_views.scopes)?;
        shadows.set_item("changed", shadow_views.changed)?;
        result.set_item("shadows", shadows)?;
        Ok(result)
    }
}
impl Packer {
    /// Drop the per-node data of nodes not drawn for `CACHE_MAX_AGE` packs.
    fn sweep(&mut self) {
        let oldest = self.generation.saturating_sub(CACHE_MAX_AGE);
        self.last_used.retain(|_, used| *used >= oldest);
        let live = &self.last_used;
        self.vectors.paths.retain(|uid, _| live.contains_key(uid));
        self.meshes.geometry.retain(|uid, _| live.contains_key(uid));
    }
}

fn shadow_list<'py, 'a>(
    py: Python<'py>,
    draws: impl Iterator<Item = (i32, &'a [u8])>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for (scope, bytes) in draws {
        list.append((scope, PyBytes::new(py, bytes)))?;
    }
    Ok(list)
}

/// A 4x4 matrix from 16 row-major values (numpy's layout).
fn row_major(values: &[f64]) -> DMat4 {
    DMat4::from_cols_array(&values.try_into().expect("16 values")).transpose()
}

fn parse_settings(config: &Bound<'_, PyAny>) -> PyResult<PackSettings> {
    let item = |index: usize| config.get_item(index);
    let ambient: (f64, f64, f64) = item(3)?.extract()?;
    let camera_position: (f64, f64, f64) = item(9)?.extract()?;
    let camera: Vec<f64> = item(10)?.extract()?;
    Ok(PackSettings {
        frame: FrameSettings {
            time: item(0)?.extract()?,
            layer_depth_step: item(1)?.extract()?,
            exposure: item(2)?.extract()?,
            ambient_linear: [ambient.0, ambient.1, ambient.2],
            shadow_softness: item(4)?.extract()?,
            shadow_samples: item(5)?.extract()?,
            shadow_resolution: item(6)?.extract()?,
            shadows: item(7)?.extract()?,
        },
        default_lights: item(8)?.extract()?,
        camera_position: DVec3::new(camera_position.0, camera_position.1, camera_position.2),
        camera: row_major(&camera),
    })
}
