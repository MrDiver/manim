//! The packer: a scene graph and a frame state in, the records of a frame
//! out. The core's counterpart of `SceneFlattener` plus `FramePacker`
//! (manimx/renderer/webgpu/packer.py).
//!
//! `Packer::pack_frame` returns a `Packet` for the core's own renderer
//! (gpu/); the Python method `pack` turns the same into a dict that
//! core_packer.py wraps in a `FramePacket` for the Python renderer.
//!
//! One packer lives as long as its renderer: its arenas mirror the
//! renderer's buffers, and its caches keep what Python built per version.

use std::collections::HashMap;

use glam::{DMat4, DVec3};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::callbacks::Callbacks;
use crate::evaluator::FrameState;
use crate::graph::{NodeData, SceneGraph, Values};
use crate::lights::{self, LightColors};
use crate::meshes::{MeshOutput, MeshPacker};
use crate::records::Light;
use crate::shadows::{self, Bounds, FrameSettings, ShadowLayers, ShadowViews};
use crate::vectors::{CameraViews, VectorDraws, VectorPacker};

/// Packs after which per-node data unused since is dropped (`CACHE_MAX_AGE`).
const CACHE_MAX_AGE: u64 = 1024;

/// What a renderer needs for one frame besides the packer's arenas.
pub struct Packet {
    pub draws: VectorDraws,
    /// Per clip scope, the index of the enclosing one (-1 for none).
    pub scope_parents: Vec<i32>,
    /// None when the frame has no Shape3D: lights, shadows and the 3D
    /// passes only affect meshes, so they are skipped.
    pub meshes: Option<MeshPacket>,
}

pub struct MeshPacket {
    pub output: MeshOutput,
    pub lights: Vec<Light>,
    pub shadows: ShadowViews,
}

/// What a pack needs from the `RenderConfig` (see `config_values` in
/// core_packer.py).
pub struct PackSettings {
    pub frame: FrameSettings,
    pub default_lights: bool,
    pub camera_position: DVec3,
    /// The camera's world matrix.
    pub camera: DMat4,
    /// The frame's camera views (`RenderConfig.camera_views`).
    pub camera_views: CameraViews,
    /// Leave the camera views' shapes out (while one is rendered).
    pub hide_camera_views: bool,
}

#[pyclass(module = "manimx._core")]
pub struct Packer {
    callbacks: Callbacks,
    pub vectors: VectorPacker,
    pub meshes: MeshPacker,
    light_colors: LightColors,
    bounds: Bounds,
    shadow_layers: ShadowLayers,
    generation: u64,
    /// The pack in which each node was last drawn.
    last_used: HashMap<u64, u64>,
}

#[pymethods]
impl Packer {
    #[new]
    pub fn new(py: Python<'_>) -> PyResult<Self> {
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
        let packet = self.pack_frame(py, graph, state, &parse_settings(config)?)?;
        self.packet_dict(py, packet)
    }
}

impl Packer {
    pub fn pack_frame(
        &mut self,
        py: Python<'_>,
        graph: &SceneGraph,
        state: &FrameState,
        settings: &PackSettings,
    ) -> PyResult<Packet> {
        self.generation += 1;
        if self.generation.is_multiple_of(CACHE_MAX_AGE) {
            self.sweep();
        }
        let values = Values { state, py };
        let mut scene = graph.flatten(&values)?;
        if settings.hide_camera_views {
            let views = &settings.camera_views;
            scene
                .items
                .retain(|item| !views.contains_key(&graph.nodes[item.node].uid));
        }
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
            &settings.camera_views,
            &self.callbacks,
            self.generation,
        )?;
        let scope_parents = scene.clips.iter().map(|scope| scope.parent).collect();

        // No mesh in the frame: no lights, shadows or 3D passes (as in the
        // reference, any Shape3D counts, even one without triangles).
        if !scene
            .items
            .iter()
            .any(|item| matches!(graph.nodes[item.node].data, NodeData::Mesh(_)))
        {
            return Ok(Packet {
                draws,
                scope_parents,
                meshes: None,
            });
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
        Ok(Packet {
            draws,
            scope_parents,
            meshes: Some(MeshPacket {
                output,
                lights: rows,
                shadows: shadow_views,
            }),
        })
    }

    /// The packet as the dict core_packer.py reads (views of the arenas,
    /// bytes of everything else); takes the arenas' dirty ranges.
    fn packet_dict<'py>(
        &mut self,
        py: Python<'py>,
        packet: Packet,
    ) -> PyResult<Bound<'py, PyDict>> {
        let draws = &packet.draws;
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
        result.set_item("scope_parents", &packet.scope_parents)?;
        let Some(meshes) = packet.meshes else {
            result.set_item("light_count", 0)?;
            result.set_item("meshes", py.None())?;
            result.set_item("shadows", py.None())?;
            return Ok(result);
        };
        let output = &meshes.output;
        let mesh = PyDict::new(py);
        mesh.set_item(
            "objects",
            PyBytes::new(py, bytemuck::cast_slice(&output.objects)),
        )?;
        mesh.set_item(
            "materials",
            PyBytes::new(py, bytemuck::cast_slice(&output.materials)),
        )?;
        mesh.set_item(
            "lights",
            PyBytes::new(py, bytemuck::cast_slice(&meshes.lights)),
        )?;
        mesh.set_item("vertices", self.meshes.vertices.view(py)?)?;
        mesh.set_item("indices", self.meshes.indices.view(py)?)?;
        let groups = PyList::empty(py);
        for group in &output.groups {
            let textures = PyTuple::new(py, group.textures.iter().map(|t| t.bind(py)))?;
            groups.append((
                textures,
                group.surface_shader.bind(py),
                PyBytes::new(py, bytemuck::cast_slice(&group.draws)),
            ))?;
        }
        mesh.set_item("groups", groups)?;
        mesh.set_item(
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
        mesh.set_item("dirty", mesh_dirty)?;
        result.set_item("meshes", mesh)?;
        result.set_item("light_count", meshes.lights.len())?;

        let shadows = PyDict::new(py);
        shadows.set_item(
            "frames",
            PyBytes::new(py, bytemuck::cast_slice(&meshes.shadows.frames)),
        )?;
        shadows.set_item(
            "views",
            PyBytes::new(py, bytemuck::cast_slice(&meshes.shadows.views)),
        )?;
        shadows.set_item("scopes", &meshes.shadows.scopes)?;
        shadows.set_item("changed", &meshes.shadows.changed)?;
        result.set_item("shadows", shadows)?;
        Ok(result)
    }

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
pub fn row_major(values: &[f64]) -> DMat4 {
    DMat4::from_cols_array(&values.try_into().expect("16 values")).transpose()
}

pub fn parse_settings(config: &Bound<'_, PyAny>) -> PyResult<PackSettings> {
    let item = |index: usize| config.get_item(index);
    let ambient: (f64, f64, f64) = item(3)?.extract()?;
    let camera_position: (f64, f64, f64) = item(9)?.extract()?;
    let camera: Vec<f64> = item(10)?.extract()?;
    let views: Vec<(u64, f32, f32)> = item(11)?.extract()?;
    let camera_views = views
        .into_iter()
        .enumerate()
        .map(|(layer, (uid, x, y))| (uid, (layer as i32, [x, y])))
        .collect();
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
        camera_views,
        hide_camera_views: item(12)?.extract()?,
    })
}
