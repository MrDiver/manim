//! The packer: a scene graph and a frame state in, the records of a frame
//! out. `Packer::pack_frame` flattens the graph (graph.rs), builds the
//! vector, mesh, light and shadow records (vectors.rs, meshes.rs,
//! lights.rs, shadows.rs) and returns them as a `Packet` for the renderer
//! (gpu/). It never touches the GPU.
//!
//! One packer lives as long as its renderer: its arenas mirror the
//! renderer's buffers, and its caches keep what Python built per version.

use std::collections::HashMap;

use glam::DMat4;
use pyo3::prelude::*;

use crate::evaluation::evaluator::FrameState;
use crate::packing::callbacks::Callbacks;
use crate::packing::lights::{self, LightColors};
use crate::packing::meshes::{MeshOutput, MeshPacker};
use crate::packing::records::Light;
use crate::packing::scene_graph::{NodeData, SceneGraph, Values};
use crate::packing::shadows::{self, Bounds, FrameSettings, ShadowLayers, ShadowViews};
use crate::packing::vectors::{CameraViews, VectorDraws, VectorPacker};

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
/// manimx/renderer/frame.py).
pub struct PackSettings {
    pub frame: FrameSettings,
    pub default_lights: bool,
    /// The camera's world matrix.
    pub camera: DMat4,
    /// The frame's camera views (`RenderConfig.camera_views`).
    pub camera_views: CameraViews,
    /// Leave the camera views' shapes out (while one is rendered).
    pub hide_camera_views: bool,
}

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

impl Packer {
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Packer {
            callbacks: Callbacks::load(py)?,
            vectors: VectorPacker::new(),
            meshes: MeshPacker::new(),
            light_colors: LightColors::default(),
            bounds: Bounds::default(),
            shadow_layers: ShadowLayers::default(),
            generation: 0,
            last_used: HashMap::new(),
        })
    }

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
            settings.camera.inverse(),
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
            self.meshes.repack();
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

    /// Drop the per-node data of nodes not drawn for `CACHE_MAX_AGE` packs.
    fn sweep(&mut self) {
        let oldest = self.generation.saturating_sub(CACHE_MAX_AGE);
        self.last_used.retain(|_, used| *used >= oldest);
        let live = &self.last_used;
        self.vectors.paths.retain(|uid, _| live.contains_key(uid));
        self.meshes.geometry.retain(|uid, _| live.contains_key(uid));
    }
}

/// A 4x4 matrix from 16 row-major values (numpy's layout).
pub fn row_major(values: &[f64]) -> DMat4 {
    DMat4::from_cols_array(&values.try_into().expect("16 values")).transpose()
}

pub fn parse_settings(config: &Bound<'_, PyAny>) -> PyResult<PackSettings> {
    let item = |index: usize| config.get_item(index);
    let ambient: (f64, f64, f64) = item(3)?.extract()?;
    let camera: Vec<f64> = item(9)?.extract()?;
    let views: Vec<(u64, f32, f32)> = item(10)?.extract()?;
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
            shadow_filter: item(14)?.extract()?,
        },
        default_lights: item(8)?.extract()?,
        camera: row_major(&camera),
        camera_views,
        hide_camera_views: item(11)?.extract()?,
    })
}
