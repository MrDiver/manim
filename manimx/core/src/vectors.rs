//! The retained vector records of `Shape2D` and `Curve3D` nodes: one slot
//! per node, records kept in arenas across frames, only what changed is
//! written. Ports `RetainedVectors` (manimx/renderer/webgpu/retained.py);
//! read its module docs first.
//!
//! What depends on geometry or style alone (a fill's triangles, a stroke's
//! curve links, per-point colors) is built by Python once per version of
//! the signals it comes from (manimx/renderer/core_callbacks.py) and kept
//! in `PathCache`.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use crate::arena::Arena;
use crate::callbacks::{bytes_of, Callbacks};
use crate::graph::{DrawItem, FlatScene, Node, NodeData, PathFields, SceneGraph, Values, NO_CLIP};
use crate::records::{self, DrawIndirect, Instance, Object, StrokeCurve};
use glam::{DMat4, DVec3};

/// Where one path's records live, and what they were built from (`_Slot`).
struct Slot {
    object: usize,
    world: Option<DMat4>,
    placement: (usize, i32, u32, f64),
    /// Points version the geometry ranges below were built from.
    points_version: u64,
    fill_version: Option<u64>,
    fill_first: usize,
    fill_count: usize,
    fill_instance: Option<usize>,
    colors_key: Option<Vec<u64>>,
    colors_first: usize,
    colors_count: usize,
    stroke_key: Option<Vec<u64>>,
    points_first: usize,
    points_count: usize,
    curves_first: usize,
    curves_count: usize,
    stroke_instances: Option<usize>,
}

impl Slot {
    fn new(object: usize) -> Slot {
        Slot {
            object,
            world: None,
            placement: (usize::MAX, NO_CLIP - 1, u32::MAX, -1.0),
            points_version: 0,
            fill_version: None,
            fill_first: 0,
            fill_count: 0,
            fill_instance: None,
            colors_key: None,
            colors_first: 0,
            colors_count: 0,
            stroke_key: None,
            points_first: 0,
            points_count: 0,
            curves_first: 0,
            curves_count: 0,
            stroke_instances: None,
        }
    }
}

/// Records built by Python for one path, each with the versions it was
/// built from (`RNodeDataCache` entries of the reference path).
#[derive(Default)]
pub struct PathCache {
    center: Option<(u64, [f64; 3])>,
    fill_vertices: Option<(u64, Vec<u8>)>,
    fill_colors: Option<(Vec<u64>, Vec<u8>, bool, bool)>,
    stroke: Option<(Vec<u64>, StrokeRecords)>,
    pub last_used: u64,
}

struct StrokeRecords {
    curves: Vec<u8>,
    points: Vec<u8>,
    visible: bool,
    opaque: bool,
}

/// One indirect draw before encoding, with what it needs for sorting (`_Draw`).
pub struct Draw {
    pub args: DrawIndirect,
    pub opaque: bool,
    pub camera_distance: f64,
    pub layer: usize,
    pub mask_of: i32,
    pub casts_shadow: bool,
    pub clip: i32,
}

/// The draw lists of one frame (the rest of `VectorBatch` is the arenas).
#[derive(Default)]
pub struct VectorDraws {
    pub opaque: Vec<u8>,
    pub transparent: Vec<u8>,
    pub masks: Vec<Vec<u8>>,
    /// Per clip scope, NO_CLIP first.
    pub shadows: Vec<(i32, Vec<u8>)>,
}

pub struct VectorPacker {
    pub objects: Arena,
    pub instances: Arena,
    pub fill_vertices: Arena,
    pub fill_colors: Arena,
    pub stroke_points: Arena,
    pub stroke_curves: Arena,
    slots: HashMap<(u64, i32), Slot>,
    free_objects: Vec<usize>,
    pub paths: HashMap<u64, PathCache>,
}

impl VectorPacker {
    pub fn new(py: Python<'_>) -> VectorPacker {
        VectorPacker {
            objects: Arena::new(py, size_of::<Object>()),
            instances: Arena::new(py, size_of::<Instance>()),
            fill_vertices: Arena::new(py, records::FILL_VERTEX_SIZE),
            fill_colors: Arena::new(py, records::FILL_COLOR_SIZE),
            stroke_points: Arena::new(py, records::STROKE_POINT_SIZE),
            stroke_curves: Arena::new(py, size_of::<StrokeCurve>()),
            slots: HashMap::new(),
            free_objects: Vec::new(),
            paths: HashMap::new(),
        }
    }

    pub fn arenas_mut(&mut self) -> [(&'static str, &mut Arena); 6] {
        [
            ("objects", &mut self.objects),
            ("instances", &mut self.instances),
            ("fill_vertices", &mut self.fill_vertices),
            ("fill_colors", &mut self.fill_colors),
            ("stroke_points", &mut self.stroke_points),
            ("stroke_curves", &mut self.stroke_curves),
        ]
    }

    fn too_much_garbage(&mut self) -> bool {
        self.arenas_mut()
            .iter()
            .any(|(_, arena)| arena.too_much_garbage(4096))
    }

    /// Update the records for `scene` and return the frame's draws
    /// (`RetainedVectors.build`).
    #[allow(clippy::too_many_arguments)]
    pub fn pack(
        &mut self,
        py: Python<'_>,
        graph: &SceneGraph,
        values: &Values,
        scene: &FlatScene,
        camera_position: [f64; 3],
        callbacks: &Callbacks,
        generation: u64,
    ) -> PyResult<VectorDraws> {
        if self.too_much_garbage() {
            // Repack everything from scratch (keeps the per-path cache).
            let paths = std::mem::take(&mut self.paths);
            *self = VectorPacker::new(py);
            self.paths = paths;
        }
        let mut draws = Vec::new();
        let mut seen = HashSet::new();
        let context = PathContext {
            py,
            graph,
            values,
            callbacks,
            camera_position,
            generation,
        };
        for item in &scene.items {
            self.add(&context, item, NO_CLIP, false, &mut draws, &mut seen)?;
        }
        for (index, scope) in scene.clips.iter().enumerate() {
            let NodeData::ClippingGroup { one_sided, .. } = graph.nodes[scope.node].data else {
                unreachable!()
            };
            let front_only = values.truthy(one_sided)?;
            for item in &scope.mask {
                self.add(
                    &context,
                    item,
                    index as i32,
                    front_only,
                    &mut draws,
                    &mut seen,
                )?;
            }
        }
        let gone: Vec<(u64, i32)> = self
            .slots
            .keys()
            .filter(|key| !seen.contains(*key))
            .copied()
            .collect();
        for key in gone {
            let slot = self.slots.remove(&key).unwrap();
            self.free(&slot);
        }
        Ok(batch(scene, draws))
    }

    fn add(
        &mut self,
        context: &PathContext,
        item: &DrawItem,
        mask_of: i32,
        front_only: bool,
        draws: &mut Vec<Draw>,
        seen: &mut HashSet<(u64, i32)>,
    ) -> PyResult<()> {
        let graph: &SceneGraph = context.graph;
        let node = &graph.nodes[item.node];
        let NodeData::Path(fields) = &node.data else {
            return Ok(());
        };
        let points = context.values.get(fields.points).array()?.clone();
        if points.rows() / 4 == 0 {
            return Ok(());
        }
        let key = (node.uid, mask_of);
        seen.insert(key);
        if !self.slots.contains_key(&key) {
            let object = self.object_slot(context.py);
            self.slots.insert(key, Slot::new(object));
        }
        let points_version = context.values.state.versions[fields.points];
        let mut slot = self.slots.remove(&key).unwrap();
        if slot.points_version != points_version {
            // New geometry: every range built from the points moves.
            slot.fill_version = None;
            slot.stroke_key = None;
            slot.colors_key = None;
            slot.points_version = points_version;
        }

        let layer = 2 * item.order;
        let is_shape2d = fields.fill.is_some();
        let flags = (if is_shape2d {
            0
        } else {
            records::FLAG_CAMERA_FACING
        }) | (if front_only {
            records::FLAG_FRONT_ONLY
        } else {
            0
        });
        let placement = (layer, item.clip, flags, item.alpha);
        if slot.world != Some(item.world) || slot.placement != placement {
            let entry: &mut Object = self.objects.record_mut(context.py, slot.object);
            entry.model = records::gpu_matrix(&item.world);
            entry.layer = layer as f32;
            entry.clip = item.clip;
            entry.flags = flags;
            entry.alpha = item.alpha as f32;
            slot.world = Some(item.world);
            slot.placement = placement;
        }
        if item.alpha <= 0.0 {
            self.slots.insert(key, slot);
            return Ok(()); // invisible: the slot stays, nothing is drawn
        }

        let cache = self.paths.entry(node.uid).or_default();
        cache.last_used = context.generation;
        let center = context.center(cache, fields, points_version)?;
        let world_center = item.world.transform_point3(DVec3::from(center));
        let distance = world_center.distance(DVec3::from(context.camera_position));
        let clipped = item.clip != NO_CLIP;
        let casts_shadow = context.values.truthy(fields.cast_shadows)? && mask_of == NO_CLIP;
        let fill = self.fill(context, node, fields, &mut slot, distance, layer)?;
        let stroke = self.stroke(
            context,
            node,
            fields,
            &mut slot,
            distance,
            layer + 1,
            points.rows() / 4,
        )?;
        for mut draw in [fill, stroke].into_iter().flatten() {
            draw.opaque &= !clipped && item.alpha >= 1.0;
            draw.mask_of = mask_of;
            draw.casts_shadow = casts_shadow;
            draw.clip = item.clip;
            draws.push(draw);
        }
        self.slots.insert(key, slot);
        Ok(())
    }

    fn fill(
        &mut self,
        context: &PathContext,
        node: &Node,
        fields: &PathFields,
        slot: &mut Slot,
        distance: f64,
        layer: usize,
    ) -> PyResult<Option<Draw>> {
        let Some((color, opacity)) = fields.fill else {
            return Ok(None);
        };
        let points_version = slot.points_version;
        let cache = self.paths.get_mut(&node.uid).unwrap();
        let colors_key = vec![
            points_version,
            context.values.state.versions[color],
            context.values.state.versions[opacity],
        ];
        let colors = context.fill_colors(cache, fields, color, opacity, &colors_key)?;
        let (visible, opaque) = (colors.2, colors.3);
        if !visible {
            return Ok(None);
        }
        let vertices = context.fill_vertices(cache, fields, points_version)?;
        if vertices.is_empty() {
            return Ok(None);
        }
        if slot.fill_version != Some(points_version) {
            slot.fill_first =
                self.fill_vertices
                    .place(context.py, vertices, slot.fill_first, slot.fill_count);
            slot.fill_count = vertices.len() / records::FILL_VERTEX_SIZE;
            slot.fill_version = Some(points_version);
        }
        if slot.colors_key.as_ref() != Some(&colors_key) {
            let colors = &cache.fill_colors.as_ref().unwrap().1;
            let first =
                self.fill_colors
                    .place(context.py, colors, slot.colors_first, slot.colors_count);
            if first != slot.colors_first || slot.colors_count == 0 {
                let entry: &mut Object = self.objects.record_mut(context.py, slot.object);
                entry.fill_colors = first as u32;
            }
            slot.colors_first = first;
            slot.colors_count = colors.len() / records::FILL_COLOR_SIZE;
            slot.colors_key = Some(colors_key);
        }
        if slot.fill_instance.is_none() {
            let instance = self.instances.allocate(context.py, 1);
            let record = Instance {
                kind: records::KIND_FILL,
                object: slot.object as u32,
                curve: 0,
                _pad: 0,
            };
            self.instances
                .write(context.py, instance, bytemuck::bytes_of(&record));
            slot.fill_instance = Some(instance);
        }
        Ok(Some(Draw {
            args: DrawIndirect {
                vertex_count: slot.fill_count as u32,
                instance_count: 1,
                first_vertex: slot.fill_first as u32,
                first_instance: slot.fill_instance.unwrap() as u32,
            },
            opaque,
            camera_distance: distance,
            layer,
            mask_of: NO_CLIP,
            casts_shadow: false,
            clip: NO_CLIP,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn stroke(
        &mut self,
        context: &PathContext,
        node: &Node,
        fields: &PathFields,
        slot: &mut Slot,
        distance: f64,
        layer: usize,
        curve_count: usize,
    ) -> PyResult<Option<Draw>> {
        let versions = &context.values.state.versions;
        let stroke_key = vec![
            slot.points_version,
            versions[fields.stroke_color],
            versions[fields.stroke_opacity],
            versions[fields.stroke_width],
            versions[fields.stroke_joint],
            versions[fields.stroke_cap],
        ];
        let cache = self.paths.get_mut(&node.uid).unwrap();
        let stroke = context.stroke(cache, fields, &stroke_key)?;
        if !stroke.visible {
            return Ok(None);
        }
        let opaque = stroke.opaque;
        if slot.stroke_key.as_ref() != Some(&stroke_key) {
            slot.points_first = self.stroke_points.place(
                context.py,
                &stroke.points,
                slot.points_first,
                slot.points_count,
            );
            slot.points_count = stroke.points.len() / records::STROKE_POINT_SIZE;
            let mut curves = stroke.curves.clone();
            offset_curves(&mut curves, slot.points_first);
            let count = curves.len() / size_of::<StrokeCurve>();
            let first =
                self.stroke_curves
                    .place(context.py, &curves, slot.curves_first, slot.curves_count);
            if first != slot.curves_first
                || count != slot.curves_count
                || slot.stroke_instances.is_none()
            {
                let rows: Vec<Instance> = (0..count)
                    .map(|curve| Instance {
                        kind: records::KIND_STROKE,
                        object: slot.object as u32,
                        curve: (first + curve) as u32,
                        _pad: 0,
                    })
                    .collect();
                let rows: &[u8] = bytemuck::cast_slice(&rows);
                let old = if slot.stroke_instances.is_some() {
                    slot.curves_count
                } else {
                    0
                };
                slot.stroke_instances = Some(self.instances.place(
                    context.py,
                    rows,
                    slot.stroke_instances.unwrap_or(0),
                    old,
                ));
            }
            slot.curves_first = first;
            slot.curves_count = count;
            slot.stroke_key = Some(stroke_key);
        }
        Ok(Some(Draw {
            args: DrawIndirect {
                vertex_count: records::STROKE_VERTEX_COUNT,
                instance_count: curve_count as u32,
                first_vertex: 0,
                first_instance: slot.stroke_instances.unwrap() as u32,
            },
            opaque,
            camera_distance: distance,
            layer,
            mask_of: NO_CLIP,
            casts_shadow: false,
            clip: NO_CLIP,
        }))
    }

    fn object_slot(&mut self, py: Python<'_>) -> usize {
        self.free_objects
            .pop()
            .unwrap_or_else(|| self.objects.allocate(py, 1))
    }

    fn free(&mut self, slot: &Slot) {
        self.free_objects.push(slot.object);
        self.fill_vertices.garbage += slot.fill_count;
        self.stroke_points.garbage += slot.points_count;
        self.stroke_curves.garbage += slot.curves_count;
        self.instances.garbage += slot.curves_count + usize::from(slot.fill_instance.is_some());
        self.fill_colors.garbage += slot.colors_count;
    }
}

/// Shift a stroke's local point indices to where its points were placed.
fn offset_curves(curves: &mut [u8], first_point: usize) {
    let curves: &mut [StrokeCurve] = bytemuck::cast_slice_mut(curves);
    for curve in curves {
        curve.first_point += first_point as u32;
        for link in [&mut curve.prev_point, &mut curve.next_point] {
            if *link != records::NO_NEIGHBOR {
                *link += first_point as i32;
            }
        }
    }
}

/// `RetainedVectors._batch`: opaque draws, transparent ones back to front,
/// mask draws per scope in painter's order, shadow casters per scope.
fn batch(scene: &FlatScene, draws: Vec<Draw>) -> VectorDraws {
    let frame: Vec<&Draw> = draws.iter().filter(|d| d.mask_of == NO_CLIP).collect();
    let opaque: Vec<DrawIndirect> = frame.iter().filter(|d| d.opaque).map(|d| d.args).collect();
    let mut transparent: Vec<&&Draw> = frame.iter().filter(|d| !d.opaque).collect();
    // Farthest first; coplanar shapes in painter's order (distances
    // rounded to 6 decimals, as the reference sorts them).
    transparent.sort_by_key(|d| (-((d.camera_distance * 1e6).round() as i64), d.layer));
    let masks = (0..scene.clips.len() as i32)
        .map(|scope| {
            let mut mask: Vec<&Draw> = draws.iter().filter(|d| d.mask_of == scope).collect();
            mask.sort_by_key(|d| d.layer);
            draw_bytes(mask.iter().map(|d| d.args))
        })
        .collect();
    let shadows = std::iter::once(NO_CLIP)
        .chain(0..scene.clips.len() as i32)
        .map(|scope| {
            let casters = frame
                .iter()
                .filter(|d| d.casts_shadow && scene.in_scope(d.clip, scope));
            (scope, draw_bytes(casters.map(|d| d.args)))
        })
        .collect();
    VectorDraws {
        opaque: draw_bytes(opaque),
        transparent: draw_bytes(transparent.iter().map(|d| d.args)),
        masks,
        shadows,
    }
}

fn draw_bytes(draws: impl IntoIterator<Item = DrawIndirect>) -> Vec<u8> {
    let draws: Vec<DrawIndirect> = draws.into_iter().collect();
    bytemuck::cast_slice(&draws).to_vec()
}

/// What building one path's records needs, besides the packer itself.
struct PathContext<'a> {
    py: Python<'a>,
    graph: &'a SceneGraph,
    values: &'a Values<'a>,
    callbacks: &'a Callbacks,
    camera_position: [f64; 3],
    generation: u64,
}

impl PathContext<'_> {
    fn python(&self, slot: usize) -> PyResult<Py<PyAny>> {
        self.values.state.kinds[slot].value_to_py(self.py, self.values.get(slot))
    }

    fn center(
        &self,
        cache: &mut PathCache,
        fields: &PathFields,
        version: u64,
    ) -> PyResult<[f64; 3]> {
        if let Some((cached, center)) = cache.center {
            if cached == version {
                return Ok(center);
            }
        }
        let center: (f64, f64, f64) = self
            .callbacks
            .path_center
            .bind(self.py)
            .call1((self.python(fields.points)?,))?
            .extract()?;
        let center = [center.0, center.1, center.2];
        cache.center = Some((version, center));
        Ok(center)
    }

    fn fill_vertices<'c>(
        &self,
        cache: &'c mut PathCache,
        fields: &PathFields,
        version: u64,
    ) -> PyResult<&'c Vec<u8>> {
        if cache.fill_vertices.as_ref().map(|entry| entry.0) != Some(version) {
            let bytes = bytes_of(
                &self
                    .callbacks
                    .fill_vertices
                    .bind(self.py)
                    .call1((self.python(fields.points)?,))?,
            )?;
            cache.fill_vertices = Some((version, bytes));
        }
        Ok(&cache.fill_vertices.as_ref().unwrap().1)
    }

    fn fill_colors<'c>(
        &self,
        cache: &'c mut PathCache,
        fields: &PathFields,
        color: usize,
        opacity: usize,
        key: &[u64],
    ) -> PyResult<&'c (Vec<u64>, Vec<u8>, bool, bool)> {
        if cache.fill_colors.as_ref().map(|entry| entry.0.as_slice()) != Some(key) {
            let count = self.values.get(fields.points).array()?.rows();
            let result = self.callbacks.fill_colors.bind(self.py).call1((
                self.python(color)?,
                self.python(opacity)?,
                count,
            ))?;
            let bytes = bytes_of(&result.get_item(0)?)?;
            let (visible, opaque): (bool, bool) = (
                result.get_item(1)?.extract()?,
                result.get_item(2)?.extract()?,
            );
            cache.fill_colors = Some((key.to_vec(), bytes, visible, opaque));
        }
        Ok(cache.fill_colors.as_ref().unwrap())
    }

    fn stroke<'c>(
        &self,
        cache: &'c mut PathCache,
        fields: &PathFields,
        key: &[u64],
    ) -> PyResult<&'c StrokeRecords> {
        if cache.stroke.as_ref().map(|entry| entry.0.as_slice()) != Some(key) {
            let arguments = (
                self.python(fields.points)?,
                self.python(fields.stroke_color)?,
                self.python(fields.stroke_opacity)?,
                self.python(fields.stroke_width)?,
                self.python(fields.stroke_joint)?,
                self.python(fields.stroke_cap)?,
            );
            let result = self
                .callbacks
                .stroke_records
                .bind(self.py)
                .call1(arguments)?;
            let records = StrokeRecords {
                curves: bytes_of(&result.get_item(0)?)?,
                points: bytes_of(&result.get_item(1)?)?,
                visible: result.get_item(2)?.extract()?,
                opaque: result.get_item(3)?.extract()?,
            };
            cache.stroke = Some((key.to_vec(), records));
        }
        Ok(&cache.stroke.as_ref().unwrap().1)
    }
}
