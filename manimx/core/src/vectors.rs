//! The retained vector records of `Shape2D` and `Curve3D` nodes, in the
//! buffers vector.wgsl reads:
//!
//! ```text
//! objects        per node: model matrix, depth layer, clip, flags, alpha
//!                (multiplied into every color on the GPU, so a fade
//!                rewrites this entry only), camera view layer and size
//! instances      what each draw instance makes (a fill or a stroke curve)
//! fill_vertices  every fill's triangles (geometry only)
//! fill_colors    per-point fill RGBA; vertices look their color up in it
//! stroke_points  control points with per-point stroke width and color
//! stroke_curves  one entry per cubic curve, linked to its neighbours
//!                for joints
//! ```
//!
//! From one frame to the next usually only a few transforms change, so the
//! records stay in arenas that mirror the GPU buffers, with one slot per
//! node (its uid, and whether it is drawn as a mask). Each frame compares
//! the scene with the slots by signal versions:
//!
//! - same node, same placement: nothing is written;
//! - new world matrix, layer, clip or alpha: its object entry;
//! - new fill colors or stroke style: its color or point records, in place;
//! - new points: its records go to a new range (the old one becomes
//!   garbage), and the records pointing at them are written again.
//!
//! Only the written ranges are uploaded. The draw lists are built again
//! every frame from the slots: an opaque list (any order, the depth test
//! resolves overlaps), a transparent list sorted back to front, and one
//! list per clip scope drawing its mask. Clipped content is always
//! transparent, since its mask may let only part of it through. Nodes
//! that leave the scene free their slots; when garbage outweighs live
//! records, everything is packed again from scratch.
//!
//! What depends on geometry or style alone (a fill's triangles, a stroke's
//! curve links, per-point colors) is built by Python once per version of
//! the signals it comes from (manimx/renderer/core_callbacks.py) and kept
//! in `PathCache`.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use crate::arena::Arena;
use crate::callbacks::{bytes_of, Callbacks};
use crate::graph::{
    DrawItem, FlatScene, Node, NodeData, PathFields, SceneGraph, ShadowList, Values, NO_CLIP,
};
use crate::records::{self, DrawIndirect, Instance, Object, StrokeCurve};
use crate::strokes::{stroke_records, StrokeStyle};
use crate::tessellation::tessellate_fill;
use crate::value::Value;
use glam::{DMat4, DVec2, DVec3};

/// Where one path's records live, and what they were built from (`_Slot`).
struct Slot {
    object: usize,
    world: Option<DMat4>,
    placement: (usize, i32, u32, f64, CameraView),
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
            placement: (usize::MAX, NO_CLIP - 1, u32::MAX, -1.0, NOT_A_VIEW),
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
/// built from.
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
    /// Depth of the path's center along the view direction: paths in one
    /// plane facing the camera tie, so their layers order them.
    pub camera_distance: f64,
    pub layer: usize,
    pub mask_of: i32,
    pub casts_shadow: bool,
    pub clip: i32,
}

/// The draw lists of one frame (the records are in the arenas).
#[derive(Default)]
pub struct VectorDraws {
    pub opaque: Vec<u8>,
    pub transparent: Vec<u8>,
    pub masks: Vec<Vec<u8>>,
    /// Per light scope (NO_CLIP first), the casters sorted by clip scope
    /// and their runs (`FlatScene::shadow_lists`).
    pub shadows: Vec<ShadowList<u8>>,
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
    pub fn new() -> VectorPacker {
        VectorPacker {
            objects: Arena::new(size_of::<Object>()),
            instances: Arena::new(size_of::<Instance>()),
            fill_vertices: Arena::new(records::FILL_VERTEX_SIZE),
            fill_colors: Arena::new(records::FILL_COLOR_SIZE),
            stroke_points: Arena::new(records::STROKE_POINT_SIZE),
            stroke_curves: Arena::new(size_of::<StrokeCurve>()),
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
    #[allow(clippy::too_many_arguments)]
    pub fn pack(
        &mut self,
        py: Python<'_>,
        graph: &SceneGraph,
        values: &Values,
        scene: &FlatScene,
        view: DMat4,
        camera_views: &CameraViews,
        callbacks: &Callbacks,
        generation: u64,
    ) -> PyResult<VectorDraws> {
        if self.too_much_garbage() {
            // Repack everything from scratch (keeps the per-path cache).
            let paths = std::mem::take(&mut self.paths);
            *self = VectorPacker::new();
            self.paths = paths;
        }
        let mut draws = Vec::new();
        let mut seen = HashSet::new();
        let context = PathContext {
            py,
            graph,
            values,
            callbacks,
            view,
            camera_views,
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
            let object = self.object_slot();
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
        let view = context
            .camera_views
            .get(&node.uid)
            .copied()
            .unwrap_or(NOT_A_VIEW);
        let placement = (layer, item.clip, flags, item.alpha, view);
        if slot.world != Some(item.world) || slot.placement != placement {
            let entry: &mut Object = self.objects.record_mut(slot.object);
            entry.model = records::gpu_matrix(&item.world);
            entry.layer = layer as f32;
            entry.clip = item.clip;
            entry.flags = flags;
            entry.alpha = item.alpha as f32;
            (entry.view, entry.view_scale) = view;
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
        let distance = -context.view.transform_point3(world_center).z;
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
        // A picture may be see-through anywhere (a transparent background).
        let (visible, opaque) = (
            colors.2,
            colors.3 && !context.camera_views.contains_key(&node.uid),
        );
        if !visible {
            return Ok(None);
        }
        let vertices = context.fill_vertices(cache, fields, points_version)?;
        if vertices.is_empty() {
            return Ok(None);
        }
        if slot.fill_version != Some(points_version) {
            slot.fill_first = self
                .fill_vertices
                .place(vertices, slot.fill_first, slot.fill_count);
            slot.fill_count = vertices.len() / records::FILL_VERTEX_SIZE;
            slot.fill_version = Some(points_version);
        }
        if slot.colors_key.as_ref() != Some(&colors_key) {
            let colors = &cache.fill_colors.as_ref().unwrap().1;
            let first = self
                .fill_colors
                .place(colors, slot.colors_first, slot.colors_count);
            if first != slot.colors_first || slot.colors_count == 0 {
                let entry: &mut Object = self.objects.record_mut(slot.object);
                entry.fill_colors = first as u32;
            }
            slot.colors_first = first;
            slot.colors_count = colors.len() / records::FILL_COLOR_SIZE;
            slot.colors_key = Some(colors_key);
        }
        if slot.fill_instance.is_none() {
            let instance = self.instances.allocate(1);
            let record = Instance {
                kind: records::KIND_FILL,
                object: slot.object as u32,
                curve: 0,
                _pad: 0,
            };
            self.instances.write(instance, bytemuck::bytes_of(&record));
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
            slot.points_first =
                self.stroke_points
                    .place(&stroke.points, slot.points_first, slot.points_count);
            slot.points_count = stroke.points.len() / records::STROKE_POINT_SIZE;
            let mut curves = stroke.curves.clone();
            offset_curves(&mut curves, slot.points_first);
            let count = curves.len() / size_of::<StrokeCurve>();
            let first = self
                .stroke_curves
                .place(&curves, slot.curves_first, slot.curves_count);
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

    fn object_slot(&mut self) -> usize {
        self.free_objects
            .pop()
            .unwrap_or_else(|| self.objects.allocate(1))
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

/// The draw lists: opaque draws, transparent ones back to front,
/// mask draws per scope in painter's order, shadow casters per scope.
fn batch(scene: &FlatScene, draws: Vec<Draw>) -> VectorDraws {
    let frame: Vec<&Draw> = draws.iter().filter(|d| d.mask_of == NO_CLIP).collect();
    let opaque: Vec<DrawIndirect> = frame.iter().filter(|d| d.opaque).map(|d| d.args).collect();
    let mut transparent: Vec<&&Draw> = frame.iter().filter(|d| !d.opaque).collect();
    // Farthest first; coplanar shapes in painter's order (distances
    // rounded to 6 decimals, so equal depths keep their tree order).
    transparent.sort_by_key(|d| (-((d.camera_distance * 1e6).round() as i64), d.layer));
    let masks = (0..scene.clips.len() as i32)
        .map(|scope| {
            let mut mask: Vec<&Draw> = draws.iter().filter(|d| d.mask_of == scope).collect();
            mask.sort_by_key(|d| d.layer);
            draw_bytes(mask.iter().map(|d| d.args))
        })
        .collect();
    let casters: Vec<(DrawIndirect, i32)> = frame
        .iter()
        .filter(|d| d.casts_shadow)
        .map(|d| (d.args, d.clip))
        .collect();
    let shadows = scene
        .shadow_lists(&casters)
        .into_iter()
        .map(|(scope, draws, runs)| (scope, draw_bytes(draws), runs))
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

/// A camera view's layer and `1 / (width, height)` of its plane.
pub type CameraView = (i32, [f32; 2]);

/// The camera views of a frame by node uid.
pub type CameraViews = HashMap<u64, CameraView>;

const NOT_A_VIEW: CameraView = (records::NO_VIEW, [0.0, 0.0]);

/// What building one path's records needs, besides the packer itself.
struct PathContext<'a> {
    py: Python<'a>,
    graph: &'a SceneGraph,
    values: &'a Values<'a>,
    callbacks: &'a Callbacks,
    /// World to view space: transparent draws sort by depth along the
    /// view direction.
    view: DMat4,
    camera_views: &'a CameraViews,
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
            let points = self.values.get(fields.points).array()?;
            let corners: Vec<DVec2> = points
                .data
                .chunks_exact(points.cols())
                .map(|row| DVec2::new(row[0], row[1]))
                .collect();
            let bytes = bytemuck::cast_slice(&tessellate_fill(&corners)).to_vec();
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

    /// The stroke records built in Rust (strokes.rs), or None for style
    /// values it does not read (then `stroke_records` in Python builds
    /// them).
    fn native_stroke(&self, fields: &PathFields) -> PyResult<Option<StrokeRecords>> {
        let (
            Value::Array(points),
            Value::Array(colors),
            Value::Array(opacities),
            Value::Array(widths),
        ) = (
            self.values.get(fields.points),
            self.values.get(fields.stroke_color),
            self.values.get(fields.stroke_opacity),
            self.values.get(fields.stroke_width),
        )
        else {
            return Ok(None);
        };
        let dimension = points.cols();
        let rgb = colors.shape.last() == Some(&3) && colors.shape.len() <= 2;
        if points.shape.len() != 2
            || !(dimension == 2 || dimension == 3)
            || !rgb
            || colors.data.is_empty()
        {
            return Ok(None);
        }
        if opacities.data.is_empty() || widths.data.is_empty() || points.rows() % 4 != 0 {
            return Ok(None);
        }
        let (Value::Object(joint), Value::Object(cap)) = (
            self.values.get(fields.stroke_joint),
            self.values.get(fields.stroke_cap),
        ) else {
            return Ok(None);
        };
        let joint = joint.bind(self.py);
        let joints: Vec<u32> = if joint.hasattr("value")? {
            vec![joint.getattr("value")?.extract()?]
        } else {
            joint
                .try_iter()?
                .map(|item| item?.getattr("value")?.extract())
                .collect::<PyResult<_>>()?
        };
        if joints.is_empty() {
            return Ok(None);
        }
        let cap: u32 = cap.bind(self.py).getattr("value")?.extract()?;
        let colors: Vec<[f64; 3]> = colors
            .data
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let style = StrokeStyle {
            colors: &colors,
            opacities: &opacities.data,
            widths: &widths.data,
            joints: &joints,
            cap,
        };
        let stroke = stroke_records(&points.data, dimension, &style);
        Ok(Some(StrokeRecords {
            curves: stroke.curves,
            points: stroke.points,
            visible: stroke.visible,
            opaque: stroke.opaque,
        }))
    }

    fn stroke<'c>(
        &self,
        cache: &'c mut PathCache,
        fields: &PathFields,
        key: &[u64],
    ) -> PyResult<&'c StrokeRecords> {
        if cache.stroke.as_ref().map(|entry| entry.0.as_slice()) != Some(key) {
            if let Some(records) = self.native_stroke(fields)? {
                cache.stroke = Some((key.to_vec(), records));
                return Ok(&cache.stroke.as_ref().unwrap().1);
            }
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
