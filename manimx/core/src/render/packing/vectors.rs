//! The retained vector records of `RShape2D` and `RCurve3D` nodes, in the
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
//! the tree with the slots by uid and by the versions of the values:
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
//! What depends on geometry or style alone (a fill's triangles and
//! per-point colors, a stroke's curve links and points, a path's center)
//! is built once per version of the values it comes from and kept in
//! `PathCache`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::{DMat4, DVec2, DVec3};

use crate::render::geometry::strokes::{stroke_records, StrokeStyle};
use crate::render::geometry::tessellation::tessellate_fill;
use crate::render::packing::arena::Arena;
use crate::render::packing::flatten::{DrawItem, FlatScene, ShadowList, NO_CLIP};
use crate::render::packing::records::{self, DrawIndirect, Instance, Object, StrokeCurve};
use crate::robjects::{Fill, Kind, Path, Points, Texture};

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

/// Records built for one path, each with the versions it was built from.
#[derive(Default)]
pub struct PathCache {
    center: Option<(u64, DVec3)>,
    fill_vertices: Option<(u64, Vec<u8>)>,
    fill_colors: Option<(Vec<u64>, FillColors)>,
    stroke: Option<(Vec<u64>, StrokeRecords)>,
    pub last_used: u64,
}

struct StrokeRecords {
    curves: Vec<u8>,
    points: Vec<u8>,
    visible: bool,
    opaque: bool,
}

/// A fill's per-point RGBA records, whether any of it shows and whether
/// all of it is opaque.
struct FillColors {
    records: Vec<u8>,
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
    /// The images fills show, by layer of the image texture.
    pub images: Vec<Arc<Texture>>,
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

    /// Update the records for the flattened `scene` and return the
    /// frame's draws.
    pub fn pack(
        &mut self,
        flat: &FlatScene,
        view: DMat4,
        camera_views: &CameraViews,
        generation: u64,
    ) -> VectorDraws {
        if self.too_much_garbage() {
            // Repack everything from scratch (keeps the per-path cache).
            let paths = std::mem::take(&mut self.paths);
            *self = VectorPacker::new();
            self.paths = paths;
        }
        let mut draws = Vec::new();
        let mut seen = HashSet::new();
        let mut images = Vec::new();
        let context = PathContext {
            view,
            camera_views,
            generation,
        };
        for item in &flat.items {
            self.add(
                &context,
                item,
                NO_CLIP,
                false,
                &mut draws,
                &mut seen,
                &mut images,
            );
        }
        for (index, scope) in flat.clips.iter().enumerate() {
            for item in &scope.mask {
                self.add(
                    &context,
                    item,
                    index as i32,
                    scope.one_sided,
                    &mut draws,
                    &mut seen,
                    &mut images,
                );
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
        VectorDraws {
            images,
            ..batch(flat, draws)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        context: &PathContext,
        item: &DrawItem,
        mask_of: i32,
        front_only: bool,
        draws: &mut Vec<Draw>,
        seen: &mut HashSet<(u64, i32)>,
        images: &mut Vec<Arc<Texture>>,
    ) {
        let node = item.node;
        let (path, fill) = match &node.kind {
            Kind::Shape2D { path, fill } => (path, Some(fill.as_ref())),
            Kind::Curve3D { path } => (path, None),
            _ => return,
        };
        let curve_count = path.points.count() / 4;
        if curve_count == 0 {
            return;
        }
        let key = (node.uid, mask_of);
        seen.insert(key);
        if !self.slots.contains_key(&key) {
            let object = self.object_slot();
            self.slots.insert(key, Slot::new(object));
        }
        let points_version = path.points.version;
        let mut slot = self.slots.remove(&key).unwrap();
        if slot.points_version != points_version {
            // New geometry: every range built from the points moves.
            slot.fill_version = None;
            slot.stroke_key = None;
            slot.colors_key = None;
            slot.points_version = points_version;
        }

        let layer = 2 * item.order;
        let is_shape2d = fill.is_some();
        let flags = (if is_shape2d {
            0
        } else {
            records::FLAG_CAMERA_FACING
        }) | (if front_only {
            records::FLAG_FRONT_ONLY
        } else {
            0
        });
        let image = fill.and_then(|fill| fill.image.as_ref());
        let view = match (context.camera_views.get(&node.uid), image) {
            (Some(view), _) => *view,
            (None, Some(image)) => image_view(images, image, &path.points),
            (None, None) => NOT_A_VIEW,
        };
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
            return; // invisible: the slot stays, nothing is drawn
        }

        let cache = self.paths.entry(node.uid).or_default();
        cache.last_used = context.generation;
        let center = path_center(cache, &path.points);
        let world_center = item.world.transform_point3(center);
        let distance = -context.view.transform_point3(world_center).z;
        let clipped = item.clip != NO_CLIP;
        let casts_shadow = path.cast_shadows && mask_of == NO_CLIP;
        let fill = fill.and_then(|fill| {
            self.fill(
                context,
                node.uid,
                &path.points,
                fill,
                &mut slot,
                distance,
                layer,
            )
        });
        let stroke = self.stroke(node.uid, path, &mut slot, distance, layer + 1, curve_count);
        for mut draw in [fill, stroke].into_iter().flatten() {
            draw.opaque &= !clipped && item.alpha >= 1.0;
            draw.mask_of = mask_of;
            draw.casts_shadow = casts_shadow;
            draw.clip = item.clip;
            draws.push(draw);
        }
        self.slots.insert(key, slot);
    }

    #[allow(clippy::too_many_arguments)]
    fn fill(
        &mut self,
        context: &PathContext,
        uid: u64,
        points: &Points,
        fill: &Fill,
        slot: &mut Slot,
        distance: f64,
        layer: usize,
    ) -> Option<Draw> {
        let points_version = slot.points_version;
        let cache = self.paths.get_mut(&uid).unwrap();
        let colors_key = vec![points_version, fill.version];
        let colors = fill_colors(cache, points, fill, &colors_key);
        // A picture or an image may be see-through anywhere.
        let (visible, opaque) = (
            colors.visible,
            colors.opaque && !context.camera_views.contains_key(&uid) && fill.image.is_none(),
        );
        if !visible {
            return None;
        }
        let vertices = fill_vertices(cache, points);
        if vertices.is_empty() {
            return None;
        }
        if slot.fill_version != Some(points_version) {
            slot.fill_first = self
                .fill_vertices
                .place(vertices, slot.fill_first, slot.fill_count);
            slot.fill_count = vertices.len() / records::FILL_VERTEX_SIZE;
            slot.fill_version = Some(points_version);
        }
        if slot.colors_key.as_ref() != Some(&colors_key) {
            let colors = &cache.fill_colors.as_ref().unwrap().1.records;
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
        Some(Draw {
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
        })
    }

    fn stroke(
        &mut self,
        uid: u64,
        path: &Path,
        slot: &mut Slot,
        distance: f64,
        layer: usize,
        curve_count: usize,
    ) -> Option<Draw> {
        let stroke_key = vec![slot.points_version, path.stroke.version];
        let cache = self.paths.get_mut(&uid).unwrap();
        let stroke = stroke(cache, path, &stroke_key);
        if !stroke.visible {
            return None;
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
        Some(Draw {
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
        })
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
        images: Vec::new(),
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

/// The view of a fill showing `image`: its layer (added to `images` on its
/// first use in the frame) and `1 / size` of the local box around the
/// origin that the image covers, as wide as the path reaches either way.
fn image_view(images: &mut Vec<Arc<Texture>>, image: &Arc<Texture>, points: &Points) -> CameraView {
    let layer = match images.iter().position(|seen| seen.uid == image.uid) {
        Some(layer) => layer,
        None => {
            images.push(image.clone());
            images.len() - 1
        }
    };
    let reach = (0..points.count())
        .map(|index| points.point(index).truncate().abs())
        .fold(DVec2::ZERO, DVec2::max);
    let scale = |half: f64| if half > 0.0 { (0.5 / half) as f32 } else { 0.0 };
    (records::image_view(layer), [scale(reach.x), scale(reach.y)])
}

/// What building one path's records needs, besides the packer itself.
struct PathContext<'a> {
    /// World to view space: transparent draws sort by depth along the
    /// view direction.
    view: DMat4,
    camera_views: &'a CameraViews,
    generation: u64,
}

/// The mean of a path's points in 3D (its distance to the camera sorts
/// transparent draws).
fn path_center(cache: &mut PathCache, points: &Points) -> DVec3 {
    if let Some((cached, center)) = cache.center {
        if cached == points.version {
            return center;
        }
    }
    let count = points.count();
    let center = (0..count).map(|index| points.point(index)).sum::<DVec3>() / count as f64;
    cache.center = Some((points.version, center));
    center
}

fn fill_vertices<'c>(cache: &'c mut PathCache, points: &Points) -> &'c Vec<u8> {
    if cache.fill_vertices.as_ref().map(|entry| entry.0) != Some(points.version) {
        let corners: Vec<DVec2> = (0..points.count())
            .map(|index| points.point(index).truncate())
            .collect();
        let bytes = bytemuck::cast_slice(&tessellate_fill(&corners)).to_vec();
        cache.fill_vertices = Some((points.version, bytes));
    }
    &cache.fill_vertices.as_ref().unwrap().1
}

/// Per-point fill RGBA (float32): the fill's colors and opacities (clamped
/// to [0, 1]) spread over the points like `resample` spreads them.
fn fill_colors<'c>(
    cache: &'c mut PathCache,
    points: &Points,
    fill: &Fill,
    key: &[u64],
) -> &'c FillColors {
    if cache.fill_colors.as_ref().map(|entry| entry.0.as_slice()) != Some(key) {
        let count = points.count();
        let opacities: Vec<f64> = fill.opacities.iter().map(|o| o.clamp(0.0, 1.0)).collect();
        let channels: Vec<Vec<f64>> = (0..3)
            .map(|channel| {
                let values: Vec<f64> = fill.colors.iter().map(|color| color[channel]).collect();
                crate::render::geometry::strokes::resample(&values, count)
            })
            .collect();
        let alpha = crate::render::geometry::strokes::resample(&opacities, count);
        let records: Vec<[f32; 4]> = (0..count)
            .map(|index| {
                [
                    channels[0][index] as f32,
                    channels[1][index] as f32,
                    channels[2][index] as f32,
                    alpha[index] as f32,
                ]
            })
            .collect();
        let colors = FillColors {
            records: bytemuck::cast_slice(&records).to_vec(),
            visible: opacities.iter().any(|&o| o > 0.0),
            opaque: opacities.iter().all(|&o| o >= 1.0),
        };
        cache.fill_colors = Some((key.to_vec(), colors));
    }
    &cache.fill_colors.as_ref().unwrap().1
}

fn stroke<'c>(cache: &'c mut PathCache, path: &Path, key: &[u64]) -> &'c StrokeRecords {
    if cache.stroke.as_ref().map(|entry| entry.0.as_slice()) != Some(key) {
        let points = &path.points;
        let style = &path.stroke;
        let records = if points.data.is_empty()
            || style.colors.is_empty()
            || style.opacities.is_empty()
            || style.widths.is_empty()
            || style.joints.is_empty()
        {
            StrokeRecords {
                curves: Vec::new(),
                points: Vec::new(),
                visible: false,
                opaque: false,
            }
        } else {
            let stroke = stroke_records(
                &points.data,
                points.dimension,
                &StrokeStyle {
                    colors: &style.colors,
                    opacities: &style.opacities,
                    widths: &style.widths,
                    joints: &style.joints,
                    cap: style.cap,
                },
            );
            StrokeRecords {
                curves: stroke.curves,
                points: stroke.points,
                visible: stroke.visible,
                opaque: stroke.opaque,
            }
        };
        cache.stroke = Some((key.to_vec(), records));
    }
    &cache.stroke.as_ref().unwrap().1
}
