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
//! node (its uid, and whether it is drawn as a mask). The scene's store
//! (store.rs) says which leaves changed; only those are looked at again,
//! the others reuse their draws (`VectorItems`). A changed leaf is
//! compared with its slot by the versions of its values:
//!
//! - same placement: nothing is written;
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
//! `PathMemos` (cache/).
//!
//! Pages: a GPU binds at most `max_storage_buffer_binding_size` bytes of a
//! buffer at once (128 MiB is WebGPU's default, 2 GiB is common), and a
//! large scene (100 000 dice: 3 GB of fill triangles) has more. So the
//! arenas come in pages (`Page`), each drawn with its own bind group, none
//! larger than `budget` bytes per arena. A node's records all live on one
//! page, and every index in them (instance to object, curve to points)
//! counts from its page's start, so a page is drawn exactly as the one set
//! of buffers was before. A new node goes to the first page with room for
//! its records; a node whose new records no longer fit its page moves to
//! one that has room. The draw lists keep their order (transparent draws
//! back to front across pages) and say which page each run of draws is on
//! (`DrawList`). Garbage is repacked per page.
//!
//! Culling: a path whose box (`Points::bounds`, computed once per points,
//! placed by its world matrix and widened by how far its stroke can
//! reach, kept until the leaf changes) lies wholly beyond one side of the
//! view is not drawn (`Culling`). The boxes are kept in a bounding volume
//! hierarchy (bvh.rs), so finding what a view shows costs about what it
//! shows, not what the scene holds. A culled shape keeps the records it
//! has, so a shape moving back into view, or showing in another view this
//! packer packs (a camera view's picture), costs nothing more; a shape
//! never in view gets no records at all. Where the view has meshes,
//! shadow casters and masks are drawn out of view too: a shadow or a cut
//! can fall into it from outside.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::{DMat4, DVec2, DVec3, DVec4};

use crate::cache::Memo;
use crate::flat::{DrawItem, FlatScene, Place, NO_CLIP};
use crate::render::pack::arena::Arena;
use crate::render::pack::bvh::{Bounds, Bvh};
use crate::render::pack::casters::shadow_lists;
use crate::render::pack::shapes::strokes::{resample, stroke_records, StrokeStyle};
use crate::render::pack::shapes::tessellation::tessellate_fill;
use crate::render::pack::store::Changes;
use crate::render::packet::records::{self, DrawIndirect, Instance, Object, StrokeCurve};
use crate::render::packet::{DrawList, VectorDraws};
use crate::robjects::{Fill, Kind, Path, Points, Texture};

/// Where one path's records live, and what they were built from.
struct Slot {
    /// The page holding all of its records.
    page: usize,
    object: usize,
    world: Option<DMat4>,
    placement: (usize, i32, u32, f64, CameraView),
    /// Points version the geometry ranges below were built from.
    points_version: u64,
    fill_version: Option<u64>,
    fill_first: usize,
    fill_count: usize,
    fill_instance: Option<usize>,
    colors_key: Option<[u64; 2]>,
    colors_first: usize,
    colors_count: usize,
    stroke_key: Option<[u64; 2]>,
    points_first: usize,
    points_count: usize,
    curves_first: usize,
    curves_count: usize,
    stroke_instances: Option<usize>,
}

impl Slot {
    fn new(page: usize, object: usize) -> Slot {
        Slot {
            page,
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

/// What was built from each path's values, by node uid: the center and
/// the fill's triangles from its points version, the fill's per-point
/// colors and the stroke's records from `[points version, fill or stroke
/// version]`.
#[derive(Default)]
pub struct PathMemos {
    centers: Memo<u64, u64, DVec3>,
    fill_vertices: Memo<u64, u64, Vec<u8>>,
    fill_colors: Memo<u64, [u64; 2], FillColors>,
    strokes: Memo<u64, [u64; 2], StrokeRecords>,
}

impl PathMemos {
    /// Keep only the entries of the nodes `live` accepts.
    pub fn retain(&mut self, live: impl Fn(&u64) -> bool) {
        self.centers.retain(&live);
        self.fill_vertices.retain(&live);
        self.fill_colors.retain(&live);
        self.strokes.retain(&live);
    }
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

/// One indirect draw before encoding, with what it needs for sorting.
pub struct Draw {
    pub args: DrawIndirect,
    /// The page whose buffers it draws from.
    pub page: usize,
    pub opaque: bool,
    /// Depth of the path's center along the view direction: paths in one
    /// plane facing the camera tie, so their layers order them.
    pub camera_distance: f64,
    pub layer: usize,
    pub mask_of: i32,
    pub casts_shadow: bool,
    pub clip: i32,
}

/// One set of the record arenas, drawn with one bind group (see "Pages").
pub struct Page {
    pub objects: Arena,
    pub instances: Arena,
    pub fill_vertices: Arena,
    pub fill_colors: Arena,
    pub stroke_points: Arena,
    pub stroke_curves: Arena,
    free_objects: Vec<usize>,
}

/// Records per arena, in `Page::arenas` order: objects, instances, fill
/// vertices, fill colors, stroke points, stroke curves.
type Records = [usize; 6];

impl Page {
    fn new() -> Page {
        Page {
            objects: Arena::new(size_of::<Object>()),
            instances: Arena::new(size_of::<Instance>()),
            fill_vertices: Arena::new(records::FILL_VERTEX_SIZE),
            fill_colors: Arena::new(records::FILL_COLOR_SIZE),
            stroke_points: Arena::new(records::STROKE_POINT_SIZE),
            stroke_curves: Arena::new(size_of::<StrokeCurve>()),
            free_objects: Vec::new(),
        }
    }

    pub fn arenas(&self) -> [&Arena; 6] {
        [
            &self.objects,
            &self.instances,
            &self.fill_vertices,
            &self.fill_colors,
            &self.stroke_points,
            &self.stroke_curves,
        ]
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

    /// Whether `need` more records keep every arena within `budget` bytes
    /// (an empty page takes anything: a node larger than a page has no
    /// better place).
    fn fits(&self, need: &Records, budget: usize) -> bool {
        self.objects.used == 0
            || self
                .arenas()
                .iter()
                .zip(need)
                .all(|(arena, &count)| (arena.used + count) * arena.record_size <= budget)
    }

    fn object_slot(&mut self) -> usize {
        self.free_objects
            .pop()
            .unwrap_or_else(|| self.objects.allocate(1))
    }
}

/// Bytes per arena of a page when nothing else is known (the renderer
/// passes its device's limit): WebGPU's default binding size.
pub const DEFAULT_PAGE_BYTES: usize = 128 << 20;

pub struct VectorPacker {
    pub pages: Vec<Page>,
    /// The largest an arena of a page may grow, in bytes.
    budget: usize,
    slots: HashMap<(u64, i32), Slot>,
    pub paths: PathMemos,
    /// Increases when a page is packed again from scratch: the slots of
    /// its nodes are gone, so every scene's draws are out of date.
    epoch: u64,
}

/// What the vector packer keeps per leaf of one scene, in the scene's
/// order: so a leaf that did not change costs nothing unless it shows,
/// and then copying its draws.
#[derive(Default)]
pub struct VectorItems {
    items: Vec<ItemState>,
    /// The items' boxes, to find those a view shows (bvh.rs).
    bvh: Bvh,
    /// Per clip scope, its mask's leaves.
    masks: Vec<Vec<ItemState>>,
    /// The slot keys of every leaf, to free those of leaves that leave.
    keys: Vec<(u64, i32)>,
    /// The packer's epoch the draws are from.
    epoch: u64,
}

impl VectorItems {
    /// Everything out of date, for the leaves of `flat`.
    fn restart(&mut self, flat: &FlatScene, epoch: u64) {
        self.items = flat.items.iter().map(ItemState::new).collect();
        self.bvh = Bvh::new(self.items.iter().map(|state| state.bounds).collect());
        self.masks = flat
            .clips
            .iter()
            .map(|scope| scope.mask.iter().map(ItemState::new).collect())
            .collect();
        self.keys = item_keys(flat).collect();
        self.epoch = epoch;
    }

    /// Add the leaves appended at the ends of the scene's lists (nodes
    /// added to the scene, `Change::Append`): the others keep their places.
    fn grow(&mut self, flat: &FlatScene) {
        for item in &flat.items[self.items.len()..] {
            let state = ItemState::new(item);
            self.bvh.push(state.bounds);
            self.keys.push((item.node.uid, NO_CLIP));
            self.items.push(state);
        }
        self.masks.resize_with(flat.clips.len(), Vec::new);
        for (scope, clip) in flat.clips.iter().enumerate() {
            for item in &clip.mask[self.masks[scope].len()..] {
                self.keys.push((item.node.uid, scope as i32));
                self.masks[scope].push(ItemState::new(item));
            }
        }
    }
}

/// One leaf's box, center and draws, and whether they are out of date.
struct ItemState {
    /// The world matrix or the values changed: the center is computed
    /// again before it is next drawn.
    moved: bool,
    /// The records are written again before it is next drawn.
    stale: bool,
    /// Its box in world space, widened by how far its stroke reaches
    /// (None: no points, or not a path).
    bounds: Option<Bounds>,
    /// Its center in world space (transparent draws sort by it).
    center: DVec3,
    /// Whether it casts shadows (and is not a mask).
    casts_shadow: bool,
    /// Whether its fill shows an image (its draws are made every frame).
    image: bool,
    /// Fill and stroke.
    draws: [Option<Draw>; 2],
}

impl ItemState {
    fn new(item: &DrawItem) -> ItemState {
        ItemState {
            moved: true,
            stale: true,
            bounds: item_box(item),
            center: DVec3::ZERO,
            casts_shadow: false,
            image: false,
            draws: [None, None],
        }
    }

    /// Its draws, at its distance from the view.
    fn push_draws(&self, frame: &VectorFrame, draws: &mut Vec<Draw>) {
        let distance = -frame.view.transform_point3(self.center).z;
        for draw in self.draws.iter().flatten() {
            draws.push(Draw {
                camera_distance: distance,
                ..*draw
            });
        }
    }

    fn changed(&mut self, item: &DrawItem) {
        self.moved = true;
        self.stale = true;
        self.bounds = item_box(item);
    }
}

/// The world box of a path leaf (`world_box`); None for anything else.
fn item_box(item: &DrawItem) -> Option<Bounds> {
    match &item.node.kind {
        Kind::Shape2D { path, .. } | Kind::Curve3D { path } => world_box(path, &item.world),
        _ => None,
    }
}

/// The slot key of every drawable leaf of `flat`.
fn item_keys(flat: &FlatScene) -> impl Iterator<Item = (u64, i32)> + '_ {
    let items = flat.items.iter().map(|item| (item.node.uid, NO_CLIP));
    let masks = flat.clips.iter().enumerate().flat_map(|(scope, clip)| {
        clip.mask
            .iter()
            .map(move |item| (item.node.uid, scope as i32))
    });
    items.chain(masks)
}

/// How one view sees the scene: what packing needs besides the scene.
pub struct VectorFrame<'a> {
    /// World to view space: transparent draws sort by depth along the
    /// view direction.
    pub view: DMat4,
    pub culling: Culling,
    pub camera_views: &'a CameraViews,
    /// Leave the camera views' shapes out (while a picture is rendered).
    pub hide_camera_views: bool,
}

impl VectorPacker {
    pub fn new(budget: usize) -> VectorPacker {
        VectorPacker {
            pages: vec![Page::new()],
            budget,
            slots: HashMap::new(),
            paths: PathMemos::default(),
            epoch: 0,
        }
    }

    /// The first page with room for `need`, a new one if none has.
    fn page_with_room(&mut self, need: &Records) -> usize {
        match self
            .pages
            .iter()
            .position(|page| page.fits(need, self.budget))
        {
            Some(index) => index,
            None => {
                self.pages.push(Page::new());
                self.pages.len() - 1
            }
        }
    }

    /// Update the records of what changed in `flat` (`changes`, taken
    /// from its store), and return the frame's draws of what `culling`
    /// does not leave out. `items` is what this packer keeps per leaf of
    /// the scene.
    pub fn pack(
        &mut self,
        flat: &FlatScene,
        items: &mut VectorItems,
        changes: &Changes,
        frame: &VectorFrame,
    ) -> VectorDraws {
        for index in 0..self.pages.len() {
            if self.pages[index].too_much_garbage() {
                // Pack this page's nodes again from scratch, into whichever
                // pages have room (keeps the path memos).
                self.slots.retain(|_, slot| slot.page != index);
                self.pages[index] = Page::new();
                self.epoch += 1;
            }
        }
        if changes.all {
            self.forget(items, flat);
        }
        if changes.all || items.epoch != self.epoch {
            items.restart(flat, self.epoch);
        }
        items.grow(flat);
        for place in &changes.places {
            match *place {
                Place::Item(index) => {
                    let state = &mut items.items[index];
                    state.changed(&flat.items[index]);
                    items.bvh.set(index, state.bounds);
                }
                Place::Mask(scope, index) => {
                    items.masks[scope][index].changed(&flat.clips[scope].mask[index])
                }
                Place::Light(_) => {}
            }
        }
        items.bvh.refit();
        let mut draws = Vec::new();
        let mut images = Vec::new();
        // The items the view may show; every item where casters matter out
        // of view (see `add`).
        let shown: Vec<usize> = match frame.culling.keep_casters {
            true => (0..flat.items.len()).collect(),
            false => items.bvh.query(&frame.culling.planes),
        };
        for index in shown {
            let state = &mut items.items[index];
            self.add(
                frame,
                &flat.items[index],
                state,
                NO_CLIP,
                false,
                &mut draws,
                &mut images,
            );
        }
        for (index, scope) in flat.clips.iter().enumerate() {
            for (item, state) in scope.mask.iter().zip(&mut items.masks[index]) {
                self.add(
                    frame,
                    item,
                    state,
                    index as i32,
                    scope.one_sided,
                    &mut draws,
                    &mut images,
                );
            }
        }
        VectorDraws {
            images,
            ..batch(flat, draws)
        }
    }

    /// Free the records of the leaves `items` held that `flat` no longer has.
    fn forget(&mut self, items: &VectorItems, flat: &FlatScene) {
        let kept: HashSet<(u64, i32)> = item_keys(flat).collect();
        for key in &items.keys {
            if !kept.contains(key) {
                if let Some(slot) = self.slots.remove(key) {
                    self.free(&slot);
                }
            }
        }
    }

    /// Free the records of every leaf of a scene that is gone.
    pub fn drop_scene(&mut self, items: &VectorItems) {
        for key in &items.keys {
            if let Some(slot) = self.slots.remove(key) {
                self.free(&slot);
            }
        }
    }

    /// Add `item`'s draws to the frame's, if it shows: its records are
    /// written again when it changed (`state.stale`), and its draws reused
    /// otherwise.
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        frame: &VectorFrame,
        item: &DrawItem,
        state: &mut ItemState,
        mask_of: i32,
        front_only: bool,
        draws: &mut Vec<Draw>,
        images: &mut Vec<Arc<Texture>>,
    ) {
        let culling = &frame.culling;
        if !state.stale && !state.moved && !state.image && frame.camera_views.is_empty() {
            // Unchanged: its draws as they were, from its state alone (its
            // node, somewhere else in memory, is not read).
            let matters_outside =
                culling.keep_casters && (state.casts_shadow || mask_of != NO_CLIP);
            if matters_outside || state.bounds.is_some_and(|bounds| culling.shows(bounds)) {
                state.push_draws(frame, draws);
            }
            return;
        }
        let node = &item.node;
        let (path, fill) = match &node.kind {
            Kind::Shape2D { path, fill } => (path, Some(fill.as_ref())),
            Kind::Curve3D { path } => (path, None),
            _ => return,
        };
        if frame.hide_camera_views && frame.camera_views.contains_key(&node.uid) {
            return;
        }
        if state.moved {
            let points = &path.points;
            let center = *self
                .paths
                .centers
                .get_or_build(node.uid, points.version, || path_center(points));
            state.center = item.world.transform_point3(center);
            state.moved = false;
        }
        // Out of view: no draw. A node with records keeps them (it may come
        // back, or show in another view this packer packs); one without
        // gets none. Casters and masks matter out of view where meshes
        // are: a caster's shadow and a mask's cut can fall into it.
        state.casts_shadow = path.cast_shadows && mask_of == NO_CLIP;
        state.image = fill.is_some_and(|fill| fill.image.is_some());
        let matters_outside = culling.keep_casters && (state.casts_shadow || mask_of != NO_CLIP);
        if !matters_outside && !state.bounds.is_some_and(|bounds| culling.shows(bounds)) {
            return;
        }
        // A picture's or an image's layer is the frame's: written every frame.
        // (An empty map is not asked: hashing a key costs on every shape.)
        let volatile = state.image
            || (!frame.camera_views.is_empty() && frame.camera_views.contains_key(&node.uid));
        if state.stale || volatile {
            state.draws = self.records(frame, item, path, fill, mask_of, front_only, images);
            state.stale = false;
        }
        state.push_draws(frame, draws);
    }

    /// Write `item`'s records where they are out of date and return its
    /// draws (fill, stroke).
    #[allow(clippy::too_many_arguments)]
    fn records(
        &mut self,
        frame: &VectorFrame,
        item: &DrawItem,
        path: &Path,
        fill: Option<&Fill>,
        mask_of: i32,
        front_only: bool,
        images: &mut Vec<Arc<Texture>>,
    ) -> [Option<Draw>; 2] {
        let node = &item.node;
        let curve_count = path.points.count() / 4;
        if curve_count == 0 {
            return [None, None];
        }
        let key = (node.uid, mask_of);
        let casts_shadow = path.cast_shadows && mask_of == NO_CLIP;
        if !self.slots.contains_key(&key) {
            let page = self.page_with_room(&[1, 0, 0, 0, 0, 0]);
            let object = self.pages[page].object_slot();
            self.slots.insert(key, Slot::new(page, object));
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
        if item.alpha > 0.0 {
            // The records about to be placed must fit the slot's page.
            let need = self.records_needed(node.uid, path, fill, &slot);
            if !self.pages[slot.page].fits(&need, self.budget) {
                slot = self.move_to_page_with_room(slot, node.uid, path, fill);
            }
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
        let view = match (frame.camera_views.get(&node.uid), image) {
            (Some(view), _) => *view,
            (None, Some(image)) => image_view(images, image, &path.points),
            (None, None) => NOT_A_VIEW,
        };
        let placement = (layer, item.clip, flags, item.alpha, view);
        if slot.world != Some(item.world) || slot.placement != placement {
            let entry: &mut Object = self.pages[slot.page].objects.record_mut(slot.object);
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
            return [None, None]; // invisible: the slot stays, nothing is drawn
        }

        let clipped = item.clip != NO_CLIP;
        let fill =
            fill.and_then(|fill| self.fill(frame, node.uid, &path.points, fill, &mut slot, layer));
        let stroke = self.stroke(node.uid, path, &mut slot, layer + 1, curve_count);
        let mut draws = [fill, stroke];
        for draw in draws.iter_mut().flatten() {
            draw.opaque &= !clipped && item.alpha >= 1.0;
            draw.mask_of = mask_of;
            draw.casts_shadow = casts_shadow;
            draw.clip = item.clip;
            draw.page = slot.page;
        }
        self.slots.insert(key, slot);
        draws
    }

    fn fill(
        &mut self,
        frame: &VectorFrame,
        uid: u64,
        points: &Points,
        fill: &Fill,
        slot: &mut Slot,
        layer: usize,
    ) -> Option<Draw> {
        let points_version = slot.points_version;
        let colors_key = [points_version, fill.version];
        let colors = self
            .paths
            .fill_colors
            .get_or_build(uid, colors_key, || fill_colors(points, fill));
        // A picture or an image may be see-through anywhere.
        let (visible, opaque) = (
            colors.visible,
            colors.opaque && !frame.camera_views.contains_key(&uid) && fill.image.is_none(),
        );
        if !visible {
            return None;
        }
        let vertices = self
            .paths
            .fill_vertices
            .get_or_build(uid, points_version, || fill_vertices(points));
        if vertices.is_empty() {
            return None;
        }
        let page = &mut self.pages[slot.page];
        if slot.fill_version != Some(points_version) {
            slot.fill_first = page
                .fill_vertices
                .place(vertices, slot.fill_first, slot.fill_count);
            slot.fill_count = vertices.len() / records::FILL_VERTEX_SIZE;
            slot.fill_version = Some(points_version);
        }
        if slot.colors_key != Some(colors_key) {
            let colors = &colors.records;
            let first = page
                .fill_colors
                .place(colors, slot.colors_first, slot.colors_count);
            if first != slot.colors_first || slot.colors_count == 0 {
                let entry: &mut Object = page.objects.record_mut(slot.object);
                entry.fill_colors = first as u32;
            }
            slot.colors_first = first;
            slot.colors_count = colors.len() / records::FILL_COLOR_SIZE;
            slot.colors_key = Some(colors_key);
        }
        if slot.fill_instance.is_none() {
            let instance = page.instances.allocate(1);
            let record = Instance {
                kind: records::KIND_FILL,
                object: slot.object as u32,
                curve: 0,
                _pad: 0,
            };
            page.instances.write(instance, bytemuck::bytes_of(&record));
            slot.fill_instance = Some(instance);
        }
        Some(Draw {
            args: DrawIndirect {
                vertex_count: slot.fill_count as u32,
                instance_count: 1,
                first_vertex: slot.fill_first as u32,
                first_instance: slot.fill_instance.unwrap() as u32,
            },
            page: slot.page,
            opaque,
            camera_distance: 0.0,
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
        layer: usize,
        curve_count: usize,
    ) -> Option<Draw> {
        let stroke_key = [slot.points_version, path.stroke.version];
        let stroke = self
            .paths
            .strokes
            .get_or_build(uid, stroke_key, || stroke(path));
        if !stroke.visible {
            return None;
        }
        let opaque = stroke.opaque;
        let page = &mut self.pages[slot.page];
        if slot.stroke_key != Some(stroke_key) {
            slot.points_first =
                page.stroke_points
                    .place(&stroke.points, slot.points_first, slot.points_count);
            slot.points_count = stroke.points.len() / records::STROKE_POINT_SIZE;
            let mut curves = stroke.curves.clone();
            offset_curves(&mut curves, slot.points_first);
            let count = curves.len() / size_of::<StrokeCurve>();
            let first = page
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
                slot.stroke_instances = Some(page.instances.place(
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
            page: slot.page,
            opaque,
            camera_distance: 0.0,
            layer,
            mask_of: NO_CLIP,
            casts_shadow: false,
            clip: NO_CLIP,
        })
    }

    /// The records placing `path`'s stale ranges would add (an upper
    /// bound: a range of the same size is written over in place).
    fn records_needed(
        &mut self,
        uid: u64,
        path: &Path,
        fill: Option<&Fill>,
        slot: &Slot,
    ) -> Records {
        let version = slot.points_version;
        let mut need = [0; 6];
        if let Some(fill) = fill {
            if slot.fill_version != Some(version) {
                let vertices = self
                    .paths
                    .fill_vertices
                    .get_or_build(uid, version, || fill_vertices(&path.points));
                need[2] = vertices.len() / records::FILL_VERTEX_SIZE;
            }
            if slot.colors_key != Some([version, fill.version]) {
                need[3] = path.points.count();
            }
            need[1] += usize::from(slot.fill_instance.is_none());
        }
        let stroke_key = [version, path.stroke.version];
        if slot.stroke_key != Some(stroke_key) {
            let stroke = self
                .paths
                .strokes
                .get_or_build(uid, stroke_key, || stroke(path));
            need[4] = stroke.points.len() / records::STROKE_POINT_SIZE;
            need[5] = stroke.curves.len() / size_of::<StrokeCurve>();
            need[1] += need[5];
        }
        need
    }

    /// Free `slot` and start the node over on a page with room for all of
    /// its records.
    fn move_to_page_with_room(
        &mut self,
        slot: Slot,
        uid: u64,
        path: &Path,
        fill: Option<&Fill>,
    ) -> Slot {
        self.free(&slot);
        let mut fresh = Slot::new(slot.page, slot.object);
        fresh.points_version = slot.points_version;
        let mut need = self.records_needed(uid, path, fill, &fresh);
        need[0] = 1;
        let page = self.page_with_room(&need);
        let object = self.pages[page].object_slot();
        let mut moved = Slot::new(page, object);
        moved.points_version = slot.points_version;
        moved
    }

    fn free(&mut self, slot: &Slot) {
        let page = &mut self.pages[slot.page];
        page.free_objects.push(slot.object);
        page.fill_vertices.garbage += slot.fill_count;
        page.stroke_points.garbage += slot.points_count;
        page.stroke_curves.garbage += slot.curves_count;
        page.instances.garbage += slot.curves_count + usize::from(slot.fill_instance.is_some());
        page.fill_colors.garbage += slot.colors_count;
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

/// The draw lists: opaque draws (grouped by page: their order does not
/// matter), transparent ones back to front, mask draws per scope in
/// painter's order, shadow casters per scope (grouped by page within each
/// clip scope).
fn batch(scene: &FlatScene, draws: Vec<Draw>) -> VectorDraws {
    // One pass sorts every draw into its lists; opaque ones straight into
    // their page's run (in the frame's order within it).
    let pages = draws.iter().map(|d| d.page + 1).max().unwrap_or(0);
    let mut opaque_pages: Vec<Vec<u8>> = vec![Vec::new(); pages];
    let mut transparent: Vec<&Draw> = Vec::new();
    let mut masks: Vec<Vec<&Draw>> = vec![Vec::new(); scene.clips.len()];
    let mut casters: Vec<((DrawIndirect, usize), i32)> = Vec::new();
    for draw in &draws {
        if draw.mask_of != NO_CLIP {
            masks[draw.mask_of as usize].push(draw);
            continue;
        }
        if draw.casts_shadow {
            casters.push(((draw.args, draw.page), draw.clip));
        }
        match draw.opaque {
            true => opaque_pages[draw.page].extend_from_slice(bytemuck::bytes_of(&draw.args)),
            false => transparent.push(draw),
        }
    }
    let mut opaque = DrawList::default();
    for (page, bytes) in opaque_pages.into_iter().enumerate() {
        if !bytes.is_empty() {
            opaque
                .runs
                .push((page, (bytes.len() / size_of::<DrawIndirect>()) as u32));
            opaque.bytes.extend(bytes);
        }
    }
    // Farthest first; coplanar shapes in painter's order (distances
    // rounded to 6 decimals, so equal depths keep their tree order).
    transparent.sort_by_key(|d| (-((d.camera_distance * 1e6).round() as i64), d.layer));
    let masks = masks
        .into_iter()
        .map(|mut mask| {
            mask.sort_by_key(|d| d.layer);
            DrawList::new(mask.iter().map(|d| (d.args, d.page)))
        })
        .collect();
    casters.sort_by_key(|((_, page), _)| *page); // shadow_lists sorts by clip scope stably after
    let shadows = shadow_lists(scene, &casters)
        .into_iter()
        .map(|(scope, draws, runs)| {
            // Page runs never cross from one clip scope's casters to the next.
            let mut list = DrawList::default();
            let mut draws = draws.into_iter();
            for &(_, count) in &runs {
                let part = DrawList::new(draws.by_ref().take(count as usize));
                list.bytes.extend(part.bytes);
                list.runs.extend(part.runs);
            }
            (scope, list, runs)
        })
        .collect();
    VectorDraws {
        opaque,
        transparent: DrawList::new(transparent.iter().map(|d| (d.args, d.page))),
        masks,
        shadows,
        images: Vec::new(),
    }
}

/// A camera view's layer and `1 / (width, height)` of its plane.
pub type CameraView = (i32, [f32; 2]);

/// The camera views of a frame by node uid.
pub type CameraViews = HashMap<u64, CameraView>;

const NOT_A_VIEW: CameraView = (records::NO_VIEW, [0.0, 0.0]);

/// How far past its points a stroke can reach, in stroke widths: a miter
/// joint up to vector.wgsl's MITER_LIMIT (10) half widths.
const STROKE_REACH: f64 = 5.0;
/// Room around the view, as a fraction of it: what the jitter of
/// RenderSamples and antialiasing move into it from just outside.
const VIEW_MARGIN: f64 = 0.01;

/// What a view shows, to leave out the shapes it cannot (culling).
pub struct Culling {
    /// The six sides of the view (widened by `VIEW_MARGIN`) as planes in
    /// world space: a point `p` is on the inner side of each where
    /// `plane · (p, 1) >= 0` (Gribb and Hartmann, "Fast Extraction of
    /// Viewing Frustum Planes from the World-View-Projection Matrix",
    /// 2001: https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf).
    planes: [DVec4; 6],
    /// Keep shadow casters and masks out of view: with meshes about, a
    /// shadow or a cut can fall into it from outside.
    pub keep_casters: bool,
}

impl Culling {
    /// The culling of the view whose world to clip space matrix (WebGPU's:
    /// depth from 0 to 1) is `view_projection`.
    pub fn new(view_projection: DMat4, keep_casters: bool) -> Culling {
        let row = |index: usize| view_projection.row(index);
        let edge = 1.0 + VIEW_MARGIN;
        Culling {
            planes: [
                row(0) + edge * row(3), // x >= -edge w
                edge * row(3) - row(0), // x <= edge w
                row(1) + edge * row(3),
                edge * row(3) - row(1),
                row(2),          // z >= 0
                row(3) - row(2), // z <= w
            ],
            keep_casters,
        }
    }

    /// Whether some of the world space box `(low, high)` may show: it does
    /// not lie wholly beyond any one side of the view. Per side, the
    /// corner farthest along its inward normal decides.
    fn shows(&self, (low, high): (DVec3, DVec3)) -> bool {
        self.planes.iter().all(|plane| {
            let corner = DVec3::select(plane.truncate().cmpge(DVec3::ZERO), high, low);
            plane.truncate().dot(corner) + plane.w >= 0.0
        })
    }
}

/// The box of `path` (`Points::bounds`) placed by `world`, in world
/// space, widened by how far its stroke can reach; None without points.
fn world_box(path: &Path, world: &DMat4) -> Option<(DVec3, DVec3)> {
    let (low, high) = path.points.bounds()?;
    let (mut world_low, mut world_high) = (DVec3::INFINITY, DVec3::NEG_INFINITY);
    for corner in corners(low, high) {
        let point = world.transform_point3(corner);
        world_low = world_low.min(point);
        world_high = world_high.max(point);
    }
    let widest = path.stroke.widths.iter().copied().fold(0.0, f64::max);
    let reach = DVec3::splat(STROKE_REACH * widest);
    Some((world_low - reach, world_high + reach))
}

/// The eight corners of the box from `low` to `high`.
fn corners(low: DVec3, high: DVec3) -> impl Iterator<Item = DVec3> {
    (0..8).map(move |i| {
        DVec3::new(
            if i & 1 == 0 { low.x } else { high.x },
            if i & 2 == 0 { low.y } else { high.y },
            if i & 4 == 0 { low.z } else { high.z },
        )
    })
}

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

/// The mean of a path's points in 3D (its distance to the camera sorts
/// transparent draws).
fn path_center(points: &Points) -> DVec3 {
    let count = points.count();
    (0..count).map(|index| points.point(index)).sum::<DVec3>() / count as f64
}

/// A fill's triangles (`FILL_VERTEX_SIZE` bytes per vertex).
fn fill_vertices(points: &Points) -> Vec<u8> {
    let corners: Vec<DVec2> = (0..points.count())
        .map(|index| points.point(index).truncate())
        .collect();
    bytemuck::cast_slice(&tessellate_fill(&corners)).to_vec()
}

/// Per-point fill RGBA (float32): the fill's colors and opacities (clamped
/// to [0, 1]) spread over the points like the stroke's lists.
fn fill_colors(points: &Points, fill: &Fill) -> FillColors {
    let count = points.count();
    let opacities: Vec<f64> = fill.opacities.iter().map(|o| o.clamp(0.0, 1.0)).collect();
    let channels: Vec<Vec<f64>> = (0..3)
        .map(|channel| {
            let values: Vec<f64> = fill.colors.iter().map(|color| color[channel]).collect();
            resample(&values, count)
        })
        .collect();
    let alpha = resample(&opacities, count);
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
    FillColors {
        records: bytemuck::cast_slice(&records).to_vec(),
        visible: opacities.iter().any(|&o| o > 0.0),
        opaque: opacities.iter().all(|&o| o >= 1.0),
    }
}

/// A path's stroke records; nothing shows without points or style.
fn stroke(path: &Path) -> StrokeRecords {
    let points = &path.points;
    let style = &path.stroke;
    if points.data.is_empty()
        || style.colors.is_empty()
        || style.opacities.is_empty()
        || style.widths.is_empty()
        || style.joints.is_empty()
    {
        return StrokeRecords {
            curves: Vec::new(),
            points: Vec::new(),
            visible: false,
            opaque: false,
        };
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::pack::store::Changes;
    use crate::robjects::{RNode, Stroke, Transform};

    /// A unit square at the origin, filled and stroked, opaque.
    fn square(uid: u64) -> DrawItem {
        let corners = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)];
        let mut data = Vec::new();
        for index in 0..4 {
            let (a, b) = (corners[index], corners[(index + 1) % 4]);
            for t in [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] {
                data.extend([a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)]);
            }
        }
        let node = Arc::new(RNode {
            uid,
            transform: Transform::IDENTITY,
            alpha: 1.0,
            kind: Kind::Shape2D {
                path: Path {
                    points: Arc::new(Points::new(data, 2)),
                    stroke: Arc::new(Stroke::new(
                        vec![[1.0; 3]],
                        vec![1.0],
                        vec![4.0],
                        vec![0],
                        0,
                    )),
                    cast_shadows: false,
                },
                fill: Arc::new(Fill::new(vec![[1.0, 0.0, 0.0]], vec![1.0])),
            },
        });
        DrawItem {
            node,
            world: DMat4::IDENTITY,
            order: 0,
            clip: NO_CLIP,
            alpha: 1.0,
        }
    }

    fn frame(views: &CameraViews) -> VectorFrame<'_> {
        VectorFrame {
            view: DMat4::IDENTITY,
            culling: Culling::new(DMat4::IDENTITY, false),
            camera_views: views,
            hide_camera_views: false,
        }
    }

    #[test]
    fn a_shape_packs_into_a_fill_and_a_stroke_draw() {
        let flat = FlatScene {
            items: vec![square(1)],
            ..FlatScene::default()
        };
        let (mut packer, mut items, views) = (
            VectorPacker::new(DEFAULT_PAGE_BYTES),
            VectorItems::default(),
            CameraViews::new(),
        );
        let all = Changes {
            all: true,
            places: Vec::new(),
        };
        let draws = packer.pack(&flat, &mut items, &all, &frame(&views));
        assert_eq!(draws.opaque.runs, vec![(0, 2)]);
        assert!(draws.transparent.runs.is_empty());
    }

    #[test]
    fn an_unchanged_scene_writes_and_builds_nothing() {
        let flat = FlatScene {
            items: vec![square(1)],
            ..FlatScene::default()
        };
        let (mut packer, mut items, views) = (
            VectorPacker::new(DEFAULT_PAGE_BYTES),
            VectorItems::default(),
            CameraViews::new(),
        );
        let all = Changes {
            all: true,
            places: Vec::new(),
        };
        packer.pack(&flat, &mut items, &all, &frame(&views));
        for arena in packer.pages[0].arenas_mut() {
            arena.1.ranges(); // as uploaded
        }
        let draws = packer.pack(&flat, &mut items, &Changes::default(), &frame(&views));
        assert_eq!(draws.opaque.runs, vec![(0, 2)]);
        for (name, arena) in packer.pages[0].arenas_mut() {
            assert_eq!(arena.ranges(), Some(Vec::new()), "{name} written again");
        }
        assert_eq!(packer.paths.fill_vertices.len(), 1);
        assert_eq!(packer.paths.strokes.len(), 1);
    }
}
