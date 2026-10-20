//! A bounding volume hierarchy over the boxes of a scene's shapes: which
//! of them a view may show, found in time that grows with what it shows,
//! not with how many there are. Culling every shape on its own cost a
//! test per shape per frame (430 000 shapes for 100 000 dice: 10 ms while
//! the view showed a hundred).
//!
//! The tree is built from the boxes alone, not from the scene's groups:
//! top down, each node's items split at the median of their centers
//! along the axis where the centers spread most, until a leaf holds at
//! most `LEAF_ITEMS` (Pharr, Jakob and Humphreys, "Physically Based
//! Rendering", 4th edition, section 7.3, the "equal counts" split:
//! https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies).
//! The items are kept in leaf order, so every node covers one run of them.
//!
//! Moving shapes keep their place in the tree: a moved item's box is set
//! and every node above it is fitted to its children again (`refit`, once
//! per frame for all moves). The tree then stays correct but may grow
//! loose: it is built again once the summed surface area of its nodes
//! is twice what it was when built (the surface area is what a box adds
//! to the cost of a query: MacDonald and Booth, "Heuristics for ray
//! tracing using space subdivision", 1990:
//! https://doi.org/10.1007/BF01911006). Shapes that move about in
//! place, like a die rolling, hardly loosen it and keep their tree.
//!
//! Items added later (`push`) wait outside the tree, tested one by one,
//! until there are enough of them to be worth building it again: a scene
//! adding shapes every frame does not build the tree every frame.
//!
//! A query walks the tree against the view's planes (`Culling`): a node
//! wholly outside one plane is skipped with everything below it, a node
//! wholly inside every plane yields its run of items untested, and only
//! nodes cut by a plane look further (Assarsson and Möller, "Optimized
//! View Frustum Culling Algorithms for Bounding Boxes", 2000:
//! https://doi.org/10.1080/10867651.2000.10487517).

use glam::{DVec3, DVec4};

/// Items a leaf holds at most.
const LEAF_ITEMS: usize = 4;

/// Items outside the tree before it is built again: at least this many,
/// or an eighth of the tree's.
const LOOSE_ITEMS: usize = 256;

/// No node (a leaf's children).
const NONE: u32 = u32::MAX;

/// An axis-aligned box: low and high corners.
pub type Bounds = (DVec3, DVec3);

#[derive(Clone, Copy)]
struct Node {
    low: DVec3,
    high: DVec3,
    /// Its run of `Bvh::order`.
    first: u32,
    count: u32,
    /// Children (NONE for both in a leaf); always after their parent.
    left: u32,
    right: u32,
    parent: u32,
}

/// The hierarchy over items `0..n` (the caller's indices), each with a
/// box or none (never shown).
#[derive(Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// The items with a box, in leaf order.
    order: Vec<u32>,
    boxes: Vec<Option<Bounds>>,
    /// Each item's leaf (NONE: no box, or added after the build).
    leaf_of: Vec<u32>,
    /// Items with a box added since the build, outside the tree.
    loose: Vec<u32>,
    /// Leaves whose items' boxes were set since the last `refit`.
    stale: Vec<u32>,
    /// The summed surface area of the nodes, now and when built.
    area: f64,
    built_area: f64,
    /// An item gained or lost its box: the tree is built again.
    rebuild: bool,
}

impl Bvh {
    /// The tree over `boxes` (item `i` has `boxes[i]`).
    pub fn new(boxes: Vec<Option<Bounds>>) -> Bvh {
        let mut bvh = Bvh {
            boxes,
            ..Bvh::default()
        };
        bvh.build();
        bvh
    }

    /// Add item `len()` with box `bounds`, outside the tree for now.
    pub fn push(&mut self, bounds: Option<Bounds>) {
        if bounds.is_some() {
            self.loose.push(self.boxes.len() as u32);
        }
        self.boxes.push(bounds);
        self.leaf_of.push(NONE);
    }

    /// Set item `item`'s box. Call `refit` before the next query.
    pub fn set(&mut self, item: usize, bounds: Option<Bounds>) {
        let had = self.boxes[item].is_some();
        self.boxes[item] = bounds;
        if had != bounds.is_some() {
            self.rebuild = true; // in or out of the tree
        } else if bounds.is_some() && self.leaf_of[item] != NONE {
            self.stale.push(self.leaf_of[item]); // a loose item has no node to fit
        }
    }

    /// Fit the tree to the boxes set since the last call: every node above
    /// a changed leaf, children before parents. Builds it again instead
    /// when it may have grown loose, or too many items wait outside it.
    pub fn refit(&mut self) {
        if self.rebuild
            || self.area > 2.0 * self.built_area
            || self.loose.len() > LOOSE_ITEMS.max(self.order.len() / 8)
        {
            self.build();
            return;
        }
        let mut marked = Vec::new();
        for leaf in std::mem::take(&mut self.stale) {
            let mut node = leaf;
            while node != NONE && self.nodes[node as usize].count != 0 {
                // A count of 0 marks a node already queued (children are
                // never empty): everything above it is queued too.
                marked.push((node, self.nodes[node as usize].count));
                self.nodes[node as usize].count = 0;
                node = self.nodes[node as usize].parent;
            }
        }
        // Children come after their parents: fit from the back.
        marked.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for (node, count) in marked {
            let index = node as usize;
            self.nodes[index].count = count;
            let (low, high) = match (self.nodes[index].left, self.nodes[index].right) {
                (NONE, NONE) => self.run_bounds(self.nodes[index].first as usize, count as usize),
                (left, right) => {
                    let (left, right) = (self.nodes[left as usize], self.nodes[right as usize]);
                    (left.low.min(right.low), left.high.max(right.high))
                }
            };
            let node = &mut self.nodes[index];
            self.area += surface(low, high) - surface(node.low, node.high);
            (node.low, node.high) = (low, high);
        }
    }

    /// The items whose boxes `planes` may show (see `Culling`), in
    /// ascending order.
    pub fn query(&self, planes: &[DVec4; 6]) -> Vec<usize> {
        let mut found: Vec<usize> = self
            .loose
            .iter()
            .map(|&item| item as usize)
            .filter(|&item| {
                self.boxes[item]
                    .is_some_and(|(low, high)| classify(planes, low, high) != Side::Outside)
            })
            .collect();
        if self.nodes.is_empty() {
            found.sort_unstable();
            return found;
        }
        let mut stack = vec![0u32];
        while let Some(node) = stack.pop() {
            let node = &self.nodes[node as usize];
            match classify(planes, node.low, node.high) {
                Side::Outside => {}
                Side::Inside => {
                    let run = &self.order[node.first as usize..(node.first + node.count) as usize];
                    found.extend(run.iter().map(|&item| item as usize));
                }
                Side::Crossing if node.left == NONE => {
                    let run = &self.order[node.first as usize..(node.first + node.count) as usize];
                    for &item in run {
                        let (low, high) = self.boxes[item as usize].expect("in the tree");
                        if classify(planes, low, high) != Side::Outside {
                            found.push(item as usize);
                        }
                    }
                }
                Side::Crossing => stack.extend([node.left, node.right]),
            }
        }
        found.sort_unstable();
        found
    }

    /// Build the tree from scratch over the items with boxes.
    fn build(&mut self) {
        self.nodes.clear();
        self.stale.clear();
        self.loose.clear();
        self.rebuild = false;
        self.area = 0.0;
        self.built_area = 0.0;
        self.order = (0..self.boxes.len() as u32)
            .filter(|&item| self.boxes[item as usize].is_some())
            .collect();
        self.leaf_of = vec![NONE; self.boxes.len()];
        if self.order.is_empty() {
            return;
        }
        let centers: Vec<DVec3> = self
            .boxes
            .iter()
            .map(|bounds| bounds.map_or(DVec3::ZERO, |(low, high)| (low + high) / 2.0))
            .collect();
        // Nodes to split: (node, first, count).
        self.push_node(0, self.order.len(), NONE);
        let mut pending = vec![0u32];
        while let Some(node) = pending.pop() {
            let (first, count) = {
                let node = &self.nodes[node as usize];
                (node.first as usize, node.count as usize)
            };
            if count <= LEAF_ITEMS {
                for &item in &self.order[first..first + count] {
                    self.leaf_of[item as usize] = node;
                }
                continue;
            }
            let run = &mut self.order[first..first + count];
            let (low, high) = run.iter().fold(
                (DVec3::INFINITY, DVec3::NEG_INFINITY),
                |(low, high), &item| {
                    let center = centers[item as usize];
                    (low.min(center), high.max(center))
                },
            );
            let axis = (high - low).max_position();
            let half = count / 2;
            run.select_nth_unstable_by(half, |&a, &b| {
                centers[a as usize][axis].total_cmp(&centers[b as usize][axis])
            });
            let left = self.push_node(first, half, node);
            let right = self.push_node(first + half, count - half, node);
            self.nodes[node as usize].left = left;
            self.nodes[node as usize].right = right;
            pending.extend([left, right]);
        }
        // Fit every node, children (always later) first.
        for index in (0..self.nodes.len()).rev() {
            let node = self.nodes[index];
            let (low, high) = if node.left == NONE {
                self.run_bounds(node.first as usize, node.count as usize)
            } else {
                let (left, right) = (
                    self.nodes[node.left as usize],
                    self.nodes[node.right as usize],
                );
                (left.low.min(right.low), left.high.max(right.high))
            };
            self.nodes[index].low = low;
            self.nodes[index].high = high;
            self.area += surface(low, high);
        }
        self.built_area = self.area;
    }

    fn push_node(&mut self, first: usize, count: usize, parent: u32) -> u32 {
        self.nodes.push(Node {
            low: DVec3::INFINITY,
            high: DVec3::NEG_INFINITY,
            first: first as u32,
            count: count as u32,
            left: NONE,
            right: NONE,
            parent,
        });
        self.nodes.len() as u32 - 1
    }

    /// The box around the boxes of `order[first..first + count]`.
    fn run_bounds(&self, first: usize, count: usize) -> Bounds {
        self.order[first..first + count].iter().fold(
            (DVec3::INFINITY, DVec3::NEG_INFINITY),
            |(low, high), &item| {
                let (item_low, item_high) = self.boxes[item as usize].expect("in the tree");
                (low.min(item_low), high.max(item_high))
            },
        )
    }
}

/// The surface area of the box from `low` to `high` (a flat box counts
/// both of its faces).
fn surface(low: DVec3, high: DVec3) -> f64 {
    let size = (high - low).max(DVec3::ZERO);
    2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Outside,
    Inside,
    Crossing,
}

/// Where the box from `low` to `high` lies against the planes: outside
/// when wholly beyond one, inside when wholly within all. Per plane, the
/// corner farthest along its normal decides outside, the nearest inside.
fn classify(planes: &[DVec4; 6], low: DVec3, high: DVec3) -> Side {
    let mut inside = true;
    for plane in planes {
        let normal = plane.truncate();
        let positive = normal.cmpge(DVec3::ZERO);
        if normal.dot(DVec3::select(positive, high, low)) + plane.w < 0.0 {
            return Side::Outside;
        }
        inside &= normal.dot(DVec3::select(positive, low, high)) + plane.w >= 0.0;
    }
    if inside {
        Side::Inside
    } else {
        Side::Crossing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The planes of the box from `low` to `high` (z unbounded).
    fn window(low: DVec3, high: DVec3) -> [DVec4; 6] {
        [
            DVec4::new(1.0, 0.0, 0.0, -low.x),
            DVec4::new(-1.0, 0.0, 0.0, high.x),
            DVec4::new(0.0, 1.0, 0.0, -low.y),
            DVec4::new(0.0, -1.0, 0.0, high.y),
            DVec4::new(0.0, 0.0, 0.0, 1.0),
            DVec4::new(0.0, 0.0, 0.0, 1.0),
        ]
    }

    fn unit(x: f64, y: f64) -> Option<Bounds> {
        Some((DVec3::new(x, y, 0.0), DVec3::new(x + 1.0, y + 1.0, 0.0)))
    }

    /// Every item a brute force test finds, in order.
    fn brute(boxes: &[Option<Bounds>], planes: &[DVec4; 6]) -> Vec<usize> {
        (0..boxes.len())
            .filter(|&item| {
                boxes[item].is_some_and(|(low, high)| classify(planes, low, high) != Side::Outside)
            })
            .collect()
    }

    #[test]
    fn a_query_finds_what_testing_every_box_finds() {
        let boxes: Vec<Option<Bounds>> = (0..1000)
            .map(|i| {
                if i % 97 == 0 {
                    None
                } else {
                    unit((i % 40) as f64 * 2.0, (i / 40) as f64 * 2.0)
                }
            })
            .collect();
        let mut bvh = Bvh::new(boxes.clone());
        let view = window(DVec3::new(10.5, 7.5, 0.0), DVec3::new(31.0, 20.0, 0.0));
        assert_eq!(bvh.query(&view), brute(&boxes, &view));
        assert!(!bvh.query(&view).is_empty());

        // Move some far away and some into view: the tree follows.
        let mut moved = boxes.clone();
        for item in (0..1000).step_by(7) {
            if moved[item].is_some() {
                moved[item] = if item % 2 == 0 {
                    unit(500.0, 500.0)
                } else {
                    unit(20.0, 10.0)
                };
                bvh.set(item, moved[item]);
            }
        }
        bvh.refit();
        assert_eq!(bvh.query(&view), brute(&moved, &view));
        moved[3] = None; // leaves the tree: built again
        bvh.set(3, None);
        bvh.refit();
        assert_eq!(bvh.query(&view), brute(&moved, &view));

        // Items added later wait outside the tree, then join it.
        for i in 0..600 {
            let bounds = unit((i % 30) as f64, 3.0 * (i / 30) as f64);
            moved.push(bounds);
            bvh.push(bounds);
            if i % 50 == 0 {
                moved[1000 + i / 2] = unit(12.0, 9.0); // a loose one moves
                bvh.set(1000 + i / 2, moved[1000 + i / 2]);
                bvh.refit();
                assert_eq!(bvh.query(&view), brute(&moved, &view));
            }
        }
        bvh.refit();
        assert_eq!(bvh.query(&view), brute(&moved, &view));
    }
}
