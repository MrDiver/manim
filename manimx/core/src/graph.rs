//! The scene graph: which nodes a scene has, how they nest, and which
//! signal slot feeds which field. Flattening it with the values of a
//! `FrameState` gives the drawable leaves with their world matrices, like
//! `Node.snapshot()` followed by `SceneFlattener.flatten()`
//! (manimx/renderer/scene_graph.py) gives a `FlatScene`, without building
//! render nodes.

use std::collections::HashMap;

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use glam::{DMat4, DQuat, DVec3};

use crate::evaluator::FrameState;
use crate::value::Value;

/// Clip index of something not inside any `ClippingGroup` (`NO_CLIP`).
pub const NO_CLIP: i32 = -1;

/// The slots of a node's transform parts (`PositionMixin`,
/// `RotationMixin`, `ScalableMixin`); a missing part is the identity.
pub struct Placement {
    pub alpha: usize,
    pub position: Option<[usize; 3]>,
    pub rotation: Option<usize>,
    pub scale: Option<[usize; 3]>,
}

/// Slots of a `Shape2D` (with `fill`) or a `Curve3D` (without).
pub struct PathFields {
    pub points: usize,
    pub fill: Option<(usize, usize)>,
    pub stroke_color: usize,
    pub stroke_opacity: usize,
    pub stroke_width: usize,
    pub stroke_joint: usize,
    pub stroke_cap: usize,
    pub cast_shadows: usize,
}

/// Slots of a `Shape3D`; `particles` for a `Particles` node.
pub struct MeshFields {
    pub mesh: usize,
    /// base_color, opacity, metallic, roughness, emissive,
    /// emissive_strength, normal_scale, occlusion_strength and the five
    /// textures, in the order of `Shape3D._material_signals()`.
    pub material: Vec<usize>,
    pub textures: [usize; 5],
    pub cast_shadows: usize,
    /// positions, rotations, scales, colors.
    pub particles: Option<[usize; 4]>,
}

#[derive(Clone, Copy)]
pub enum LightType {
    Directional,
    Point,
    Spot,
}

pub struct LightFields {
    pub light: LightType,
    pub color: usize,
    pub intensity: usize,
    pub size: usize,
    pub cast_shadows: usize,
    pub range: Option<usize>,
    /// inner, outer.
    pub cone: Option<(usize, usize)>,
}

pub enum NodeData {
    Group {
        children: Vec<usize>,
    },
    ClippingGroup {
        mask: usize,
        content: usize,
        one_sided: usize,
    },
    Path(PathFields),
    Mesh(MeshFields),
    Light(LightFields),
}

pub struct Node {
    pub uid: u64,
    pub placement: Placement,
    pub data: NodeData,
    /// Every slot of the node: a node whose slots all keep their versions
    /// looks the same (its snapshot would be the same render node).
    pub slots: Vec<usize>,
}

/// One drawable leaf with everything resolved from its ancestors (`DrawItem`).
pub struct DrawItem {
    pub node: usize,
    pub world: DMat4,
    /// Position in depth-first tree order (painter's order).
    pub order: usize,
    pub clip: i32,
    pub alpha: f64,
}

/// The mask of one `ClippingGroup` (`ClipScope`).
pub struct ClipScope {
    pub node: usize,
    pub parent: i32,
    pub mask: Vec<DrawItem>,
}

pub struct LightItem {
    pub node: usize,
    pub world: DMat4,
    pub clip: i32,
    pub alpha: f64,
}

#[derive(Default)]
pub struct FlatScene {
    pub items: Vec<DrawItem>,
    pub clips: Vec<ClipScope>,
    pub lights: Vec<LightItem>,
}

impl FlatScene {
    /// Whether something in clip scope `clip` belongs to `scope` (the scope
    /// itself or one nested in it); everything is in NO_CLIP.
    pub fn in_scope(&self, mut clip: i32, scope: i32) -> bool {
        loop {
            if clip == scope || scope == NO_CLIP {
                return true;
            }
            if clip == NO_CLIP {
                return false;
            }
            clip = self.clips[clip as usize].parent;
        }
    }
}

/// A node as Python describes it: `(kind, uid, slots by field name,
/// children)`.
type NodeDescription = (String, u64, HashMap<String, usize>, Vec<usize>);

#[pyclass(module = "manimx._core", frozen)]
pub struct SceneGraph {
    pub nodes: Vec<Node>,
    /// `(node, present slot)` of every root, in the order they were added.
    pub roots: Vec<(usize, usize)>,
}

#[pymethods]
impl SceneGraph {
    /// `nodes[i]` is `(kind, uid, fields, children)`: the node class's
    /// name, its uid, its slots by field name, and the indices of its
    /// children (a `ClippingGroup` has two: mask, content). See
    /// `ScenePlan` in manimx/animation/plan.py.
    #[new]
    fn new(nodes: Vec<NodeDescription>, roots: Vec<(usize, usize)>) -> PyResult<Self> {
        let nodes = nodes
            .into_iter()
            .map(|(kind, uid, fields, children)| parse_node(&kind, uid, &fields, children))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(SceneGraph { nodes, roots })
    }
}

fn parse_node(
    kind: &str,
    uid: u64,
    fields: &HashMap<String, usize>,
    children: Vec<usize>,
) -> PyResult<Node> {
    let field = |name: &str| {
        fields
            .get(name)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("{kind} needs the field {name}")))
    };
    let optional = |name: &str| fields.get(name).copied();
    let axes = |name: &str| -> Option<[usize; 3]> {
        Some([
            optional(&format!("{name}.x"))?,
            optional(&format!("{name}.y"))?,
            optional(&format!("{name}.z"))?,
        ])
    };
    let placement = Placement {
        alpha: field("alpha")?,
        position: axes("position"),
        rotation: optional("rotation"),
        scale: axes("scale"),
    };
    let path = |fill: Option<(usize, usize)>| -> PyResult<NodeData> {
        Ok(NodeData::Path(PathFields {
            points: field("points")?,
            fill,
            stroke_color: field("stroke_color")?,
            stroke_opacity: field("stroke_opacity")?,
            stroke_width: field("stroke_width")?,
            stroke_joint: field("stroke_joint")?,
            stroke_cap: field("stroke_cap")?,
            cast_shadows: field("cast_shadows")?,
        }))
    };
    let mesh = |particles: bool| -> PyResult<NodeData> {
        let names = [
            "base_color",
            "opacity",
            "metallic",
            "roughness",
            "emissive",
            "emissive_strength",
            "normal_scale",
            "occlusion_strength",
            "base_color_texture",
            "metallic_roughness_texture",
            "normal_texture",
            "occlusion_texture",
            "emissive_texture",
        ];
        let material = names
            .iter()
            .map(|name| field(name))
            .collect::<PyResult<Vec<_>>>()?;
        let textures = [
            material[8],
            material[9],
            material[10],
            material[11],
            material[12],
        ];
        let particles = if particles {
            Some([
                field("positions")?,
                field("rotations")?,
                field("scales")?,
                field("colors")?,
            ])
        } else {
            None
        };
        Ok(NodeData::Mesh(MeshFields {
            mesh: field("mesh")?,
            material,
            textures,
            cast_shadows: field("cast_shadows")?,
            particles,
        }))
    };
    let light = |light: LightType| -> PyResult<NodeData> {
        let cone = match light {
            LightType::Spot => Some((field("inner_cone_angle")?, field("outer_cone_angle")?)),
            _ => None,
        };
        Ok(NodeData::Light(LightFields {
            light,
            color: field("color")?,
            intensity: field("intensity")?,
            size: field("size")?,
            cast_shadows: field("cast_shadows")?,
            range: optional("range"),
            cone,
        }))
    };
    let data = match kind {
        "group" => NodeData::Group { children },
        "clipping_group" => NodeData::ClippingGroup {
            mask: children[0],
            content: children[1],
            one_sided: field("one_sided")?,
        },
        "shape2d" => path(Some((field("fill_color")?, field("fill_opacity")?)))?,
        "curve3d" => path(None)?,
        "shape3d" => mesh(false)?,
        "particles" => mesh(true)?,
        "directional_light" => light(LightType::Directional)?,
        "point_light" => light(LightType::Point)?,
        "spot_light" => light(LightType::Spot)?,
        _ => return Err(PyValueError::new_err(format!("unknown node kind {kind:?}"))),
    };
    let mut slots: Vec<usize> = fields.values().copied().collect();
    slots.sort_unstable();
    Ok(Node {
        uid,
        placement,
        data,
        slots,
    })
}

/// Reads field values of a `FrameState`.
pub struct Values<'a> {
    pub state: &'a FrameState,
    pub py: Python<'a>,
}

impl Values<'_> {
    pub fn get(&self, slot: usize) -> &Value {
        &self.state.values[slot]
    }

    pub fn scalar(&self, slot: usize) -> PyResult<f64> {
        self.state.values[slot].scalar()
    }

    /// Python truthiness of a DISCRETE value (`bool(signal.value)`).
    pub fn truthy(&self, slot: usize) -> PyResult<bool> {
        match &self.state.values[slot] {
            Value::Object(object) => object.bind(self.py).is_truthy(),
            Value::Scalar(value) => Ok(*value != 0.0),
            _ => Ok(true),
        }
    }

    pub fn quat(&self, slot: usize) -> PyResult<DQuat> {
        self.state.values[slot].quat()
    }
}

impl SceneGraph {
    /// The node's local matrix (`Node._transform().matrix()`).
    fn local_matrix(&self, node: &Node, values: &Values) -> PyResult<DMat4> {
        let placement = &node.placement;
        let mut position = [0.0; 3];
        if let Some(slots) = placement.position {
            for axis in 0..3 {
                position[axis] = values.scalar(slots[axis])?;
            }
        }
        let rotation = match placement.rotation {
            Some(slot) => values.quat(slot)?,
            None => DQuat::IDENTITY,
        };
        let mut scale = [1.0; 3];
        if let Some(slots) = placement.scale {
            for axis in 0..3 {
                scale[axis] = values.scalar(slots[axis])?;
            }
        }
        // `Transform` normalizes its rotation; the matrix is T @ R @ S.
        Ok(DMat4::from_scale_rotation_translation(
            DVec3::from(scale),
            rotation.normalize(),
            DVec3::from(position),
        ))
    }

    /// `ScalableMixin._hidden()`: scaled to nothing on any axis.
    fn hidden(&self, node: &Node, values: &Values) -> PyResult<bool> {
        if let Some(slots) = node.placement.scale {
            for slot in slots {
                if values.scalar(slot)?.abs() < 1e-9 {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// The drawable leaves, clip scopes and lights of the present roots.
    pub fn flatten(&self, values: &Values) -> PyResult<FlatScene> {
        let mut scene = FlatScene::default();
        for &(root, present) in &self.roots {
            if values.truthy(present)? {
                self.walk(
                    root,
                    &DMat4::IDENTITY,
                    NO_CLIP,
                    &mut scene,
                    None,
                    1.0,
                    values,
                )?;
            }
        }
        Ok(scene)
    }

    /// `_walk` in scene_graph.py, with the snapshot rules of the scene
    /// side: a hidden node is skipped with its subtree, and a clipping
    /// group whose mask or content is hidden shows nothing.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        index: usize,
        parent_world: &DMat4,
        clip: i32,
        scene: &mut FlatScene,
        mask_of: Option<usize>,
        parent_alpha: f64,
        values: &Values,
    ) -> PyResult<()> {
        let node = &self.nodes[index];
        if self.hidden(node, values)? {
            return Ok(());
        }
        let world = *parent_world * self.local_matrix(node, values)?;
        let alpha = (parent_alpha * values.scalar(node.placement.alpha)?).max(0.0);
        match &node.data {
            NodeData::Light(_) => {
                if mask_of.is_none() {
                    scene.lights.push(LightItem {
                        node: index,
                        world,
                        clip,
                        alpha,
                    });
                }
            }
            NodeData::Mesh(_) if mask_of.is_some() => {
                return Err(PyNotImplementedError::new_err(
                    "3D meshes cannot be part of a mask",
                ));
            }
            NodeData::Path(_) | NodeData::Mesh(_) => {
                let target = match mask_of {
                    None => &mut scene.items,
                    Some(scope) => &mut scene.clips[scope].mask,
                };
                let order = target.len();
                target.push(DrawItem {
                    node: index,
                    world,
                    order,
                    clip,
                    alpha,
                });
            }
            NodeData::Group { children } => {
                for &child in children {
                    self.walk(child, &world, clip, scene, mask_of, alpha, values)?;
                }
            }
            NodeData::ClippingGroup { mask, content, .. } => {
                if mask_of.is_some() {
                    return Err(PyNotImplementedError::new_err(
                        "an RClippingGroup inside a mask is not supported",
                    ));
                }
                if self.hidden(&self.nodes[*mask], values)?
                    || self.hidden(&self.nodes[*content], values)?
                {
                    return Ok(()); // nothing to show through
                }
                scene.clips.push(ClipScope {
                    node: index,
                    parent: clip,
                    mask: Vec::new(),
                });
                let scope = scene.clips.len() - 1;
                self.walk(*mask, &world, clip, scene, Some(scope), alpha, values)?;
                self.walk(*content, &world, scope as i32, scene, None, alpha, values)?;
            }
        }
        Ok(())
    }
}
