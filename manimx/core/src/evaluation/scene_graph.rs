//! The scene graph: which nodes a scene has, how they nest, and which
//! signal slot feeds which field of each node. `SceneGraph::snapshot`
//! turns the values of a `FrameState` into the render object tree
//! (robjects/), what `Node.snapshot()` builds for one node in Python:
//!
//! - a root that is not present, and a node scaled to nothing, are left
//!   out with everything below them; a clipping group whose mask or
//!   content is left out shows nothing;
//! - opacities and widths are clamped as the scene side clamps them
//!   (`style.fill`, `style.stroke`, `Shape3D`, the lights).
//!
//! Values are built once per version of the slots they come from and
//! shared by every snapshot until a slot changes; a leaf whose slots all
//! keep their versions is the same render object. So a renderer sees the
//! same versions while nothing changes, and building a tree costs little
//! more than walking it.

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use glam::{DQuat, DVec3};

use crate::evaluation::evaluator::FrameState;
use crate::evaluation::value::Value;
use crate::python::robjects::{joint_values, wrap, PyMeshData, PySurfaceShader, PyTexture};
use crate::robjects::{
    Fill, Instances, Kind, Light, LightKind, Material, Mesh, Path, Points, RNode, Stroke, Surface,
    SurfaceShader, Texture, Transform,
};

/// The slots of a node's transform parts (`PositionMixin`,
/// `RotationMixin`, `ScalableMixin`); a missing part is the identity.
struct Placement {
    alpha: usize,
    position: Option<[usize; 3]>,
    rotation: Option<usize>,
    scale: Option<[usize; 3]>,
}

/// Slots of a `Shape2D` (with `fill`) or a `Curve3D` (without).
struct PathFields {
    points: usize,
    /// color, opacity.
    fill: Option<[usize; 2]>,
    /// The fill's image (a `Texture` or None), if the node has one.
    fill_image: Option<usize>,
    /// color, opacity, width, joint, cap.
    stroke: [usize; 5],
    cast_shadows: usize,
}

/// Slots of a `Shape3D`; `particles` for a `Particles` node.
struct MeshFields {
    mesh: usize,
    /// base_color, opacity, metallic, roughness, emissive,
    /// emissive_strength, normal_scale, occlusion_strength, the five
    /// textures and the surface shader (`MATERIAL_FIELDS`).
    material: Vec<usize>,
    cast_shadows: usize,
    /// positions, rotations, scales, colors.
    particles: Option<[usize; 4]>,
}

const MATERIAL_FIELDS: [&str; 14] = [
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
    "surface_shader",
];

struct LightFields {
    kind: LightKind,
    color: usize,
    intensity: usize,
    size: usize,
    cast_shadows: usize,
    range: Option<usize>,
    /// inner, outer.
    cone: Option<[usize; 2]>,
}

enum NodeData {
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

struct Node {
    uid: u64,
    placement: Placement,
    data: NodeData,
    /// Every slot of the node: a leaf whose slots all keep their versions
    /// is the same render object.
    slots: Vec<usize>,
}

/// What was built for one node, with the slot versions it was built from.
#[derive(Default)]
struct Built {
    leaf: Option<(Vec<u64>, Arc<RNode>)>,
    points: Option<(u64, Arc<Points>)>,
    fill: Option<(Vec<u64>, Arc<Fill>)>,
    stroke: Option<([u64; 5], Arc<Stroke>)>,
    mesh: Option<(u64, Arc<Mesh>)>,
    material: Option<(Vec<u64>, Arc<Material>)>,
    instances: Option<([u64; 4], Arc<Instances>)>,
}

/// A node as Python describes it: `(kind, uid, slots by field name,
/// children)`.
type NodeDescription = (String, u64, HashMap<String, usize>, Vec<usize>);

#[pyclass(module = "manimx._core", frozen)]
pub struct SceneGraph {
    nodes: Vec<Node>,
    /// `(node, present slot)` of every root, in the order they were added;
    /// a root without a present slot is always shown.
    roots: Vec<(usize, Option<usize>)>,
    built: Mutex<Vec<Built>>,
}

#[pymethods]
impl SceneGraph {
    /// `nodes[i]` is `(kind, uid, fields, children)`: the node's kind
    /// (`Node.render_kind`), its uid, its slots by field name, and the
    /// indices of its children (a `ClippingGroup` has two: mask, content).
    /// See `ScenePlan` in manimx/animation/plan.py.
    #[new]
    fn new(nodes: Vec<NodeDescription>, roots: Vec<(usize, Option<usize>)>) -> PyResult<Self> {
        let nodes = nodes
            .into_iter()
            .map(|(kind, uid, fields, children)| parse_node(&kind, uid, &fields, children))
            .collect::<PyResult<Vec<_>>>()?;
        let built = Mutex::new(nodes.iter().map(|_| Built::default()).collect());
        Ok(SceneGraph {
            nodes,
            roots,
            built,
        })
    }

    /// The render object tree for the values of `state`: an `RGroup` with
    /// uid `uid` holding the present roots.
    fn snapshot(&self, py: Python<'_>, state: &FrameState, uid: u64) -> PyResult<Py<PyAny>> {
        wrap(py, self.build(py, state, uid)?)
    }
}

impl SceneGraph {
    pub fn build(&self, py: Python<'_>, state: &FrameState, uid: u64) -> PyResult<Arc<RNode>> {
        let values = Values { state, py };
        let mut built = self.built.lock().expect("not poisoned");
        let mut children = Vec::with_capacity(self.roots.len());
        for &(root, present) in &self.roots {
            if present.map_or(Ok(true), |slot| values.truthy(slot))? {
                if let Some(node) = self.node(root, &values, &mut built)? {
                    children.push(node);
                }
            }
        }
        Ok(Arc::new(RNode {
            uid,
            transform: Transform::IDENTITY,
            alpha: 1.0,
            kind: Kind::Group { children },
        }))
    }

    /// The render object of node `index`, None if it is not drawn.
    fn node(
        &self,
        index: usize,
        values: &Values,
        built: &mut [Built],
    ) -> PyResult<Option<Arc<RNode>>> {
        let node = &self.nodes[index];
        let transform = transform(&node.placement, values)?;
        if transform.is_flat() {
            return Ok(None);
        }
        let alpha = values.scalar(node.placement.alpha)?;
        let kind = match &node.data {
            NodeData::Group { children } => {
                let mut nodes = Vec::with_capacity(children.len());
                for &child in children {
                    if let Some(child) = self.node(child, values, built)? {
                        nodes.push(child);
                    }
                }
                Kind::Group { children: nodes }
            }
            NodeData::ClippingGroup {
                mask,
                content,
                one_sided,
            } => {
                let mask = self.node(*mask, values, built)?;
                let content = self.node(*content, values, built)?;
                if mask.is_none() || content.is_none() {
                    return Ok(None); // nothing to show through
                }
                Kind::ClippingGroup {
                    mask,
                    content,
                    one_sided: values.truthy(*one_sided)?,
                }
            }
            data => {
                let versions: Vec<u64> = node
                    .slots
                    .iter()
                    .map(|&slot| values.state.versions[slot])
                    .collect();
                if let Some((seen, leaf)) = &built[index].leaf {
                    if *seen == versions {
                        return Ok(Some(leaf.clone()));
                    }
                }
                let kind = leaf_kind(data, values, &mut built[index])?;
                let leaf = Arc::new(RNode {
                    uid: node.uid,
                    transform,
                    alpha,
                    kind,
                });
                built[index].leaf = Some((versions, leaf.clone()));
                return Ok(Some(leaf));
            }
        };
        Ok(Some(Arc::new(RNode {
            uid: node.uid,
            transform,
            alpha,
            kind,
        })))
    }
}

/// The node's `Transform` (`Node._transform()`).
fn transform(placement: &Placement, values: &Values) -> PyResult<Transform> {
    let axes = |slots: Option<[usize; 3]>, default: f64| -> PyResult<DVec3> {
        Ok(match slots {
            Some(slots) => DVec3::new(
                values.scalar(slots[0])?,
                values.scalar(slots[1])?,
                values.scalar(slots[2])?,
            ),
            None => DVec3::splat(default),
        })
    };
    Ok(Transform {
        position: axes(placement.position, 0.0)?,
        rotation: match placement.rotation {
            Some(slot) => values.quat(slot)?.normalize(),
            None => DQuat::IDENTITY,
        },
        scale: axes(placement.scale, 1.0)?,
    })
}

fn leaf_kind(data: &NodeData, values: &Values, built: &mut Built) -> PyResult<Kind> {
    Ok(match data {
        NodeData::Path(fields) => {
            let path = path(fields, values, built)?;
            match fields.fill {
                Some(slots) => Kind::Shape2D {
                    path,
                    fill: fill(slots, fields.fill_image, values, built)?,
                },
                None => Kind::Curve3D { path },
            }
        }
        NodeData::Mesh(fields) => {
            let surface = surface(fields, values, built)?;
            match fields.particles {
                Some(slots) => Kind::Particles {
                    surface,
                    instances: instances(slots, values, built)?,
                },
                None => Kind::Shape3D(surface),
            }
        }
        NodeData::Light(fields) => Kind::Light(light(fields, values)?),
        NodeData::Group { .. } | NodeData::ClippingGroup { .. } => unreachable!(),
    })
}

/// A value built once per version of its slots (`key`).
fn cached<K: PartialEq, T>(
    entry: &mut Option<(K, Arc<T>)>,
    key: K,
    build: impl FnOnce() -> PyResult<T>,
) -> PyResult<Arc<T>> {
    if let Some((seen, value)) = entry {
        if *seen == key {
            return Ok(value.clone());
        }
    }
    let value = Arc::new(build()?);
    *entry = Some((key, value.clone()));
    Ok(value)
}

fn path(fields: &PathFields, values: &Values, built: &mut Built) -> PyResult<Path> {
    let slot = fields.points;
    let points = cached(&mut built.points, values.version(slot), || {
        let Value::Array(array) = values.get(slot) else {
            return Err(PyTypeError::new_err("points must be a path"));
        };
        // Without points the array has no columns: a Shape2D's are 2D, a Curve3D's 3D.
        let dimension = match (array.data.is_empty(), fields.fill) {
            (false, _) => array.cols(),
            (true, Some(_)) => 2,
            (true, None) => 3,
        };
        Ok(Points::new(array.data.clone(), dimension))
    })?;
    let [color, opacity, width, joint, cap] = fields.stroke;
    let stroke = cached(
        &mut built.stroke,
        fields.stroke.map(|s| values.version(s)),
        || {
            Ok(Stroke::new(
                values.colors(color)?,
                unit(values.floats(opacity)?),
                values
                    .floats(width)?
                    .into_iter()
                    .map(|w| w.max(0.0))
                    .collect(),
                joint_values(values.object(joint)?.bind(values.py))?,
                values
                    .object(cap)?
                    .bind(values.py)
                    .getattr("value")?
                    .extract()?,
            ))
        },
    )?;
    Ok(Path {
        points,
        stroke,
        cast_shadows: values.truthy(fields.cast_shadows)?,
    })
}

fn fill(
    slots: [usize; 2],
    image: Option<usize>,
    values: &Values,
    built: &mut Built,
) -> PyResult<Arc<Fill>> {
    let mut key: Vec<u64> = slots.iter().map(|&s| values.version(s)).collect();
    key.extend(image.map(|s| values.version(s)));
    cached(&mut built.fill, key, || {
        Ok(Fill::with_image(
            values.colors(slots[0])?,
            unit(values.floats(slots[1])?),
            image
                .map(|slot| texture(slot, values))
                .transpose()?
                .flatten(),
        ))
    })
}

/// A `Texture` signal's value (None for None).
fn texture(slot: usize, values: &Values) -> PyResult<Option<Arc<Texture>>> {
    let object = values.object(slot)?;
    let object = object.bind(values.py);
    if object.is_none() {
        return Ok(None);
    }
    Ok(Some(object.cast::<PyTexture>()?.get().texture.clone()))
}

fn unit(values: Vec<f64>) -> Vec<f64> {
    values.into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

fn surface(fields: &MeshFields, values: &Values, built: &mut Built) -> PyResult<Surface> {
    // A MeshData already is a shared value: keep it, version and all.
    let version = values.version(fields.mesh);
    let mesh = match &built.mesh {
        Some((seen, mesh)) if *seen == version => mesh.clone(),
        _ => {
            let object = values.object(fields.mesh)?;
            let mesh = object
                .bind(values.py)
                .cast::<PyMeshData>()?
                .get()
                .mesh
                .clone();
            built.mesh = Some((version, mesh.clone()));
            mesh
        }
    };
    let versions: Vec<u64> = fields.material.iter().map(|&s| values.version(s)).collect();
    let material = cached(&mut built.material, versions, || {
        material(&fields.material, values)
    })?;
    Ok(Surface {
        mesh,
        material,
        cast_shadows: values.truthy(fields.cast_shadows)?,
    })
}

/// The `PbrMaterial` of a Shape3D's material signals, clamped as
/// `Shape3D` clamps them.
fn material(slots: &[usize], values: &Values) -> PyResult<Material> {
    let unit = |slot: usize| -> PyResult<f64> { Ok(values.scalar(slot)?.clamp(0.0, 1.0)) };
    let texture = |slot: usize| texture(slot, values);
    let surface_shader = {
        let object = values.object(slots[13])?;
        let object = object.bind(values.py);
        if object.is_none() {
            None::<Arc<SurfaceShader>>
        } else {
            Some(object.cast::<PySurfaceShader>()?.get().shader.clone())
        }
    };
    Ok(Material {
        base_color: values.color(slots[0])?,
        opacity: unit(slots[1])?,
        metallic: unit(slots[2])?,
        roughness: unit(slots[3])?,
        emissive: values.color(slots[4])?,
        emissive_strength: values.scalar(slots[5])?.max(0.0),
        normal_scale: values.scalar(slots[6])?,
        occlusion_strength: unit(slots[7])?,
        textures: [
            texture(slots[8])?,
            texture(slots[9])?,
            texture(slots[10])?,
            texture(slots[11])?,
            texture(slots[12])?,
        ],
        surface_shader,
        ..Material::default()
    })
}

fn instances(slots: [usize; 4], values: &Values, built: &mut Built) -> PyResult<Arc<Instances>> {
    cached(
        &mut built.instances,
        slots.map(|s| values.version(s)),
        || {
            Ok(Instances::new(
                values.rows::<3>(slots[0])?,
                values.rows::<4>(slots[1])?,
                values.rows::<3>(slots[2])?,
                values.rows::<4>(slots[3])?,
            ))
        },
    )
}

/// A light, clamped as `Light._build_snapshot` clamped it: no negative
/// intensity or size, a range of 0 or less is none, and spot cone angles
/// kept valid (0 <= inner < outer <= pi / 2).
fn light(fields: &LightFields, values: &Values) -> PyResult<Light> {
    let range = match fields.range {
        Some(slot) => Some(values.scalar(slot)?).filter(|&range| range > 0.0),
        None => None,
    };
    let cone = match fields.cone {
        Some([inner, outer]) => {
            let outer = values.scalar(outer)?.clamp(1e-6, FRAC_PI_2);
            let inner = values.scalar(inner)?.max(0.0).min(outer * (1.0 - 1e-6));
            Some((inner, outer))
        }
        None => None,
    };
    Ok(Light {
        kind: fields.kind,
        color: values.color(fields.color)?,
        intensity: values.scalar(fields.intensity)?.max(0.0),
        size: values.scalar(fields.size)?.max(0.0),
        cast_shadows: values.truthy(fields.cast_shadows)?,
        range,
        cone,
    })
}

/// Reads the values of a `FrameState`.
struct Values<'a> {
    state: &'a FrameState,
    py: Python<'a>,
}

impl Values<'_> {
    fn get(&self, slot: usize) -> &Value {
        &self.state.values[slot]
    }

    fn version(&self, slot: usize) -> u64 {
        self.state.versions[slot]
    }

    /// The value as the Python signal holds it.
    fn object(&self, slot: usize) -> PyResult<Py<PyAny>> {
        self.state.kinds[slot].value_to_py(self.py, self.get(slot))
    }

    fn scalar(&self, slot: usize) -> PyResult<f64> {
        match self.get(slot) {
            Value::Scalar(value) => Ok(*value),
            _ => self.object(slot)?.extract(self.py),
        }
    }

    fn quat(&self, slot: usize) -> PyResult<DQuat> {
        self.get(slot).quat()
    }

    /// Python truthiness (`bool(signal.value)`).
    fn truthy(&self, slot: usize) -> PyResult<bool> {
        match self.get(slot) {
            Value::Object(object) => object.bind(self.py).is_truthy(),
            Value::Scalar(value) => Ok(*value != 0.0),
            _ => Ok(true),
        }
    }

    /// All numbers of the value, row by row.
    fn floats(&self, slot: usize) -> PyResult<Vec<f64>> {
        Ok(match self.get(slot) {
            Value::Scalar(value) => vec![*value],
            Value::Tuple(values) => values.to_vec(),
            Value::Array(array) => array.data.clone(),
            _ => {
                let numpy = self.py.import("numpy")?;
                let array = numpy.call_method1("ravel", (self.object(slot)?,))?;
                numpy
                    .call_method1("ascontiguousarray", (array, "float64"))?
                    .call_method0("tolist")?
                    .extract()?
            }
        })
    }

    /// Rows of `N` numbers.
    fn rows<const N: usize>(&self, slot: usize) -> PyResult<Vec<[f64; N]>> {
        let floats = self.floats(slot)?;
        if !floats.len().is_multiple_of(N) {
            return Err(PyValueError::new_err(format!(
                "expected rows of {N} numbers"
            )));
        }
        Ok(floats
            .chunks_exact(N)
            .map(|row| row.try_into().unwrap())
            .collect())
    }

    /// A COLORS value: one sRGB color or a list of them.
    fn colors(&self, slot: usize) -> PyResult<Vec<[f64; 3]>> {
        self.rows::<3>(slot)
    }

    /// A COLOR value (3 or 4 channels): its sRGB color.
    fn color(&self, slot: usize) -> PyResult<[f64; 3]> {
        let channels = self.floats(slot)?;
        if channels.len() < 3 {
            return Err(PyValueError::new_err("a color needs 3 or 4 channels"));
        }
        Ok([channels[0], channels[1], channels[2]])
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
    let path = |fill: Option<[usize; 2]>| -> PyResult<NodeData> {
        Ok(NodeData::Path(PathFields {
            points: field("points")?,
            fill,
            fill_image: if fill.is_some() {
                optional("fill_image")
            } else {
                None
            },
            stroke: [
                field("stroke_color")?,
                field("stroke_opacity")?,
                field("stroke_width")?,
                field("stroke_joint")?,
                field("stroke_cap")?,
            ],
            cast_shadows: field("cast_shadows")?,
        }))
    };
    let mesh = |particles: bool| -> PyResult<NodeData> {
        let material = MATERIAL_FIELDS
            .iter()
            .map(|name| field(name))
            .collect::<PyResult<Vec<_>>>()?;
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
            cast_shadows: field("cast_shadows")?,
            particles,
        }))
    };
    let light = |kind: LightKind| -> PyResult<NodeData> {
        let cone = match kind {
            LightKind::Spot => Some([field("inner_cone_angle")?, field("outer_cone_angle")?]),
            _ => None,
        };
        Ok(NodeData::Light(LightFields {
            kind,
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
        "shape2d" => path(Some([field("fill_color")?, field("fill_opacity")?]))?,
        "curve3d" => path(None)?,
        "shape3d" => mesh(false)?,
        "particles" => mesh(true)?,
        "directional_light" => light(LightKind::Directional)?,
        "point_light" => light(LightKind::Point)?,
        "spot_light" => light(LightKind::Spot)?,
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
