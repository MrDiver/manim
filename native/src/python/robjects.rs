//! The render objects as Python classes in `manim._native`: `RGroup`,
//! `RShape2D`, ..., and the values they hold (`Fill`, `Stroke`,
//! `MeshData`, `PbrMaterial`, `Texture`, `SurfaceShader`).
//!
//! Every class wraps the Rust value of robjects/ behind an `Arc`, so a tree
//! built in Python is handed to a renderer without a copy. The classes are
//! immutable. Reading a field makes a new Python object each time (a numpy
//! array, a `Fill`, a child node); compare render objects by `uid` and
//! values by `version`, never by identity.
//!
//! The constructors take plain data and the getters return plain data:
//! colors as sRGB float rows, opacities and widths as float arrays,
//! joints and caps as ints, points and meshes as numpy arrays. Turning
//! what users write (color names, `ManimColor`, enums) into this data is
//! the scene side's job, so nothing here imports a manim module except
//! the shared value `manim.utils.transform.Transform`.

use std::f64::consts::FRAC_PI_4;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use glam::{DQuat, DVec3};

use crate::robjects::values::POINTS_PER_CURVE;
use crate::robjects::{
    new_uid, Fill, Instances, Kind, Light, LightKind, Material, Mesh, Path, Points, RNode, Stroke,
    Surface, SurfaceShader, Texture, Transform,
};

// -- conversions ----------------------------------------------------------------

/// A numpy array of `data` with `shape`, read-only.
fn read_only<'py, T: numpy::Element>(
    py: Python<'py>,
    data: Vec<T>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let array = data.into_pyarray(py).reshape(shape.to_vec())?;
    let flags = PyDict::new(py);
    flags.set_item("write", false)?;
    array.call_method("setflags", (), Some(&flags))?;
    Ok(array.into_any())
}

/// `object` as a C-contiguous numpy array of `dtype`: its shape and values.
fn array_of<T: numpy::Element + Copy>(
    object: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<(Vec<usize>, Vec<T>)> {
    let numpy = object.py().import("numpy")?;
    let array = numpy.call_method1("ascontiguousarray", (object, dtype))?;
    let array = array.cast::<PyArrayDyn<T>>()?;
    Ok((array.shape().to_vec(), array.to_vec()?))
}

/// Rows of `N` values: `object` must have shape `(rows, N)`.
fn rows<T: numpy::Element + Copy + Default, const N: usize>(
    object: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<Vec<[T; N]>> {
    let (shape, data) = array_of::<T>(object, dtype)?;
    if shape.len() != 2 || shape[1] != N {
        return Err(PyValueError::new_err(format!(
            "expected shape (N, {N}), got {}",
            shape_text(&shape)
        )));
    }
    Ok(data
        .chunks_exact(N)
        .map(|row| {
            let mut out = [T::default(); N];
            out.copy_from_slice(row);
            out
        })
        .collect())
}

fn shape_text(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("({})", parts.join(", "))
}

fn flat<T: Copy, const N: usize>(rows: &[[T; N]]) -> Vec<T> {
    rows.iter().flatten().copied().collect()
}

/// One sRGB color `(3,)` or a list of them `(K, 3)`, each component in
/// `[0, 1]`; white for None.
fn colors(color: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<[f64; 3]>> {
    let Some(color) = color else {
        return Ok(vec![[1.0; 3]]);
    };
    let (shape, data) = array_of::<f64>(color, "float64")?;
    let rows = match shape.as_slice() {
        [3] => vec![[data[0], data[1], data[2]]],
        [count, 3] if *count > 0 => data
            .chunks_exact(3)
            .map(|row| [row[0], row[1], row[2]])
            .collect(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "color must have shape (3,) or (K, 3) with K > 0, got {}",
                shape_text(&shape)
            )))
        }
    };
    if data.iter().any(|c| !(0.0..=1.0).contains(c)) {
        return Err(PyValueError::new_err("color components must be in [0, 1]"));
    }
    Ok(rows)
}

/// One sRGB color `(3,)`, each component in `[0, 1]`; `default` for None.
fn color(value: Option<&Bound<'_, PyAny>>, default: [f64; 3]) -> PyResult<[f64; 3]> {
    match value {
        None => Ok(default),
        Some(value) => match colors(Some(value))?.as_slice() {
            [one] => Ok(*one),
            _ => Err(PyValueError::new_err("expected one color of shape (3,)")),
        },
    }
}

/// One number or a list of them, each in `[low, high]`; `default` for None.
fn scalars(
    values: Option<&Bound<'_, PyAny>>,
    default: f64,
    name: &str,
    low: f64,
    high: f64,
) -> PyResult<Vec<f64>> {
    let Some(values) = values else {
        return Ok(vec![default]);
    };
    let (shape, data) = array_of::<f64>(values, "float64")?;
    if shape.len() > 1 || data.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{name} must be one number or a non-empty list, got shape {}",
            shape_text(&shape)
        )));
    }
    if let Some(bad) = data.iter().find(|v| !(low..=high).contains(*v)) {
        return Err(PyValueError::new_err(format!(
            "{name} must be in [{low}, {high}], got {bad}"
        )));
    }
    Ok(data)
}

/// One int or a non-empty list of them.
fn ints(values: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u32>> {
    let (shape, data) = array_of::<u32>(values, "uint32")?;
    if shape.len() > 1 || data.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{name} must be one int or a non-empty list, got shape {}",
            shape_text(&shape)
        )));
    }
    Ok(data)
}

/// `value` unless it is Python's None.
fn given<'a, 'py>(value: Option<&'a Bound<'py, PyAny>>) -> Option<&'a Bound<'py, PyAny>> {
    value.filter(|v| !v.is_none())
}

/// A `manim.utils.transform.Transform` (identity for None).
fn transform_from_py(transform: Option<&Bound<'_, PyAny>>) -> PyResult<Transform> {
    let Some(transform) = given(transform) else {
        return Ok(Transform::IDENTITY);
    };
    let position: (f64, f64, f64) = transform.getattr("position")?.extract()?;
    let scale: (f64, f64, f64) = transform.getattr("scale")?.extract()?;
    // numpy-quaternion components are (w, x, y, z).
    let rotation: Vec<f64> = transform
        .getattr("rotation")?
        .getattr("components")?
        .extract()?;
    Ok(Transform {
        position: DVec3::from(position),
        rotation: DQuat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]),
        scale: DVec3::from(scale),
    })
}

fn transform_to_py<'py>(py: Python<'py>, t: &Transform) -> PyResult<Bound<'py, PyAny>> {
    let q = t.rotation;
    let rotation = py
        .import("quaternion")?
        .getattr("quaternion")?
        .call1((q.w, q.x, q.y, q.z))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("position", t.position.to_array())?;
    kwargs.set_item("rotation", rotation)?;
    kwargs.set_item("scale", t.scale.to_array())?;
    py.import("manim.utils.transform")?
        .getattr("Transform")?
        .call((), Some(&kwargs))
}

// -- values ---------------------------------------------------------------------

/// Fill style: one sRGB color and opacity, or lists of them (a gradient
/// along the path, spread over its points by the renderer), and
/// optionally an `image` (a `Texture`) the colors tint, stretched over
/// the shape's local box around its origin.
#[pyclass(module = "manim._native", name = "Fill", frozen)]
pub struct PyFill {
    pub fill: Arc<Fill>,
}

#[pymethods]
impl PyFill {
    #[new]
    #[pyo3(signature = (color=None, opacity=None, image=None))]
    fn new(
        color: Option<&Bound<'_, PyAny>>,
        opacity: Option<&Bound<'_, PyAny>>,
        image: Option<Bound<'_, PyTexture>>,
    ) -> PyResult<Self> {
        let colors = colors(color)?;
        let opacities = scalars(opacity, 0.0, "opacity", 0.0, 1.0)?;
        let image = image.map(|image| image.get().texture.clone());
        Ok(PyFill {
            fill: Arc::new(Fill::with_image(colors, opacities, image)),
        })
    }

    /// The image the colors tint, or None.
    #[getter]
    fn image(&self, py: Python<'_>) -> PyResult<Option<Py<PyTexture>>> {
        self.fill
            .image
            .as_ref()
            .map(|texture| {
                Py::new(
                    py,
                    PyTexture {
                        texture: texture.clone(),
                    },
                )
            })
            .transpose()
    }

    /// The colors, `(K, 3)` sRGB.
    #[getter]
    fn color<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_only(py, flat(&self.fill.colors), &[self.fill.colors.len(), 3])
    }

    /// The opacities, `(K,)`.
    #[getter]
    fn opacity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_only(
            py,
            self.fill.opacities.clone(),
            &[self.fill.opacities.len()],
        )
    }

    /// Equal versions: the same fill.
    #[getter]
    fn version(&self) -> u64 {
        self.fill.version
    }

    fn __repr__(&self) -> String {
        format!(
            "Fill(color={:?}, opacity={:?})",
            self.fill.colors, self.fill.opacities
        )
    }
}

/// Stroke style; every field except `cap` takes one value or a list.
/// Colors (sRGB), opacities and widths are spread over the path's points,
/// joints over its curves (entry `i` is the joint at the end of curve
/// `i`). Joints and the cap are the int values of `LineJointType` and
/// `CapStyleType`.
#[pyclass(module = "manim._native", name = "Stroke", frozen)]
pub struct PyStroke {
    pub stroke: Arc<Stroke>,
}

#[pymethods]
impl PyStroke {
    #[new]
    #[pyo3(signature = (color=None, opacity=None, width=None, joint=None, cap=0))]
    fn new(
        color: Option<&Bound<'_, PyAny>>,
        opacity: Option<&Bound<'_, PyAny>>,
        width: Option<&Bound<'_, PyAny>>,
        joint: Option<&Bound<'_, PyAny>>,
        cap: u32,
    ) -> PyResult<Self> {
        let joints = match joint {
            None => vec![0],
            Some(joint) => ints(joint, "joint")?,
        };
        Ok(PyStroke {
            stroke: Arc::new(Stroke::new(
                colors(color)?,
                scalars(opacity, 1.0, "opacity", 0.0, 1.0)?,
                scalars(width, 4.0, "width", 0.0, f64::INFINITY)?,
                joints,
                cap,
            )),
        })
    }

    /// The colors, `(K, 3)` sRGB.
    #[getter]
    fn color<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_only(
            py,
            flat(&self.stroke.colors),
            &[self.stroke.colors.len(), 3],
        )
    }

    #[getter]
    fn opacity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_only(
            py,
            self.stroke.opacities.clone(),
            &[self.stroke.opacities.len()],
        )
    }

    #[getter]
    fn width<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_only(py, self.stroke.widths.clone(), &[self.stroke.widths.len()])
    }

    /// The joints, `(K,)` ints.
    #[getter]
    fn joint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_only(py, self.stroke.joints.clone(), &[self.stroke.joints.len()])
    }

    #[getter]
    fn cap(&self) -> u32 {
        self.stroke.cap
    }

    /// Equal versions: the same stroke.
    #[getter]
    fn version(&self) -> u64 {
        self.stroke.version
    }

    fn __repr__(&self) -> String {
        let s = &self.stroke;
        format!(
            "Stroke(color={:?}, opacity={:?}, width={:?}, joint={:?}, cap={})",
            s.colors, s.opacities, s.widths, s.joints, s.cap
        )
    }
}

/// An indexed triangle mesh in local coordinates with a normal per
/// vertex. `tangents` (xyz plus handedness in w, as in glTF) are only
/// needed for normal maps.
#[pyclass(module = "manim._native", name = "MeshData", frozen)]
pub struct PyMeshData {
    pub mesh: Arc<Mesh>,
}

#[pymethods]
impl PyMeshData {
    #[new]
    #[pyo3(signature = (positions, indices, normals, uvs=None, tangents=None))]
    fn new(
        py: Python<'_>,
        positions: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        normals: &Bound<'_, PyAny>,
        uvs: Option<&Bound<'_, PyAny>>,
        tangents: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let positions = rows::<f32, 3>(positions, "float32")?;
        let flat_indices = py
            .import("numpy")?
            .call_method1("asarray", (indices,))?
            .call_method1("reshape", (-1,))?;
        let indices = array_of::<u32>(&flat_indices, "uint32")?.1;
        let normals = rows::<f32, 3>(normals, "float32")?;
        let uvs = given(uvs)
            .map(|u| rows::<f32, 2>(u, "float32"))
            .transpose()?;
        let tangents = given(tangents)
            .map(|t| rows::<f32, 4>(t, "float32"))
            .transpose()?;
        let mesh =
            Mesh::new(positions, indices, normals, uvs, tangents).map_err(PyValueError::new_err)?;
        Ok(PyMeshData {
            mesh: Arc::new(mesh),
        })
    }

    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let m = &self.mesh;
        read_only(py, flat(&m.positions), &[m.positions.len(), 3])
    }

    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        read_only(py, self.mesh.indices.clone(), &[self.mesh.indices.len()])
    }

    #[getter]
    fn normals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let m = &self.mesh;
        read_only(py, flat(&m.normals), &[m.normals.len(), 3])
    }

    #[getter]
    fn uvs<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.mesh
            .uvs
            .as_ref()
            .map(|uvs| read_only(py, flat(uvs), &[uvs.len(), 2]))
            .transpose()
    }

    #[getter]
    fn tangents<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.mesh
            .tangents
            .as_ref()
            .map(|tangents| read_only(py, flat(tangents), &[tangents.len(), 4]))
            .transpose()
    }

    /// Equal versions: the same mesh.
    #[getter]
    fn version(&self) -> u64 {
        self.mesh.version
    }

    #[getter]
    fn vertex_count(&self) -> usize {
        self.mesh.positions.len()
    }

    #[getter]
    fn triangle_count(&self) -> usize {
        self.mesh.triangle_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "MeshData({} vertices, {} triangles)",
            self.mesh.positions.len(),
            self.mesh.triangle_count()
        )
    }
}

/// An RGBA8 image used by a material, `(H, W, 4)` uint8 with the first row
/// at the top. `uid` identifies it, so a renderer uploads it once however
/// many materials share it.
#[pyclass(module = "manim._native", name = "Texture", frozen)]
pub struct PyTexture {
    pub texture: Arc<Texture>,
}

#[pymethods]
impl PyTexture {
    #[new]
    fn new(image: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (shape, pixels) = array_of::<u8>(image, "uint8")?;
        if shape.len() != 3 || shape[2] != 4 {
            return Err(PyValueError::new_err(format!(
                "texture image must have shape (H, W, 4), got {}",
                shape_text(&shape)
            )));
        }
        Ok(PyTexture {
            texture: Arc::new(Texture::new(shape[1] as u32, shape[0] as u32, pixels)),
        })
    }

    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let t = &self.texture;
        read_only(
            py,
            t.pixels.clone(),
            &[t.height as usize, t.width as usize, 4],
        )
    }

    /// `(width, height)` in pixels.
    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.texture.width, self.texture.height)
    }

    #[getter]
    fn uid(&self) -> u64 {
        self.texture.uid
    }

    fn __repr__(&self) -> String {
        format!(
            "Texture({}x{}, uid={})",
            self.texture.width, self.texture.height, self.texture.uid
        )
    }
}

/// WGSL that changes a `RShape3D`'s surface per pixel, before lighting.
/// `code` defines `fn shade(surface: Surface, in: SurfaceInput) -> Surface`
/// (and any helpers it needs). `uid` identifies the shader, so a renderer
/// compiles it once however many shapes share it.
#[pyclass(module = "manim._native", name = "SurfaceShader", frozen)]
pub struct PySurfaceShader {
    pub shader: Arc<SurfaceShader>,
}

#[pymethods]
impl PySurfaceShader {
    #[new]
    fn new(code: String) -> PyResult<Self> {
        if !code.contains("fn shade(") {
            return Err(PyValueError::new_err(
                "a SurfaceShader must define fn shade(surface: Surface, in: SurfaceInput) -> Surface",
            ));
        }
        Ok(PySurfaceShader {
            shader: Arc::new(SurfaceShader::new(code)),
        })
    }

    #[getter]
    fn code(&self) -> &str {
        &self.shader.code
    }

    #[getter]
    fn uid(&self) -> u64 {
        self.shader.uid
    }
}

/// The glTF 2.0 metallic-roughness material. `base_color` and `emissive`
/// are sRGB `(3,)`; lighting happens in linear RGB. Each factor is
/// multiplied with its texture where one is set: `base_color_texture`
/// (sRGB RGBA), `metallic_roughness_texture` (linear, roughness in G,
/// metallic in B), `normal_texture` (tangent-space normals, needs mesh
/// UVs), `occlusion_texture` (linear, in R), `emissive_texture` (sRGB
/// RGB). `surface_shader` changes the surface per pixel.
#[pyclass(module = "manim._native", name = "PbrMaterial", frozen)]
pub struct PyPbrMaterial {
    pub material: Arc<Material>,
}

#[pymethods]
impl PyPbrMaterial {
    #[new]
    #[pyo3(signature = (
        base_color=None, opacity=1.0, metallic=0.0, roughness=0.5, emissive=None,
        emissive_strength=1.0, normal_scale=1.0, occlusion_strength=1.0,
        base_color_texture=None, metallic_roughness_texture=None, normal_texture=None,
        occlusion_texture=None, emissive_texture=None, surface_shader=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_color: Option<&Bound<'_, PyAny>>,
        opacity: f64,
        metallic: f64,
        roughness: f64,
        emissive: Option<&Bound<'_, PyAny>>,
        emissive_strength: f64,
        normal_scale: f64,
        occlusion_strength: f64,
        base_color_texture: Option<Bound<'_, PyTexture>>,
        metallic_roughness_texture: Option<Bound<'_, PyTexture>>,
        normal_texture: Option<Bound<'_, PyTexture>>,
        occlusion_texture: Option<Bound<'_, PyTexture>>,
        emissive_texture: Option<Bound<'_, PyTexture>>,
        surface_shader: Option<Bound<'_, PySurfaceShader>>,
    ) -> PyResult<Self> {
        for (name, value) in [
            ("opacity", opacity),
            ("metallic", metallic),
            ("roughness", roughness),
            ("occlusion_strength", occlusion_strength),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(PyValueError::new_err(format!(
                    "{name} must be in [0, 1], got {value}"
                )));
            }
        }
        let texture = |t: Option<Bound<'_, PyTexture>>| t.map(|t| t.get().texture.clone());
        let material = Material {
            base_color: color(base_color, [1.0; 3])?,
            opacity,
            metallic,
            roughness,
            emissive: color(emissive, [0.0; 3])?,
            emissive_strength,
            normal_scale,
            occlusion_strength,
            textures: [
                texture(base_color_texture),
                texture(metallic_roughness_texture),
                texture(normal_texture),
                texture(occlusion_texture),
                texture(emissive_texture),
            ],
            surface_shader: surface_shader.map(|s| s.get().shader.clone()),
            ..Material::default()
        };
        Ok(PyPbrMaterial {
            material: Arc::new(material),
        })
    }

    /// sRGB `(r, g, b)`.
    #[getter]
    fn base_color(&self) -> (f64, f64, f64) {
        let [r, g, b] = self.material.base_color;
        (r, g, b)
    }

    /// Equal versions: the same material.
    #[getter]
    fn version(&self) -> u64 {
        self.material.version
    }

    #[getter]
    fn opacity(&self) -> f64 {
        self.material.opacity
    }

    #[getter]
    fn metallic(&self) -> f64 {
        self.material.metallic
    }

    #[getter]
    fn roughness(&self) -> f64 {
        self.material.roughness
    }

    /// sRGB `(r, g, b)`.
    #[getter]
    fn emissive(&self) -> (f64, f64, f64) {
        let [r, g, b] = self.material.emissive;
        (r, g, b)
    }

    #[getter]
    fn emissive_strength(&self) -> f64 {
        self.material.emissive_strength
    }

    #[getter]
    fn normal_scale(&self) -> f64 {
        self.material.normal_scale
    }

    #[getter]
    fn occlusion_strength(&self) -> f64 {
        self.material.occlusion_strength
    }

    #[getter]
    fn base_color_texture(&self, py: Python<'_>) -> PyResult<Option<Py<PyTexture>>> {
        self.texture(py, 0)
    }

    #[getter]
    fn metallic_roughness_texture(&self, py: Python<'_>) -> PyResult<Option<Py<PyTexture>>> {
        self.texture(py, 1)
    }

    #[getter]
    fn normal_texture(&self, py: Python<'_>) -> PyResult<Option<Py<PyTexture>>> {
        self.texture(py, 2)
    }

    #[getter]
    fn occlusion_texture(&self, py: Python<'_>) -> PyResult<Option<Py<PyTexture>>> {
        self.texture(py, 3)
    }

    #[getter]
    fn emissive_texture(&self, py: Python<'_>) -> PyResult<Option<Py<PyTexture>>> {
        self.texture(py, 4)
    }

    #[getter]
    fn surface_shader(&self, py: Python<'_>) -> PyResult<Option<Py<PySurfaceShader>>> {
        self.material
            .surface_shader
            .as_ref()
            .map(|shader| {
                Py::new(
                    py,
                    PySurfaceShader {
                        shader: shader.clone(),
                    },
                )
            })
            .transpose()
    }
}

impl PyPbrMaterial {
    fn texture(&self, py: Python<'_>, slot: usize) -> PyResult<Option<Py<PyTexture>>> {
        self.material.textures[slot]
            .as_ref()
            .map(|texture| {
                Py::new(
                    py,
                    PyTexture {
                        texture: texture.clone(),
                    },
                )
            })
            .transpose()
    }
}

// -- render objects ---------------------------------------------------------------

/// The base of render objects: identity, local transform and alpha.
/// Geometry and style live on leaves such as `RShape2D`; groups only
/// arrange their children. Render objects carry no names: names belong
/// to the scene objects they are snapshots of.
#[pyclass(module = "manim._native", name = "RNode", subclass, frozen)]
pub struct PyRNode {
    pub node: Arc<RNode>,
}

/// The node behind any render object (an instance of a subclass of RNode).
pub fn node_of(object: &Bound<'_, PyAny>) -> PyResult<Arc<RNode>> {
    match object.cast::<PyRNode>() {
        Ok(node) => Ok(node.get().node.clone()),
        Err(_) => Err(PyTypeError::new_err(format!(
            "expected a render object (RNode), got {}",
            object.get_type().name()?
        ))),
    }
}

#[pymethods]
impl PyRNode {
    /// Unique per scene object; every snapshot of an object has its uid.
    #[getter]
    fn uid(&self) -> u64 {
        self.node.uid
    }

    #[getter]
    fn transform<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        transform_to_py(py, &self.node.transform)
    }

    /// Multiplies the opacity of this node and its whole subtree.
    #[getter]
    fn alpha(&self) -> f64 {
        self.node.alpha
    }

    /// The version of a value field (`points`, `fill`, `stroke`, `mesh`,
    /// `material`, or the particles' `positions`, `rotations`, `scales`,
    /// `colors`): equal versions promise equal values, so a renderer can
    /// keep what it built from a value while its version holds.
    fn version(&self, field: &str) -> PyResult<u64> {
        let path = |path: &Path| match field {
            "points" => Some(path.points.version),
            "stroke" => Some(path.stroke.version),
            _ => None,
        };
        let surface = |surface: &Surface| match field {
            "mesh" => Some(surface.mesh.version),
            "material" => Some(surface.material.version),
            _ => None,
        };
        let version = match &self.node.kind {
            Kind::Shape2D { fill, .. } if field == "fill" => Some(fill.version),
            Kind::Shape2D { path: p, .. } | Kind::Curve3D { path: p } => path(p),
            Kind::Particles { instances, .. }
                if matches!(field, "positions" | "rotations" | "scales" | "colors") =>
            {
                Some(instances.version)
            }
            Kind::Shape3D(s) | Kind::Particles { surface: s, .. } => surface(s),
            _ => None,
        };
        version.ok_or_else(|| PyValueError::new_err(format!("no versioned field {field:?}")))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "{}(uid={})",
            slf.get_type().name()?,
            slf.get().node.uid
        ))
    }
}

/// The Python object of `node`, of the class of its kind.
pub fn wrap(py: Python<'_>, node: Arc<RNode>) -> PyResult<Py<PyAny>> {
    let base = PyClassInitializer::from(PyRNode { node: node.clone() });
    let object = match &node.kind {
        Kind::Group { .. } => Py::new(py, base.add_subclass(PyRGroup))?.into_any(),
        Kind::ClippingGroup { .. } => Py::new(py, base.add_subclass(PyRClippingGroup))?.into_any(),
        Kind::Shape2D { .. } => Py::new(
            py,
            base.add_subclass(PyRBezierPath).add_subclass(PyRShape2D),
        )?
        .into_any(),
        Kind::Curve3D { .. } => Py::new(
            py,
            base.add_subclass(PyRBezierPath).add_subclass(PyRCurve3D),
        )?
        .into_any(),
        Kind::Shape3D(_) => Py::new(py, base.add_subclass(PyRShape3D))?.into_any(),
        Kind::Particles { .. } => {
            Py::new(py, base.add_subclass(PyRShape3D).add_subclass(PyRParticles))?.into_any()
        }
        Kind::Light(light) => {
            let base = base.add_subclass(PyRLight);
            match light.kind {
                LightKind::Directional => {
                    Py::new(py, base.add_subclass(PyRDirectionalLight))?.into_any()
                }
                LightKind::Point => Py::new(py, base.add_subclass(PyRPointLight))?.into_any(),
                LightKind::Spot => Py::new(py, base.add_subclass(PyRSpotLight))?.into_any(),
            }
        }
    };
    Ok(object)
}

fn new_node(
    transform: Option<&Bound<'_, PyAny>>,
    alpha: f64,
    uid: Option<u64>,
    kind: Kind,
) -> PyResult<PyClassInitializer<PyRNode>> {
    Ok(PyClassInitializer::from(PyRNode {
        node: Arc::new(RNode {
            uid: uid.unwrap_or_else(new_uid),
            transform: transform_from_py(transform)?,
            alpha,
            kind,
        }),
    }))
}

/// Invisible container: no geometry or style, only children. Its
/// transform and alpha apply to the whole subtree.
#[pyclass(module = "manim._native", name = "RGroup", extends = PyRNode, frozen)]
pub struct PyRGroup;

#[pymethods]
impl PyRGroup {
    #[new]
    #[pyo3(signature = (transform=None, *, alpha=1.0, uid=None, children=Vec::new()))]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        children: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let children = children.iter().map(node_of).collect::<PyResult<Vec<_>>>()?;
        Ok(new_node(transform, alpha, uid, Kind::Group { children })?.add_subclass(PyRGroup))
    }

    #[getter]
    fn children<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
        children_of(slf.as_any())
    }
}

fn children_of<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let py = object.py();
    let node = node_of(object)?;
    let children = node
        .children()
        .into_iter()
        .map(|child| wrap(py, child.clone()))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, children)
}

/// Invisible container that shows `content` only inside `mask`. The mask
/// is never drawn itself; its alpha (fill and stroke as they would be
/// drawn) multiplies the content's alpha. With `one_sided=True` a mask
/// shape only counts where its front side (its local +z axis) faces the
/// camera, which turns flat masks into portals. Lights inside `content`
/// only light objects inside this group, so every clipped world can have
/// its own lighting.
#[pyclass(module = "manim._native", name = "RClippingGroup", extends = PyRNode, frozen)]
pub struct PyRClippingGroup;

#[pymethods]
impl PyRClippingGroup {
    #[new]
    #[pyo3(signature = (transform=None, *, alpha=1.0, uid=None, mask=None, content=None, one_sided=false))]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        mask: Option<&Bound<'_, PyAny>>,
        content: Option<&Bound<'_, PyAny>>,
        one_sided: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let kind = Kind::ClippingGroup {
            mask: mask.map(node_of).transpose()?,
            content: content.map(node_of).transpose()?,
            one_sided,
        };
        Ok(new_node(transform, alpha, uid, kind)?.add_subclass(PyRClippingGroup))
    }

    #[getter]
    fn mask(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let Kind::ClippingGroup { mask, .. } = &node_of(slf.as_any())?.kind else {
            unreachable!()
        };
        mask.clone().map(|node| wrap(slf.py(), node)).transpose()
    }

    #[getter]
    fn content(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let Kind::ClippingGroup { content, .. } = &node_of(slf.as_any())?.kind else {
            unreachable!()
        };
        content.clone().map(|node| wrap(slf.py(), node)).transpose()
    }

    #[getter]
    fn one_sided(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let Kind::ClippingGroup { one_sided, .. } = node_of(slf.as_any())?.kind else {
            unreachable!()
        };
        Ok(one_sided)
    }

    /// The mask and the content (those that are set).
    #[getter]
    fn children<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
        children_of(slf.as_any())
    }
}

/// The base of paths of cubic Bezier curves: `points` (four per curve:
/// anchor, handle, handle, anchor; a new subpath starts wherever a curve
/// starts away from where the last ended) and a stroke.
#[pyclass(module = "manim._native", name = "RBezierPath", extends = PyRNode, subclass, frozen)]
pub struct PyRBezierPath;

fn with_path<T>(node: &RNode, read: impl FnOnce(&Path) -> T) -> T {
    match &node.kind {
        Kind::Shape2D { path, .. } | Kind::Curve3D { path } => read(path),
        _ => unreachable!("only paths are RBezierPaths"),
    }
}

#[pymethods]
impl PyRBezierPath {
    /// `(N, 2)` for an RShape2D, `(N, 3)` for an RCurve3D; read-only.
    #[getter]
    fn points<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let node = node_of(slf.as_any())?;
        with_path(&node, |path| {
            let points = &path.points;
            read_only(
                slf.py(),
                points.data.clone(),
                &[points.count(), points.dimension],
            )
        })
    }

    #[getter]
    fn stroke(slf: &Bound<'_, Self>) -> PyResult<Py<PyStroke>> {
        let node = node_of(slf.as_any())?;
        let stroke = with_path(&node, |path| path.stroke.clone());
        Py::new(slf.py(), PyStroke { stroke })
    }

    /// Whether this path blocks light for lit meshes behind it.
    #[getter]
    fn cast_shadows(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let node = node_of(slf.as_any())?;
        Ok(with_path(&node, |path| path.cast_shadows))
    }
}

/// `points` as a path of `dimension` numbers per point: an
/// `(N, dimension)` array with N a multiple of four.
fn points_from_py(points: Option<&Bound<'_, PyAny>>, dimension: usize) -> PyResult<Points> {
    let Some(points) = points else {
        return Ok(Points::new(Vec::new(), dimension));
    };
    let (shape, data) = array_of::<f64>(points, "float64")?;
    if data.is_empty() {
        return Ok(Points::new(Vec::new(), dimension));
    }
    if shape.len() != 2 || shape[1] != dimension {
        return Err(PyValueError::new_err(format!(
            "points must have shape (N, {dimension}), got {}",
            shape_text(&shape)
        )));
    }
    if !shape[0].is_multiple_of(POINTS_PER_CURVE) {
        return Err(PyValueError::new_err(format!(
            "points need four entries (anchor, handle, handle, anchor) per cubic Bezier curve, got {}",
            shape[0]
        )));
    }
    Ok(Points::new(data, dimension))
}

fn path_from_py(
    points: Option<&Bound<'_, PyAny>>,
    stroke: Option<Bound<'_, PyStroke>>,
    cast_shadows: bool,
    dimension: usize,
) -> PyResult<Path> {
    let stroke = match stroke {
        Some(stroke) => stroke.get().stroke.clone(),
        None => Arc::new(Stroke::new(
            vec![[1.0; 3]],
            vec![1.0],
            vec![4.0],
            vec![0],
            0,
        )),
    };
    Ok(Path {
        points: Arc::new(points_from_py(points, dimension)?),
        stroke,
        cast_shadows,
    })
}

/// A drawable vector shape: `(N, 2)` points in the local XY plane, which
/// the transform places in 3D; the stroke is a flat ribbon in that plane.
/// Holes and disjoint pieces (a glyph) are subpaths of one shape.
#[pyclass(module = "manim._native", name = "RShape2D", extends = PyRBezierPath, frozen)]
pub struct PyRShape2D;

#[pymethods]
impl PyRShape2D {
    #[new]
    #[pyo3(signature = (transform=None, *, alpha=1.0, uid=None, points=None, stroke=None, cast_shadows=true, fill=None))]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        points: Option<&Bound<'_, PyAny>>,
        stroke: Option<Bound<'_, PyStroke>>,
        cast_shadows: bool,
        fill: Option<Bound<'_, PyFill>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let fill = match fill {
            Some(fill) => fill.get().fill.clone(),
            None => Arc::new(Fill::new(vec![[1.0; 3]], vec![0.0])),
        };
        let kind = Kind::Shape2D {
            path: path_from_py(points, stroke, cast_shadows, 2)?,
            fill,
        };
        Ok(new_node(transform, alpha, uid, kind)?
            .add_subclass(PyRBezierPath)
            .add_subclass(PyRShape2D))
    }

    #[getter]
    fn fill(slf: &Bound<'_, Self>) -> PyResult<Py<PyFill>> {
        let Kind::Shape2D { fill, .. } = &node_of(slf.as_any())?.kind else {
            unreachable!()
        };
        Py::new(slf.py(), PyFill { fill: fill.clone() })
    }
}

/// A stroke-only path of cubic Bezier curves in 3D: `(N, 3)` points. Its
/// ribbon always faces the camera, and its width is in world units, so
/// under a perspective camera it gets thinner with distance.
#[pyclass(module = "manim._native", name = "RCurve3D", extends = PyRBezierPath, frozen)]
pub struct PyRCurve3D;

#[pymethods]
impl PyRCurve3D {
    #[new]
    #[pyo3(signature = (transform=None, *, alpha=1.0, uid=None, points=None, stroke=None, cast_shadows=true))]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        points: Option<&Bound<'_, PyAny>>,
        stroke: Option<Bound<'_, PyStroke>>,
        cast_shadows: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let kind = Kind::Curve3D {
            path: path_from_py(points, stroke, cast_shadows, 3)?,
        };
        Ok(new_node(transform, alpha, uid, kind)?
            .add_subclass(PyRBezierPath)
            .add_subclass(PyRCurve3D))
    }
}

fn surface_from_py(
    mesh: Option<Bound<'_, PyMeshData>>,
    material: Option<Bound<'_, PyPbrMaterial>>,
    cast_shadows: bool,
) -> PyResult<Surface> {
    let mesh = match mesh {
        Some(mesh) => mesh.get().mesh.clone(),
        None => Arc::new(
            Mesh::new(Vec::new(), Vec::new(), Vec::new(), None, None)
                .expect("an empty mesh is valid"),
        ),
    };
    let material = match material {
        Some(material) => material.get().material.clone(),
        None => Arc::new(Material::default()),
    };
    Ok(Surface {
        mesh,
        material,
        cast_shadows,
    })
}

fn surface_of<T>(node: &RNode, read: impl FnOnce(&Surface) -> T) -> T {
    match &node.kind {
        Kind::Shape3D(surface) | Kind::Particles { surface, .. } => read(surface),
        _ => unreachable!("only meshes are RShape3Ds"),
    }
}

/// A lit triangle mesh with a physically based material. `mesh` is in
/// local coordinates; the transform places it.
#[pyclass(module = "manim._native", name = "RShape3D", extends = PyRNode, subclass, frozen)]
pub struct PyRShape3D;

#[pymethods]
impl PyRShape3D {
    #[new]
    #[pyo3(signature = (transform=None, *, alpha=1.0, uid=None, mesh=None, material=None, cast_shadows=true))]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        mesh: Option<Bound<'_, PyMeshData>>,
        material: Option<Bound<'_, PyPbrMaterial>>,
        cast_shadows: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let kind = Kind::Shape3D(surface_from_py(mesh, material, cast_shadows)?);
        Ok(new_node(transform, alpha, uid, kind)?.add_subclass(PyRShape3D))
    }

    #[getter]
    fn mesh(slf: &Bound<'_, Self>) -> PyResult<Py<PyMeshData>> {
        let mesh = surface_of(node_of(slf.as_any())?.as_ref(), |s| s.mesh.clone());
        Py::new(slf.py(), PyMeshData { mesh })
    }

    #[getter]
    fn material(slf: &Bound<'_, Self>) -> PyResult<Py<PyPbrMaterial>> {
        let material = surface_of(node_of(slf.as_any())?.as_ref(), |s| s.material.clone());
        Py::new(slf.py(), PyPbrMaterial { material })
    }

    /// Whether this mesh blocks light (it always receives shadows).
    #[getter]
    fn cast_shadows(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(surface_of(node_of(slf.as_any())?.as_ref(), |s| {
            s.cast_shadows
        }))
    }
}

/// Many copies of one mesh, drawn as one instanced draw. Particle `i` is
/// the mesh placed by `positions[i]`, `rotations[i]` (a quaternion w, x,
/// y, z) and `scales[i]` in the object's local space; `colors[i]` is r,
/// g, b, glow: the color multiplies the material's base and emissive
/// color (in linear light), the glow its emissive strength. A particle
/// scaled to nothing on any axis is not drawn.
#[pyclass(module = "manim._native", name = "RParticles", extends = PyRShape3D, frozen)]
pub struct PyRParticles;

#[pymethods]
impl PyRParticles {
    #[new]
    #[pyo3(signature = (
        transform=None, *, alpha=1.0, uid=None, mesh=None, material=None, cast_shadows=true,
        positions=None, rotations=None, scales=None, colors=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        mesh: Option<Bound<'_, PyMeshData>>,
        material: Option<Bound<'_, PyPbrMaterial>>,
        cast_shadows: bool,
        positions: Option<&Bound<'_, PyAny>>,
        rotations: Option<&Bound<'_, PyAny>>,
        scales: Option<&Bound<'_, PyAny>>,
        colors: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        fn optional<const N: usize>(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<[f64; N]>> {
            value.map_or(Ok(Vec::new()), |v| rows::<f64, N>(v, "float64"))
        }
        let instances = Instances::new(
            optional::<3>(positions)?,
            optional::<4>(rotations)?,
            optional::<3>(scales)?,
            optional::<4>(colors)?,
        );
        let kind = Kind::Particles {
            surface: surface_from_py(mesh, material, cast_shadows)?,
            instances: Arc::new(instances),
        };
        Ok(new_node(transform, alpha, uid, kind)?
            .add_subclass(PyRShape3D)
            .add_subclass(PyRParticles))
    }

    #[getter]
    fn positions<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let i = instances_of(slf.as_any())?;
        read_only(slf.py(), flat(&i.positions), &[i.positions.len(), 3])
    }

    #[getter]
    fn rotations<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let i = instances_of(slf.as_any())?;
        read_only(slf.py(), flat(&i.rotations), &[i.rotations.len(), 4])
    }

    #[getter]
    fn scales<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let i = instances_of(slf.as_any())?;
        read_only(slf.py(), flat(&i.scales), &[i.scales.len(), 3])
    }

    #[getter]
    fn colors<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let i = instances_of(slf.as_any())?;
        read_only(slf.py(), flat(&i.colors), &[i.colors.len(), 4])
    }
}

fn instances_of(object: &Bound<'_, PyAny>) -> PyResult<Arc<Instances>> {
    let Kind::Particles { instances, .. } = &node_of(object)?.kind else {
        unreachable!()
    };
    Ok(instances.clone())
}

/// A light source: a leaf of the tree like a shape, so it moves with its
/// groups, but it draws nothing. Only meshes are lit; shapes and curves
/// keep their flat colors. Lights with a direction shine along their
/// local -z axis (the glTF and Blender convention).
///
/// `color` is sRGB `(3,)`. `intensity` scales the radiance: a white
/// diffuse surface lit head-on by a directional light of intensity pi
/// appears pure white. `cast_shadows` turns this light's shadows on
/// (RenderConfig.shadows must be on too). `size` is how large the light
/// appears, which sets how soft its shadows get away from the caster: the
/// radius for point and spot lights, the angular diameter in radians for
/// directional lights. A light inside an `RClippingGroup`'s content only
/// lights, and only gets shadows from, objects in that group.
#[pyclass(module = "manim._native", name = "RLight", extends = PyRNode, subclass, frozen)]
pub struct PyRLight;

fn light_of(object: &Bound<'_, PyAny>) -> PyResult<Light> {
    let Kind::Light(light) = node_of(object)?.kind else {
        unreachable!()
    };
    Ok(light)
}

#[pymethods]
impl PyRLight {
    /// sRGB `(r, g, b)`.
    #[getter]
    fn color(slf: &Bound<'_, Self>) -> PyResult<(f64, f64, f64)> {
        let [r, g, b] = light_of(slf.as_any())?.color;
        Ok((r, g, b))
    }

    #[getter]
    fn intensity(slf: &Bound<'_, Self>) -> PyResult<f64> {
        Ok(light_of(slf.as_any())?.intensity)
    }

    #[getter]
    fn cast_shadows(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(light_of(slf.as_any())?.cast_shadows)
    }

    #[getter]
    fn size(slf: &Bound<'_, Self>) -> PyResult<f64> {
        Ok(light_of(slf.as_any())?.size)
    }
}

#[allow(clippy::too_many_arguments)]
fn new_light(
    transform: Option<&Bound<'_, PyAny>>,
    alpha: f64,
    uid: Option<u64>,
    kind: LightKind,
    light_color: Option<&Bound<'_, PyAny>>,
    intensity: f64,
    cast_shadows: bool,
    size: f64,
    range: Option<f64>,
    cone: Option<(f64, f64)>,
) -> PyResult<PyClassInitializer<PyRLight>> {
    let light = Light {
        kind,
        color: color(light_color, [1.0; 3])?,
        intensity,
        size,
        cast_shadows,
        range,
        cone,
    };
    Ok(new_node(transform, alpha, uid, Kind::Light(light))?.add_subclass(PyRLight))
}

/// Parallel light (like the sun). Its position does not matter.
#[pyclass(module = "manim._native", name = "RDirectionalLight", extends = PyRLight, frozen)]
pub struct PyRDirectionalLight;

#[pymethods]
impl PyRDirectionalLight {
    #[new]
    #[pyo3(signature = (transform=None, *, alpha=1.0, uid=None, color=None, intensity=3.0, cast_shadows=true, size=0.03))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        color: Option<&Bound<'_, PyAny>>,
        intensity: f64,
        cast_shadows: bool,
        size: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(new_light(
            transform,
            alpha,
            uid,
            LightKind::Directional,
            color,
            intensity,
            cast_shadows,
            size,
            None,
            None,
        )?
        .add_subclass(PyRDirectionalLight))
    }
}

/// Light emitted in all directions from the light's position. Falls off
/// with the inverse square of the distance; `range` (if set) smoothly
/// fades it to zero there.
#[pyclass(module = "manim._native", name = "RPointLight", extends = PyRLight, frozen)]
pub struct PyRPointLight;

#[pymethods]
impl PyRPointLight {
    #[new]
    #[pyo3(signature = (transform=None, *, alpha=1.0, uid=None, color=None, intensity=20.0, cast_shadows=true, size=0.1, range=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        color: Option<&Bound<'_, PyAny>>,
        intensity: f64,
        cast_shadows: bool,
        size: f64,
        range: Option<f64>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(new_light(
            transform,
            alpha,
            uid,
            LightKind::Point,
            color,
            intensity,
            cast_shadows,
            size,
            range,
            None,
        )?
        .add_subclass(PyRPointLight))
    }

    #[getter]
    fn range(slf: &Bound<'_, Self>) -> PyResult<Option<f64>> {
        Ok(light_of(slf.as_any())?.range)
    }
}

/// A point light restricted to a cone around its local -z axis: full
/// intensity inside `inner_cone_angle`, fading smoothly to zero at
/// `outer_cone_angle` (both in radians from the axis, as in glTF).
#[pyclass(module = "manim._native", name = "RSpotLight", extends = PyRLight, frozen)]
pub struct PyRSpotLight;

#[pymethods]
impl PyRSpotLight {
    #[new]
    #[pyo3(signature = (
        transform=None, *, alpha=1.0, uid=None, color=None, intensity=40.0, cast_shadows=true,
        size=0.1, range=None, inner_cone_angle=0.0, outer_cone_angle=FRAC_PI_4,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        transform: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        uid: Option<u64>,
        color: Option<&Bound<'_, PyAny>>,
        intensity: f64,
        cast_shadows: bool,
        size: f64,
        range: Option<f64>,
        inner_cone_angle: f64,
        outer_cone_angle: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let valid = 0.0 <= inner_cone_angle
            && inner_cone_angle < outer_cone_angle
            && outer_cone_angle <= std::f64::consts::FRAC_PI_2;
        if !valid {
            return Err(PyValueError::new_err(
                "cone angles need 0 <= inner < outer <= pi / 2",
            ));
        }
        Ok(new_light(
            transform,
            alpha,
            uid,
            LightKind::Spot,
            color,
            intensity,
            cast_shadows,
            size,
            range,
            Some((inner_cone_angle, outer_cone_angle)),
        )?
        .add_subclass(PyRSpotLight))
    }

    #[getter]
    fn range(slf: &Bound<'_, Self>) -> PyResult<Option<f64>> {
        Ok(light_of(slf.as_any())?.range)
    }

    #[getter]
    fn inner_cone_angle(slf: &Bound<'_, Self>) -> PyResult<f64> {
        Ok(light_of(slf.as_any())?.cone.map_or(0.0, |cone| cone.0))
    }

    #[getter]
    fn outer_cone_angle(slf: &Bound<'_, Self>) -> PyResult<f64> {
        Ok(light_of(slf.as_any())?
            .cone
            .map_or(FRAC_PI_4, |cone| cone.1))
    }
}

/// A fresh process-unique uid. Scene objects get one when they are made
/// and pass it to every render object built for them, so all snapshots of
/// an object share its identity.
#[pyfunction(name = "new_uid")]
pub fn py_new_uid() -> u64 {
    new_uid()
}

/// Add the render object classes to `module`.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRNode>()?;
    module.add_class::<PyRGroup>()?;
    module.add_class::<PyRClippingGroup>()?;
    module.add_class::<PyRBezierPath>()?;
    module.add_class::<PyRShape2D>()?;
    module.add_class::<PyRCurve3D>()?;
    module.add_class::<PyRShape3D>()?;
    module.add_class::<PyRParticles>()?;
    module.add_class::<PyRLight>()?;
    module.add_class::<PyRDirectionalLight>()?;
    module.add_class::<PyRPointLight>()?;
    module.add_class::<PyRSpotLight>()?;
    module.add_class::<PyFill>()?;
    module.add_class::<PyStroke>()?;
    module.add_class::<PyMeshData>()?;
    module.add_class::<PyPbrMaterial>()?;
    module.add_class::<PyTexture>()?;
    module.add_class::<PySurfaceShader>()?;
    module.add_function(wrap_pyfunction!(py_new_uid, module)?)?;
    Ok(())
}
