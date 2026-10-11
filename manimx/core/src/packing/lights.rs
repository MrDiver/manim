//! The lights of a frame and their GPU records (the `Light` struct of
//! lighting.wgsl), and the default lights of a scene without any. The
//! fields come from the light nodes (manimx/animation/nodes/light.py).

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;

use glam::{DMat4, DVec3};
use pyo3::prelude::*;

use crate::packing::callbacks::Callbacks;
use crate::packing::records::{self, Light};
use crate::packing::scene_graph::{FlatScene, LightType, NodeData, SceneGraph, Values, NO_CLIP};

/// One light as the renderer needs it (`LightItem` with its `RLight`).
pub struct LightSource {
    pub light: LightType,
    pub world: DMat4,
    pub clip: i32,
    pub alpha: f64,
    pub color_linear: [f64; 3],
    pub intensity: f64,
    pub size: f64,
    pub cast_shadows: bool,
    /// 0 for none.
    pub range: f64,
    /// inner, outer (spot lights).
    pub cone: (f64, f64),
}

impl LightSource {
    pub fn position(&self) -> DVec3 {
        self.world.w_axis.truncate()
    }

    /// Lights shine along their local -z axis.
    pub fn direction(&self) -> DVec3 {
        let z = self.world.z_axis.truncate();
        -z / z.length()
    }

    /// `light_array` for one light (its shadow view is assigned later).
    pub fn record(&self) -> Light {
        let p = self.position();
        let d = self.direction();
        let strength = self.intensity * self.alpha;
        let c = self.color_linear;
        let (kind, inner, outer) = match self.light {
            LightType::Directional => (records::LIGHT_DIRECTIONAL, 0.0, 0.0),
            LightType::Point => (records::LIGHT_POINT, 0.0, 0.0),
            LightType::Spot => (records::LIGHT_SPOT, self.cone.0.cos(), self.cone.1.cos()),
        };
        Light {
            position: [p.x as f32, p.y as f32, p.z as f32],
            kind,
            direction: [d.x as f32, d.y as f32, d.z as f32],
            range: if matches!(self.light, LightType::Directional) {
                0.0
            } else {
                self.range as f32
            },
            color: [
                (c[0] * strength) as f32,
                (c[1] * strength) as f32,
                (c[2] * strength) as f32,
            ],
            inner_cone_cos: inner as f32,
            outer_cone_cos: outer as f32,
            shadow_view: -1,
            size: self.size as f32,
            scope: self.clip,
        }
    }
}

/// Linear colors of lights, per light uid with the color's version.
#[derive(Default)]
pub struct LightColors {
    colors: HashMap<u64, (u64, [f64; 3])>,
}

impl LightColors {
    fn linear(
        &mut self,
        py: Python<'_>,
        callbacks: &Callbacks,
        values: &Values,
        uid: u64,
        slot: usize,
    ) -> PyResult<[f64; 3]> {
        let version = values.state.versions[slot];
        if let Some((cached, color)) = self.colors.get(&uid) {
            if *cached == version {
                return Ok(*color);
            }
        }
        let color = values.state.kinds[slot].value_to_py(py, values.get(slot))?;
        let (r, g, b): (f64, f64, f64) =
            callbacks.light_color.bind(py).call1((color,))?.extract()?;
        self.colors.insert(uid, (version, [r, g, b]));
        Ok([r, g, b])
    }
}

/// The scene's lights, or the default light when it has none and
/// `default_lights` is on (`camera` is the camera's world matrix).
#[allow(clippy::too_many_arguments)]
pub fn light_sources(
    py: Python<'_>,
    graph: &SceneGraph,
    values: &Values,
    scene: &FlatScene,
    colors: &mut LightColors,
    callbacks: &Callbacks,
    default_lights: bool,
    camera: &DMat4,
) -> PyResult<Vec<LightSource>> {
    let mut sources = Vec::with_capacity(scene.lights.len());
    for item in &scene.lights {
        let node = &graph.nodes[item.node];
        let NodeData::Light(fields) = &node.data else {
            unreachable!()
        };
        // `_range`: a range of 0 or less means none.
        let range = match fields.range {
            Some(slot) => values.scalar(slot)?.max(0.0),
            None => 0.0,
        };
        let cone = match fields.cone {
            Some((inner, outer)) => {
                let outer = values.scalar(outer)?.clamp(1e-6, FRAC_PI_2);
                let inner = values.scalar(inner)?.max(0.0).min(outer * (1.0 - 1e-6));
                (inner, outer)
            }
            None => (0.0, 0.0),
        };
        sources.push(LightSource {
            light: fields.light,
            world: item.world,
            clip: item.clip,
            alpha: item.alpha,
            color_linear: colors.linear(py, callbacks, values, node.uid, fields.color)?,
            intensity: values.scalar(fields.intensity)?.max(0.0),
            size: values.scalar(fields.size)?.max(0.0),
            cast_shadows: values.truthy(fields.cast_shadows)?,
            range,
            cone,
        });
    }
    if sources.is_empty() && default_lights {
        sources.push(default_light(py, callbacks, camera)?);
    }
    Ok(sources)
}

fn default_light(py: Python<'_>, callbacks: &Callbacks, camera: &DMat4) -> PyResult<LightSource> {
    let numpy = py.import("numpy")?;
    let rows: Vec<Vec<f64>> = (0..4)
        .map(|row| camera.row(row).to_array().to_vec())
        .collect();
    let matrix = numpy.call_method1("array", (rows,))?;
    let result = callbacks.default_light.bind(py).call1((matrix,))?;
    let world: Vec<f64> = result.get_item(0)?.extract()?;
    let color: (f64, f64, f64) = result.get_item(1)?.extract()?;
    Ok(LightSource {
        light: LightType::Directional,
        world: DMat4::from_cols_array(&world.try_into().unwrap()).transpose(),
        clip: NO_CLIP,
        alpha: 1.0,
        color_linear: [color.0, color.1, color.2],
        intensity: result.get_item(2)?.extract()?,
        size: result.get_item(3)?.extract()?,
        cast_shadows: result.get_item(4)?.extract()?,
        range: 0.0,
        cone: (0.0, 0.0),
    })
}
