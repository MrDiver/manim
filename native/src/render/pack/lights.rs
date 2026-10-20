//! The lights of a frame and their GPU records (the `Light` struct of
//! lighting.wgsl), and the default light of a scene without any.

use glam::{DMat4, DQuat, DVec3};

use crate::flat::{FlatScene, NO_CLIP};
use crate::render::pack::color::linear_rgb;
use crate::render::packet::records::{self, Light};
use crate::robjects::{Kind, LightKind};

/// One light as the renderer needs it (`LightItem` with its `RLight`).
pub struct LightSource {
    pub light: LightKind,
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
            LightKind::Directional => (records::LIGHT_DIRECTIONAL, 0.0, 0.0),
            LightKind::Point => (records::LIGHT_POINT, 0.0, 0.0),
            LightKind::Spot => (records::LIGHT_SPOT, self.cone.0.cos(), self.cone.1.cos()),
        };
        Light {
            position: [p.x as f32, p.y as f32, p.z as f32],
            kind,
            direction: [d.x as f32, d.y as f32, d.z as f32],
            range: if matches!(self.light, LightKind::Directional) {
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

/// The scene's lights, or the default light when it has none and
/// `default_lights` is on (`camera` is the camera's world matrix).
pub fn light_sources(scene: &FlatScene, default_lights: bool, camera: &DMat4) -> Vec<LightSource> {
    let mut sources = Vec::with_capacity(scene.lights.len());
    for item in &scene.lights {
        let Kind::Light(light) = &item.node.kind else {
            unreachable!()
        };
        sources.push(LightSource {
            light: light.kind,
            world: item.world,
            clip: item.clip,
            alpha: item.alpha,
            color_linear: linear_rgb(light.color),
            intensity: light.intensity,
            size: light.size,
            cast_shadows: light.cast_shadows,
            range: light.range.unwrap_or(0.0),
            cone: light.cone.unwrap_or((0.0, 0.0)),
        });
    }
    if sources.is_empty() && default_lights {
        sources.push(default_light(camera));
    }
    sources
}

/// The key light of a scene without lights: a white directional light of
/// intensity 3 from above, to the right of and behind the camera, so
/// meshes are lit from the front.
fn default_light(camera: &DMat4) -> LightSource {
    let right = camera.x_axis.truncate();
    let up = camera.y_axis.truncate();
    let backwards = camera.z_axis.truncate();
    let direction = -(up - 0.6 * right + 0.8 * backwards);
    LightSource {
        light: LightKind::Directional,
        world: DMat4::from_quat(rotation_towards(direction)),
        clip: NO_CLIP,
        alpha: 1.0,
        color_linear: [1.0; 3],
        intensity: 3.0,
        size: 0.03,
        cast_shadows: true,
        range: 0.0,
        cone: (0.0, 0.0),
    }
}

/// The rotation turning local -z onto `direction` along the shortest arc.
fn rotation_towards(direction: DVec3) -> DQuat {
    let target = direction.normalize();
    let forward = DVec3::NEG_Z;
    let axis = forward.cross(target);
    let cosine = forward.dot(target);
    if axis.length() < 1e-9 {
        // Already aligned, or exactly opposite: turn half way around x.
        return if cosine > 0.0 {
            DQuat::IDENTITY
        } else {
            DQuat::from_xyzw(1.0, 0.0, 0.0, 0.0)
        };
    }
    let angle = axis.length().atan2(cosine);
    DQuat::from_axis_angle(axis.normalize(), angle)
}
