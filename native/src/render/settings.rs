//! What the renderer reads from a `RenderConfig` (manim/renderer/protocol.py)
//! besides the render object tree: the camera, the camera views, and the
//! lighting and shadow settings of one view. The config says what is
//! meant (a color, a method, sample counts); what the shaders need from
//! it (linear light, the packed shadow filter) is worked out here.

use glam::{DMat4, DVec3};

use crate::render::pack::color::linear_rgb;
use crate::render::pack::packer::PackSettings;
use crate::render::pack::shadows::{frame_record, FrameSettings};
use crate::render::pack::vectors::CameraViews;
use crate::render::packet::records::Frame;
use crate::robjects::Transform;

/// A camera looking along its local -z (`Camera`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub transform: Transform,
    /// World units shown vertically by an orthographic camera.
    pub frame_height: f64,
    /// Vertical field of view in radians; None for orthographic.
    pub field_of_view: Option<f64>,
    pub near: f64,
    pub far: f64,
    /// Depth of field: the distance along the view axis that is sharp.
    pub focus_distance: f64,
    /// Depth of field: the blur of something infinitely far, as a fraction
    /// of the image height (0: everything sharp).
    pub aperture: f64,
}

impl Camera {
    /// World to view space (the inverse of the camera's placement).
    pub fn view_matrix(&self) -> DMat4 {
        self.transform.matrix().inverse()
    }

    /// View to clip space with WebGPU's depth range [0, 1].
    pub fn projection_matrix(&self, aspect_ratio: f64) -> DMat4 {
        let (near, far) = (self.near, self.far);
        let rows = match self.field_of_view {
            None => {
                let half_height = self.frame_height / 2.0;
                let half_width = half_height * aspect_ratio;
                [
                    [1.0 / half_width, 0.0, 0.0, 0.0],
                    [0.0, 1.0 / half_height, 0.0, 0.0],
                    [0.0, 0.0, -1.0 / (far - near), -near / (far - near)],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
            Some(field_of_view) => {
                let focal = 1.0 / (field_of_view / 2.0).tan();
                [
                    [focal / aspect_ratio, 0.0, 0.0, 0.0],
                    [0.0, focal, 0.0, 0.0],
                    [0.0, 0.0, far / (near - far), near * far / (near - far)],
                    [0.0, 0.0, -1.0, 0.0],
                ]
            }
        };
        DMat4::from_cols_array_2d(&rows).transpose()
    }

    fn fingerprint(&self, into: &mut Vec<u64>) {
        let t = &self.transform;
        into.extend(t.position.to_array().map(f64::to_bits));
        into.extend(t.rotation.to_array().map(f64::to_bits));
        into.extend(t.scale.to_array().map(f64::to_bits));
        into.extend([
            self.frame_height.to_bits(),
            self.field_of_view.map_or(u64::MAX, f64::to_bits),
            self.near.to_bits(),
            self.far.to_bits(),
            self.focus_distance.to_bits(),
            self.aperture.to_bits(),
        ]);
    }
}

/// Another camera's picture, shown as the fill of the node `uid` (a shape
/// `width` by `height` world units, centered on its origin;
/// `CameraViewState`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    pub uid: u64,
    pub camera: Camera,
    pub width: f64,
    pub height: f64,
}

/// How one view is rendered: the `RenderConfig` values the renderer reads
/// per render (the pixel size and sample count are fixed per renderer).
#[derive(Clone, Debug, PartialEq)]
pub struct ViewSettings {
    pub camera: Camera,
    pub camera_views: Vec<CameraView>,
    /// Leave the shapes of `camera_views` out: set while a camera view's
    /// picture is rendered, so views never show views.
    pub hide_camera_views: bool,
    /// The width / height of the picture (None: of the pixels).
    pub view_aspect: Option<f64>,
    /// Moves the picture by this many pixels (right, up): each render of
    /// `RenderSamples` covers its pixels at another point.
    pub jitter: [f64; 2],
    /// Seconds, `frame.time` in shaders.
    pub time: f64,
    pub layer_depth_step: f64,
    pub exposure: f64,
    /// sRGB.
    pub ambient_color: [f64; 3],
    pub ambient_intensity: f64,
    /// Light meshes with a key light when the scene has no lights.
    pub default_lights: bool,
    pub shadows: bool,
    pub shadow_softness: f64,
    /// Taps per shadow lookup.
    pub shadow_samples: u32,
    pub shadow_resolution: u32,
    pub shadow_method: ShadowMethod,
    /// How many renders a frame averages; each takes a share of the taps.
    pub shadow_share: u32,
    /// Which of the `shadow_share` renders this is.
    pub shadow_index: u32,
}

/// How shadow edges are softened (`ShadowMethod`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowMethod {
    /// Edges widen with the distance to the caster (contact hardening).
    Pcss,
    /// A fixed narrow soft edge, much cheaper.
    Pcf,
}

/// The fewest taps a shadow lookup takes when renders share the taps.
pub const MIN_SHADOW_TAPS: u32 = 4;

impl ViewSettings {
    pub fn aspect_ratio(&self, size: (u32, u32)) -> f64 {
        self.view_aspect.unwrap_or(size.0 as f64 / size.1 as f64)
    }

    /// The ambient light in linear RGB, times its intensity.
    pub fn ambient_linear(&self) -> [f64; 3] {
        linear_rgb(self.ambient_color).map(|c| c * self.ambient_intensity)
    }

    /// `frame.shadow_filter` of the shaders: the method (1 PCSS, 0 PCF),
    /// the taps per lookup and the turn (0 to 1) of the tap pattern for
    /// this render. When the renders of a frame share the taps, each takes
    /// its share (at least `MIN_SHADOW_TAPS`), rotated differently per
    /// render, so the average still gets them all; a golden-ratio sequence
    /// spreads the renders' patterns evenly.
    pub fn shadow_filter(&self) -> [f32; 4] {
        let share = self.shadow_share.max(1);
        let taps = self
            .shadow_samples
            .min(MIN_SHADOW_TAPS.max(self.shadow_samples.div_ceil(share)));
        let turn = (self.shadow_index as f64 * 0.6180339887) % 1.0;
        let method = match self.shadow_method {
            ShadowMethod::Pcss => 1.0,
            ShadowMethod::Pcf => 0.0,
        };
        [method, taps as f32, turn as f32, 0.0]
    }

    pub fn frame_settings(&self) -> FrameSettings {
        FrameSettings {
            time: self.time,
            layer_depth_step: self.layer_depth_step,
            exposure: self.exposure,
            ambient_linear: self.ambient_linear(),
            shadow_softness: self.shadow_softness,
            shadow_samples: self.shadow_samples as f64,
            shadow_resolution: self.shadow_resolution,
            shadows: self.shadows,
            shadow_filter: self.shadow_filter(),
        }
    }

    /// How the packer packs this view, a picture of `size` pixels: camera
    /// view `i` is layer `i` of the camera view texture.
    pub fn pack_settings(&self, size: (u32, u32)) -> PackSettings {
        let camera_views: CameraViews = self
            .camera_views
            .iter()
            .enumerate()
            .map(|(layer, view)| {
                let scale = [(1.0 / view.width) as f32, (1.0 / view.height) as f32];
                (view.uid, (layer as i32, scale))
            })
            .collect();
        PackSettings {
            frame: self.frame_settings(),
            default_lights: self.default_lights,
            camera: self.camera.transform.matrix(),
            view_projection: self.camera.projection_matrix(self.aspect_ratio(size))
                * self.camera.view_matrix(),
            camera_views,
            hide_camera_views: self.hide_camera_views,
        }
    }

    /// The camera's Frame record for a picture of `size` pixels.
    pub fn camera_frame(&self, size: (u32, u32), light_count: u32) -> Frame {
        let camera = &self.camera;
        // A shift in normalized device coordinates (2 per picture width),
        // applied after the projection: x_clip + shift * w_clip.
        let shift = DVec3::new(
            2.0 * self.jitter[0] / size.0 as f64,
            2.0 * self.jitter[1] / size.1 as f64,
            0.0,
        );
        let projection =
            DMat4::from_translation(shift) * camera.projection_matrix(self.aspect_ratio(size));
        frame_record(
            &camera.view_matrix(),
            &projection,
            camera.transform.position,
            size,
            &self.frame_settings(),
            light_count,
        )
    }

    /// The settings that render camera view `view`'s picture.
    pub fn picture(&self, view: &CameraView) -> ViewSettings {
        ViewSettings {
            camera: view.camera,
            view_aspect: Some(view.width / view.height),
            hide_camera_views: true,
            ..self.clone()
        }
    }

    /// Equal fingerprints promise equal pictures of equal trees: every
    /// setting except the time.
    pub fn fingerprint(&self, into: &mut Vec<u64>) {
        self.camera.fingerprint(into);
        into.push(self.camera_views.len() as u64);
        for view in &self.camera_views {
            into.extend([view.uid, view.width.to_bits(), view.height.to_bits()]);
            view.camera.fingerprint(into);
        }
        into.extend([
            self.hide_camera_views as u64,
            self.view_aspect.map_or(u64::MAX, f64::to_bits),
            self.jitter[0].to_bits(),
            self.jitter[1].to_bits(),
            self.layer_depth_step.to_bits(),
            self.exposure.to_bits(),
            self.default_lights as u64,
            self.shadows as u64,
            self.shadow_softness.to_bits(),
            self.shadow_samples as u64,
            self.shadow_resolution as u64,
            self.shadow_method as u64,
            self.shadow_share as u64,
            self.shadow_index as u64,
            self.ambient_intensity.to_bits(),
        ]);
        into.extend(self.ambient_color.map(f64::to_bits));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robjects::Transform;

    fn settings(samples: u32, share: u32, index: u32, method: ShadowMethod) -> ViewSettings {
        ViewSettings {
            camera: Camera {
                transform: Transform::IDENTITY,
                frame_height: 8.0,
                field_of_view: None,
                near: 0.1,
                far: 100.0,
                focus_distance: 11.0,
                aperture: 0.0,
            },
            camera_views: Vec::new(),
            hide_camera_views: false,
            view_aspect: None,
            jitter: [0.0; 2],
            time: 0.0,
            layer_depth_step: 2e-7,
            exposure: 1.0,
            ambient_color: [1.0; 3],
            ambient_intensity: 0.15,
            default_lights: true,
            shadows: true,
            shadow_softness: 1.0,
            shadow_samples: samples,
            shadow_resolution: 2048,
            shadow_method: method,
            shadow_share: share,
            shadow_index: index,
        }
    }

    #[test]
    fn the_shadow_filter_shares_the_taps_between_renders() {
        let one = settings(16, 1, 0, ShadowMethod::Pcss).shadow_filter();
        assert_eq!(one, [1.0, 16.0, 0.0, 0.0]);
        assert_eq!(
            settings(16, 1, 0, ShadowMethod::Pcf).shadow_filter()[0],
            0.0
        );
        // 16 / 8 = 2 taps, but never fewer than four.
        let filters: Vec<[f32; 4]> = (0..8)
            .map(|index| settings(16, 8, index, ShadowMethod::Pcss).shadow_filter())
            .collect();
        assert!(filters.iter().all(|f| f[1] == MIN_SHADOW_TAPS as f32));
        let turns: std::collections::HashSet<u32> =
            filters.iter().map(|f| f[2].to_bits()).collect();
        assert_eq!(turns.len(), 8); // every render turns the pattern differently
        assert_eq!(
            settings(64, 4, 0, ShadowMethod::Pcss).shadow_filter()[1],
            16.0
        );
        assert_eq!(
            settings(2, 4, 0, ShadowMethod::Pcss).shadow_filter()[1],
            2.0
        ); // never more than asked for
    }

    #[test]
    fn white_ambient_light_is_its_intensity() {
        assert_eq!(
            settings(16, 1, 0, ShadowMethod::Pcss).ambient_linear(),
            [0.15; 3]
        );
    }
}
