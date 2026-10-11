//! What the renderer reads from a `RenderConfig` (manimx/renderer/config.py)
//! besides the render object tree: the camera, the camera views, and the
//! lighting and shadow settings of one view.

use glam::DMat4;

use crate::render::packing::packer::PackSettings;
use crate::render::packing::records::Frame;
use crate::render::packing::shadows::{frame_record, FrameSettings};
use crate::render::packing::vectors::CameraViews;
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
    /// Seconds, `frame.time` in shaders.
    pub time: f64,
    pub layer_depth_step: f64,
    pub exposure: f64,
    pub ambient_linear: [f64; 3],
    /// Light meshes with a key light when the scene has no lights.
    pub default_lights: bool,
    pub shadows: bool,
    pub shadow_softness: f64,
    pub shadow_samples: f64,
    pub shadow_resolution: u32,
    /// `RenderConfig.shadow_filter`.
    pub shadow_filter: [f32; 4],
}

impl ViewSettings {
    pub fn aspect_ratio(&self, size: (u32, u32)) -> f64 {
        self.view_aspect.unwrap_or(size.0 as f64 / size.1 as f64)
    }

    pub fn frame_settings(&self) -> FrameSettings {
        FrameSettings {
            time: self.time,
            layer_depth_step: self.layer_depth_step,
            exposure: self.exposure,
            ambient_linear: self.ambient_linear,
            shadow_softness: self.shadow_softness,
            shadow_samples: self.shadow_samples,
            shadow_resolution: self.shadow_resolution,
            shadows: self.shadows,
            shadow_filter: self.shadow_filter,
        }
    }

    /// How the packer packs this view: camera view `i` is layer `i` of the
    /// camera view texture.
    pub fn pack_settings(&self) -> PackSettings {
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
            camera_views,
            hide_camera_views: self.hide_camera_views,
        }
    }

    /// The camera's Frame record for a picture of `size` pixels.
    pub fn camera_frame(&self, size: (u32, u32), light_count: u32) -> Frame {
        let camera = &self.camera;
        frame_record(
            &camera.view_matrix(),
            &camera.projection_matrix(self.aspect_ratio(size)),
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
            self.layer_depth_step.to_bits(),
            self.exposure.to_bits(),
            self.default_lights as u64,
            self.shadows as u64,
            self.shadow_softness.to_bits(),
            self.shadow_samples.to_bits(),
            self.shadow_resolution as u64,
        ]);
        into.extend(self.ambient_linear.map(f64::to_bits));
        into.extend(self.shadow_filter.map(|v| v.to_bits() as u64));
    }
}
