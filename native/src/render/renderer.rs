//! `Renderer`: the wgpu renderer as the compositor drives it. It is sent
//! scenes (`apply`, their `Update`s), keeps what it needs of them until
//! told to `forget` them, and renders one scene seen with a view's
//! settings into layers (`render`):
//!
//! ```text
//! Update ──apply──▶ Packer ──Packet + Uploads──▶ Drawer ──▶ Layers
//! ```
//!
//! The layers are the renderer's own textures, valid until its next
//! render; the compositor (composite/) turns them into frames.

use crate::error::Result;
use crate::flat::Update;
use crate::gpu::device::Gpu;
use crate::render::draw::Pictures;
use crate::render::draw::{Drawer, TargetSettings};
use crate::render::pack::packer::Packer;
use crate::render::packet::records::Frame;
use crate::render::packet::{Packet, VectorDraws};
use crate::render::settings::ViewSettings;

/// What one render made: the lit layer (meshes, linear HDR), the vector
/// layer (shapes and curves, premultiplied), the depth of each, and the
/// camera's `Frame` uniform they were drawn with (effects read it).
pub struct Layers {
    pub lit: wgpu::Texture,
    pub vector: wgpu::Texture,
    pub mesh_depth: wgpu::TextureView,
    pub vector_depth: wgpu::TextureView,
    pub frame: wgpu::Buffer,
}

pub struct Renderer {
    packer: Packer,
    drawer: Drawer,
    settings: TargetSettings,
}

impl Renderer {
    /// A renderer of layers of `settings`' size and sampling, whose
    /// vector record pages hold at most `page_bytes` per buffer.
    pub fn new(gpu: &Gpu, settings: TargetSettings, page_bytes: u64) -> Result<Renderer> {
        Ok(Renderer {
            packer: Packer::with_page_bytes(page_bytes as usize),
            drawer: Drawer::new(gpu, page_bytes)?,
            settings,
        })
    }

    /// Bring what is kept of `update`'s scene up to date.
    pub fn apply(&mut self, update: Update) -> Result<()> {
        self.packer.apply(update)
    }

    /// The revision of what is kept of `scene` and whether what it shows
    /// may change with the time alone; None for a scene not sent yet.
    pub fn state(&self, scene: u64) -> Option<(u64, bool)> {
        self.packer.state(scene)
    }

    /// Drop everything kept for `scene`.
    pub fn forget(&mut self, scene: u64) {
        self.packer.forget(scene);
    }

    /// Render `scene` seen with `settings` into the layers, its camera
    /// views' shapes showing `pictures`. The passes are recorded into
    /// `encoder`, which the caller submits.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        scene: u64,
        settings: &ViewSettings,
        pictures: Pictures,
    ) -> Result<Layers> {
        let size = (self.settings.width, self.settings.height);
        let packet = self
            .packer
            .pack_frame(scene, &settings.pack_settings(size))?;
        let light_count = packet
            .meshes
            .as_ref()
            .map_or(0, |meshes| meshes.lights.len() as u32);
        let frame = settings.camera_frame(size, light_count);
        self.draw(gpu, encoder, &packet, &frame, pictures)
    }

    /// The layers of nothing: an empty scene.
    pub fn render_empty(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        pictures: Pictures,
    ) -> Result<Layers> {
        let packet = Packet {
            draws: VectorDraws::default(),
            scope_parents: Vec::new(),
            meshes: None,
        };
        let frame: Frame = bytemuck::Zeroable::zeroed();
        self.draw(gpu, encoder, &packet, &frame, pictures)
    }

    fn draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        packet: &Packet,
        frame: &Frame,
        pictures: Pictures,
    ) -> Result<Layers> {
        let uploads = self.packer.uploads(packet.meshes.is_some());
        let targets = self.drawer.draw(
            gpu,
            packet,
            uploads,
            pictures,
            frame,
            self.settings,
            encoder,
        )?;
        Ok(Layers {
            lit: targets.lit.clone(),
            vector: targets.vector.clone(),
            mesh_depth: targets.depth_view.clone(),
            vector_depth: targets.vector_depth_view.clone(),
            frame: self.drawer.frame_uniform().clone(),
        })
    }
}
