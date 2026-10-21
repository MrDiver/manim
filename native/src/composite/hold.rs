//! `FrameHold`: whether a frame shows exactly what the frame rendered
//! last showed. Then the pipeline need not render it, and its caller
//! shows the last frame again.
//!
//! ```text
//! frame_key ──▶ FrameHold::holds ──no──▶ RenderPipeline::render ──▶ FrameHold::rendered
//!                       │
//!                      yes: show the last frame
//! ```
//!
//! The keys come from `RenderPipeline::frame_key`; the hold only compares
//! them, so a caller that renders every frame leaves it out.

/// The key of the frame rendered last.
#[derive(Default)]
pub struct FrameHold {
    /// None: no frame to hold (none rendered yet, or one without a key).
    last: Option<Vec<u64>>,
}

impl FrameHold {
    /// Whether a frame with `key` shows what the frame rendered last
    /// showed. A frame without a key is always rendered.
    pub fn holds(&self, key: &Option<Vec<u64>>) -> bool {
        key.is_some() && *key == self.last
    }

    /// Remember the key of the frame just rendered.
    pub fn rendered(&mut self, key: Option<Vec<u64>>) {
        self.last = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_key_of_the_last_rendered_frame_holds() {
        let mut hold = FrameHold::default();
        assert!(!hold.holds(&Some(vec![1])), "nothing rendered yet");
        hold.rendered(Some(vec![1]));
        assert!(hold.holds(&Some(vec![1])));
        assert!(!hold.holds(&Some(vec![2])));
        assert!(!hold.holds(&None), "a frame without a key is rendered");
        hold.rendered(None);
        assert!(!hold.holds(&Some(vec![1])));
    }
}
