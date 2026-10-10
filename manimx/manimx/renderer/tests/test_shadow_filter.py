"""How shadow lookups are spread over the renders a frame averages."""

from __future__ import annotations

from manimx.renderer.config import MIN_SHADOW_TAPS, RenderConfig


def test_one_render_takes_every_tap_unturned() -> None:
    assert RenderConfig(shadow_samples=16).shadow_filter == (1.0, 16.0, 0.0, 0.0)
    assert RenderConfig(shadow_samples=16, shadow_method="pcf").shadow_filter[0] == 0.0


def test_motion_blur_renders_share_the_taps_with_turned_patterns() -> None:
    filters = [RenderConfig(shadow_samples=16, shadow_share=8, shadow_index=index).shadow_filter for index in range(8)]
    assert {f[1] for f in filters} == {float(MIN_SHADOW_TAPS)}  # 16 / 8 = 2, but at least 4
    turns = sorted(f[2] for f in filters)
    assert len(set(turns)) == 8 and all(0.0 <= turn < 1.0 for turn in turns)
    # The turn depends on the render's index only: equal frames stay equal.
    assert RenderConfig(time=1.0, shadow_share=8, shadow_index=3).shadow_filter == RenderConfig(time=2.0, shadow_share=8, shadow_index=3).shadow_filter
    assert RenderConfig(shadow_samples=64, shadow_share=4).shadow_filter[1] == 16.0
    assert RenderConfig(shadow_samples=2, shadow_share=4).shadow_filter[1] == 2.0  # never more than asked for


def test_full_renders_use_pcss_and_previews_pcf() -> None:
    from manimx.cli import app
    from manimx.settings import Settings

    class Project:
        settings = Settings()

        def render_settings(self, chosen):
            return chosen

    assert Settings().effects.shadows.method == "pcss"
    assert app._preview_settings(Project()).effects.shadows.method == "pcf"
