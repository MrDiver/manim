"""Resolved render-session configuration."""

from __future__ import annotations

__all__ = [
    "PresentationSpec",
    "RenderSessionSpec",
    "resolve_render_session",
]

from dataclasses import dataclass
from typing import Protocol

from .output import OutputFormat, OutputSpec
from .video_encoder import VideoEncoderSpec, resolve_video_encoder


@dataclass(frozen=True, slots=True)
class PresentationSpec:
    """Immutable presentation requests for one render session."""

    open_after_render: bool
    live_preview: bool
    show_in_file_browser: bool


@dataclass(frozen=True, slots=True)
class RenderSessionSpec:
    """Validated artifact, presentation, and execution intent for one session."""

    output: OutputSpec
    presentation: PresentationSpec
    dry_run: bool
    frame_rate: float
    video_encoder: VideoEncoderSpec | None


class _SessionConfigSource(Protocol):
    format: str | OutputFormat | None
    save_sections: bool
    transparent: bool
    preview: bool
    live_preview: bool
    show_in_file_browser: bool
    dry_run: bool
    pixel_width: int
    pixel_height: int
    frame_rate: float
    video_codec: str
    pixel_format: str
    video_encoder_options: dict[str, str]


def resolve_render_session(config: _SessionConfigSource) -> RenderSessionSpec:
    """Resolve and validate one renderer-independent session request."""
    live_preview = config.live_preview
    dry_run = config.dry_run
    requested_format = OutputFormat.parse(config.format)
    fallback_to_still = False
    if dry_run:
        requested_format = OutputFormat.NONE
        save_sections = False
    else:
        save_sections = config.save_sections
        if requested_format is OutputFormat.AUTO:
            if live_preview:
                requested_format = OutputFormat.NONE
            else:
                requested_format = (
                    OutputFormat.MOV if config.transparent else OutputFormat.MP4
                )
                fallback_to_still = True

    output = OutputSpec(
        format=requested_format,
        transparent=config.transparent,
        save_sections=save_sections,
        fallback_to_still=fallback_to_still,
    )
    presentation = PresentationSpec(
        open_after_render=config.preview,
        live_preview=live_preview,
        show_in_file_browser=config.show_in_file_browser,
    )

    if live_preview:
        # The preview window comes back with the native renderer's window.
        raise ValueError("Live preview is not available; remove --live-preview.")
    if presentation.open_after_render and not output.enabled:
        raise ValueError("--preview requires a media artifact.")
    if presentation.show_in_file_browser and not output.enabled:
        raise ValueError("--show_in_file_browser requires a media artifact.")

    video_encoder = resolve_video_encoder(
        output,
        width=config.pixel_width,
        height=config.pixel_height,
        frame_rate=config.frame_rate,
        codec=config.video_codec,
        pixel_format=config.pixel_format,
        options=config.video_encoder_options,
    )
    return RenderSessionSpec(
        output=output,
        presentation=presentation,
        dry_run=dry_run,
        frame_rate=float(config.frame_rate),
        video_encoder=video_encoder,
    )
