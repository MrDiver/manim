"""Render a manimx Scene to a video or an image sequence.

    uv run python scripts/render_scene.py manimx.examples.animated_logo:AnimatedLogo out.mp4 --fps 60 --workers 4
    uv run python scripts/render_scene.py manimx.examples.navigation:Navigation frames/ --png

Uses the offline driver (manimx.drivers): pipelined GPU rendering with
motion blur, optionally across worker processes.
"""

from __future__ import annotations

import argparse
import sys

from manimx.drivers import ImageSequence, RenderSettings, VideoFile, render
from manimx.renderer import Camera, MotionBlur, RenderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", help="module:SceneClass (or a factory function)")
    parser.add_argument("output", help="video file, e.g. scene.mp4, or a directory with --png")
    parser.add_argument("--png", action="store_true", help="write a PNG sequence into the output directory")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--background", default="#ece6e2")
    parser.add_argument("--frame-height", type=float, default=8.0, help="world units shown vertically")
    parser.add_argument("--blur-samples", type=int, default=8, help="1 disables motion blur")
    parser.add_argument("--workers", type=int, default=1, help="render processes")
    parser.add_argument("--in-flight", type=int, default=3, help="frames on the GPU at once per process")
    args = parser.parse_args()

    settings = RenderSettings(
        RenderConfig(
            pixel_width=args.width,
            pixel_height=args.height,
            background_color=args.background,
            camera=Camera(frame_height=args.frame_height),
        ),
        fps=args.fps,
        effects=(MotionBlur(samples=args.blur_samples, shutter=0.5 / args.fps),),
        in_flight=args.in_flight,
    )
    sink = ImageSequence(args.output) if args.png else VideoFile(args.output)

    def progress(done: int, total: int) -> None:
        print(f"\r{done}/{total} frames", end="", file=sys.stderr, flush=True)

    stats = render(args.scene, sink, settings, workers=args.workers, progress=progress)
    print(f"\n{stats.frames} frames in {stats.seconds:.1f} s ({stats.fps:.0f} fps) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
