"""Render a manimx Scene to a video (a test script, not a video pipeline).

    uv run python scripts/render_scene.py manimx.examples.animated_logo:AnimatedLogo out.mp4 --fps 60

Prepares the scene once, then renders every frame with state_at(t) through
a FrameComposer with motion blur and pipes the frames into ffmpeg.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import subprocess
import time

from manimx.renderer import FrameComposer, MotionBlur, RenderConfig, WGPURenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", help="module:SceneClass")
    parser.add_argument("output", help="video file, e.g. scene.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--background", default="#ece6e2")
    parser.add_argument("--blur-samples", type=int, default=8, help="1 disables motion blur")
    args = parser.parse_args()

    module_name, class_name = args.scene.split(":")
    prepared = getattr(importlib.import_module(module_name), class_name)().prepare()

    renderer = WGPURenderer()
    composer = FrameComposer(renderer, [MotionBlur(samples=args.blur_samples, shutter=0.5 / args.fps)])
    base = RenderConfig(pixel_width=args.width, pixel_height=args.height, background_color=args.background)

    frames = round(prepared.duration * args.fps) + 1
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{args.width}x{args.height}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.output],
        stdin=subprocess.PIPE,
    )
    start = time.perf_counter()
    for index in range(frames):
        config = dataclasses.replace(base, time=index / args.fps)
        ffmpeg.stdin.write(composer.compose(prepared.state_at, config).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"{frames} frames ({prepared.duration:.2f} s) in {time.perf_counter() - start:.1f} s -> {args.output}")


if __name__ == "__main__":
    main()
