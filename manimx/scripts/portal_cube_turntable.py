"""Quick turntable video of the portal cube (a test script, not a video pipeline).

    uv run python scripts/portal_cube_turntable.py portal_cube.mp4 --seconds 6 --fps 30

Orbits the camera once around the cube, renders every frame with motion
blur, and pipes the frames into ffmpeg (which must be installed).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import subprocess
import time

from manimx.examples.portal_cube import BACKGROUND, orbit_camera, portal_cube
from manimx import Transform
from manimx.renderer import FrameComposer, MotionBlur, RenderConfig, WGPURenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="video file, e.g. portal_cube.mp4")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--blur-samples", type=int, default=8, help="1 disables motion blur")
    args = parser.parse_args()

    renderer = WGPURenderer()
    scene = portal_cube()
    composer = FrameComposer(renderer, [MotionBlur(samples=args.blur_samples, shutter=0.5 / args.fps)])
    base = RenderConfig(pixel_width=args.width, pixel_height=args.height, background_color=BACKGROUND)

    # Motion blur samples the scene over time, so instead of orbiting the
    # camera we turn the whole scene (lights included) the other way.
    def scene_at(t: float):
        scene.transform = Transform().rotated(-t / args.seconds * math.tau, (0, 0, 1))
        return scene

    frames = round(args.seconds * args.fps)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{args.width}x{args.height}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.output],
        stdin=subprocess.PIPE,
    )
    start = time.perf_counter()
    for index in range(frames):
        config = dataclasses.replace(base, time=index / args.fps, camera=orbit_camera(-90.0))
        ffmpeg.stdin.write(composer.compose(scene_at, config).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"{frames} frames in {time.perf_counter() - start:.1f} s -> {args.output}")


if __name__ == "__main__":
    main()
