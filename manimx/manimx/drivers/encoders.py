"""H.264 encoders: the GPU's video encoder when there is one, else x264.

Rendered frames become a video through a :class:`~manimx.drivers.sinks.FrameSink`
(frames in, a file out); :class:`~manimx.drivers.sinks.VideoFile` is the
desktop's sink, piping frames to ffmpeg. Which H.264 encoder ffmpeg uses
is chosen here: the hardware encoders of NVIDIA (NVENC), AMD (AMF), Intel
(Quick Sync), Linux's VA-API and Apple's VideoToolbox encode on the GPU's
fixed-function video block, leaving the CPU to manimx; libx264 encodes on
the CPU and is always there.

    encoder = choose_encoder("auto")       # the first one that works here
    VideoFile("out.mp4", encoder=encoder)

Design decisions:

* **Tried, not only listed.** ``ffmpeg -encoders`` lists what ffmpeg was
  built with, not what the machine can do (an NVENC build without an
  NVIDIA GPU, VA-API without a render node). Each candidate encodes a few
  tiny frames first; the first that succeeds is used. The answer is kept
  for the process.
* **Small videos go to x264.** Hardware encoders refuse frames below a
  size of their own; below the size they were tried at, x264 encodes.
* **Quality before speed.** Each encoder is set for constant quality
  about like ``-crf 18`` of x264 (visually lossless for manimx's flat
  colors and sharp edges), with yuv420p, which every player shows. The
  frames arrive as NV12, converted on the GPU (sinks.PixelFormat).
* **The same contract in the browser.** A browser build encodes with
  WebCodecs' ``VideoEncoder`` (the GPU's encoder too) behind the same
  sink contract (``open(format)``, ``write(index, frame)``, ``close()``),
  so the render driver does not know where frames go.
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass

TEST_SIZE = (256, 144)
"""The frame size encoders are tried at."""
ENCODER_NAMES = ("auto", "nvenc", "amf", "qsv", "vaapi", "videotoolbox", "x264")
"""What ``output.encoder`` may say: "auto" picks the first that works."""


@dataclass(frozen=True)
class H264Encoder:
    name: str
    """The setting's name for it, e.g. "nvenc"."""
    ffmpeg_name: str
    """ffmpeg's name, e.g. h264_nvenc."""
    description: str
    input_args: tuple[str, ...] = ()
    """ffmpeg options before the inputs (e.g. the device to encode on)."""
    codec_args: tuple[str, ...] = ()
    """ffmpeg options of the video stream."""

    @property
    def hardware(self) -> bool:
        return self.name != "x264"

    def fits(self, width: int, height: int) -> bool:
        """Whether frames this size are known to work: hardware encoders have
        a smallest size (NVENC's is about 145×49), so they are trusted from
        the size they were tried at on."""
        return not self.hardware or (width >= TEST_SIZE[0] and height >= TEST_SIZE[1])


def _vaapi_device() -> str:
    return next((f"/dev/dri/{node}" for node in sorted(os.listdir("/dev/dri")) if node.startswith("renderD")), "/dev/dri/renderD128") if os.path.isdir("/dev/dri") else "/dev/dri/renderD128"


def candidates() -> tuple[H264Encoder, ...]:
    """Every encoder in the order "auto" tries them (hardware first)."""
    return (
        H264Encoder("nvenc", "h264_nvenc", "NVIDIA NVENC", (), ("-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p")),
        H264Encoder("amf", "h264_amf", "AMD AMF", (), ("-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20", "-pix_fmt", "yuv420p")),
        H264Encoder("qsv", "h264_qsv", "Intel Quick Sync", (), ("-c:v", "h264_qsv", "-preset", "slower", "-global_quality", "20", "-pix_fmt", "nv12")),
        H264Encoder(
            "vaapi", "h264_vaapi", "VA-API", ("-vaapi_device", _vaapi_device()), ("-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", "20")
        ),
        H264Encoder("videotoolbox", "h264_videotoolbox", "Apple VideoToolbox", (), ("-c:v", "h264_videotoolbox", "-q:v", "65", "-pix_fmt", "yuv420p")),
        H264Encoder("x264", "libx264", "x264 (CPU)", (), ("-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p")),
    )


X264 = candidates()[-1]


def choose_encoder(wanted: str = "auto", ffmpeg: str = "ffmpeg") -> H264Encoder:
    """The encoder named ``wanted`` if it works here, or with "auto" the
    first that does (x264 when no hardware encoder works)."""
    if wanted not in ENCODER_NAMES:
        raise ValueError(f"unknown encoder {wanted!r}; use one of {', '.join(ENCODER_NAMES)}")
    working = working_encoders(ffmpeg)
    if wanted == "auto":
        return working[0] if working else X264
    chosen = next((encoder for encoder in working if encoder.name == wanted), None)
    if chosen is None:
        raise RuntimeError(f"the {wanted} encoder does not work here (ffmpeg found: {', '.join(e.name for e in working) or 'none'})")
    return chosen


@functools.cache
def working_encoders(ffmpeg: str = "ffmpeg") -> tuple[H264Encoder, ...]:
    """The encoders ffmpeg has and that encode a test video here, in "auto" order."""
    try:
        listed = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ()
    return tuple(encoder for encoder in candidates() if f" {encoder.ffmpeg_name} " in listed and _encodes(encoder, ffmpeg))


def _encodes(encoder: H264Encoder, ffmpeg: str) -> bool:
    """Whether ``encoder`` turns a few 256×144 test frames into H.264 here."""
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", *encoder.input_args,
        "-f", "lavfi", "-i", f"testsrc2=size={TEST_SIZE[0]}x{TEST_SIZE[1]}:rate=30:duration=0.2",
        *encoder.codec_args, "-f", "null", "-",
    ]  # fmt: skip
    try:
        return subprocess.run(command, capture_output=True, timeout=20, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


__all__ = ["ENCODER_NAMES", "X264", "H264Encoder", "candidates", "choose_encoder", "working_encoders"]
