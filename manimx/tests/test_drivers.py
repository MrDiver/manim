"""The offline render driver: pipelining, sinks and worker processes."""

import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from manimx.animation import Scene, Sound
from manimx.drivers import (
    FrameList,
    FramePipeline,
    FrameSink,
    FrameSource,
    ImageSequence,
    RenderSettings,
    VideoFile,
    load_source,
    render,
)
from manimx.examples.navigation import Navigation, config
from manimx.renderer import FrameComposer, WGPURenderer

SOURCE = "manimx.examples.navigation:Navigation"


@pytest.fixture(scope="module")
def renderer() -> WGPURenderer:
    try:
        return WGPURenderer()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable GPU: {error}")


def settings(**options) -> RenderSettings:
    return RenderSettings(config(64, 36), fps=10, start=1.5, end=2.5, **options)


def expected_frames(renderer: WGPURenderer) -> list[np.ndarray]:
    prepared = Navigation().prepare()
    composer = FrameComposer(renderer)
    return [composer.compose(prepared.frame_at(t), config(64, 36)) for t in settings().frame_times(prepared.duration)]


def test_frame_times() -> None:
    assert RenderSettings(fps=4).frame_times(1.0) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert RenderSettings(fps=10, start=0.3, end=0.5).frame_times(9.0) == pytest.approx([0.3, 0.4, 0.5])
    assert RenderSettings(start=2.0).frame_times(1.0) == []


def test_sources_load_from_names_classes_factories_and_objects() -> None:
    class Tiny(Scene):
        def construct(self):
            self.wait(2)

    assert load_source(SOURCE).duration == Navigation().prepare().duration
    assert load_source(Tiny).duration == 2.0
    assert load_source(lambda: Tiny().prepare()).duration == 2.0
    fixed = FrameSource(lambda t: None, 1.0)
    assert load_source(fixed) is fixed
    with pytest.raises(ValueError, match="module:attr"):
        load_source("manimx.examples.navigation")
    with pytest.raises(TypeError, match="duration and state_at"):
        load_source(lambda: 42)


def test_pipelined_frames_match_synchronous_renders(renderer: WGPURenderer) -> None:
    sink = FrameList()
    stats = render(SOURCE, sink, settings())

    assert stats.frames == 11 and stats.fps > 0
    for frame, expected in zip(sink.frames, expected_frames(renderer), strict=True):
        np.testing.assert_array_equal(frame, expected)
    with pytest.raises(ValueError, match="at least one"):
        FramePipeline(FrameComposer(renderer), config(64, 36), in_flight=0)


def test_worker_processes_give_the_same_frames_in_order(renderer: WGPURenderer) -> None:
    sink = FrameList()
    progress = []
    render(SOURCE, sink, settings(chunk=3, in_flight=2), workers=2, progress=lambda done, total: progress.append((done, total)))

    for frame, expected in zip(sink.frames, expected_frames(renderer), strict=True):
        np.testing.assert_array_equal(frame, expected)
    assert progress[-1] == (11, 11)
    with pytest.raises(TypeError, match="picklable"):
        render(lambda: Navigation(), FrameList(), settings(), workers=2)


def test_image_sequences_are_written_unordered(renderer: WGPURenderer, tmp_path) -> None:
    sink = ImageSequence(tmp_path / "frames", threads=2)
    assert not sink.ordered
    render(SOURCE, sink, settings())

    files = sorted((tmp_path / "frames").iterdir())
    assert [file.name for file in files[:2]] == ["frame_00000.png", "frame_00001.png"] and len(files) == 11
    np.testing.assert_array_equal(np.asarray(Image.open(files[3])), expected_frames(renderer)[3])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_files_and_encoder_errors(tmp_path) -> None:
    path = tmp_path / "clip.mp4"
    render(SOURCE, VideoFile(path), RenderSettings(config(64, 36), fps=10, end=0.5))
    assert path.stat().st_size > 0

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        render(SOURCE, VideoFile(tmp_path / "bad.mp4", codec_args=["-c:v", "no-such-codec"]), settings())


def test_sink_errors_stop_the_render() -> None:
    class Broken(FrameSink):
        closed = False

        def write(self, index, image):
            raise OSError("disk full")

        def close(self):
            Broken.closed = True

    with pytest.raises(RuntimeError, match="writing frames failed") as error:
        render(SOURCE, Broken(), settings())
    assert isinstance(error.value.__cause__, OSError) and Broken.closed


BOUNCING = "manimx.examples.bouncing:Bouncing"


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_sounds_are_muxed_into_videos_and_written_next_to_images(tmp_path) -> None:
    short = RenderSettings(config(64, 36), fps=10, end=2.0)
    render(BOUNCING, VideoFile(tmp_path / "with_sound.mp4"), short)
    render(BOUNCING, ImageSequence(tmp_path / "frames"), short)
    render(SOURCE, VideoFile(tmp_path / "silent.mp4"), short)  # no sounds: no audio track
    render(BOUNCING, VideoFile(tmp_path / "muted.mp4"), RenderSettings(config(64, 36), fps=10, end=2.0, audio=False))

    def streams(name):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(tmp_path / name)],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return result.stdout.split()

    assert streams("with_sound.mp4") == ["video", "audio"]
    assert streams("silent.mp4") == ["video"] and streams("muted.mp4") == ["video"]
    sound = Sound.load(tmp_path / "frames" / "audio.wav")
    assert sound.duration == pytest.approx(2.1) and np.abs(sound.samples).max() > 0.05
