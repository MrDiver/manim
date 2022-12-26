import pytest

from manim._config import tempconfig
from manim.constants import UP
from manim.mobject.geometry.arc import Circle, Dot
from manim.mobject.geometry.polygram import Square
from manim.mobject.types.vectorized_mobject import VGroup
from manim.scene.scene import Scene

TEST_SIZE = 1000


@pytest.fixture
def group_of_circles():
    return VGroup(*[Circle() for _ in range(TEST_SIZE)])


@pytest.fixture
def group_of_squares():
    return VGroup(*[Square() for _ in range(TEST_SIZE)])


def test_benchmark_vmobject_add(benchmark):
    def bench():
        VGroup(*[Circle() for _ in range(TEST_SIZE)])

    benchmark(bench)


def test_benchmark_shift(benchmark, group_of_circles):
    def bench():
        group_of_circles.shift(UP)

    benchmark(bench)


def test_benchmark_become_single(benchmark, group_of_circles, group_of_squares):
    def bench():
        for c, s in zip(group_of_circles, group_of_squares):
            c.become(s)

    benchmark(bench)


def test_benchmark_become_group(benchmark, group_of_circles, group_of_squares):
    def bench():
        group_of_circles.become(group_of_squares)

    benchmark(bench)


def test_benchmark_render_scene(benchmark, group_of_circles):
    scene = Scene()
    scene.add(*group_of_circles)

    def bench():
        with tempconfig(
            {"use_opengl_renderer": False, "disable_caching": True, "dry_run": True}
        ):
            scene.render()

    benchmark(bench)
