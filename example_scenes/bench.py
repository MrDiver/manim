from re import U
from manim import *
import itertools as it
import numpy as np

from manim.mobject.opengl.opengl_surface import OpenGLSurface
from manim.mobject.opengl.opengl_three_dimensions import OpenGLSurfaceMesh


def plot_clipped(f, ax, UPPER_BOUND=None, LOWER_BOUND=None, resolution=300, **kwargs):
    # Create Bounds
    if LOWER_BOUND is None:
        LOWER_BOUND = ax.y_range[0]
    if UPPER_BOUND is None:
        UPPER_BOUND = ax.y_range[1]

    def predicate(x):
        return (x < LOWER_BOUND) or (UPPER_BOUND < x)

    # Create Points and set invalid points to None
    xs = np.linspace(ax.x_range[0], ax.x_range[1], resolution)
    ys = [f(x) for x in xs]
    ys = [None if predicate(y) else y for y in ys]
    points = np.vstack((xs, ys, np.zeros(xs.shape))).T

    # print(points)
    # Group by None
    groups = it.groupby(
        points, lambda x: x[1] is None or x[1] is np.nan or x[1] is np.inf
    )
    # print(list(groups))
    # Filter out None groups
    filtered = it.filterfalse(lambda x: x[0], groups)
    # Get the points and discard the group_id from (id, elements)
    mapped = [np.asarray(list(x[1])) for x in filtered]

    # Create VMobjects which substitute ParametricFunctions
    functions = VGroup()
    for points in mapped:
        dummy = VMobject(**kwargs)
        dummy.set_points_smoothly(points)
        functions.add(dummy)
    functions.apply_function(lambda x: ax.c2p(x[0], x[1], x[2]))
    return functions


class GradientIntegral(ThreeDScene):
    def construct(self):
        def f(x):
            return x * x

        ax = Axes(y_range=[-5, 5])
        labels = ax.get_axis_labels(x_label="x", y_label="y")
        functions = plot_clipped(f, ax, color=RED)
        functions2 = plot_clipped(lambda x: f(x) - 1, ax, color=BLUE)
        functions3 = plot_clipped(lambda x: f(x - 2) - 1, ax, color=GREEN)
        functions4 = plot_clipped(lambda x: np.sin(x), ax, color=YELLOW)
        self.add(ax, functions, labels)
        self.play(functions.animate.become(functions2))
        self.play(functions.animate.become(functions3))
        self.play(functions.animate.become(functions4))


class Pend2(VGroup):
    def __init__(self, length=1.5, width=5, arrow_length=0.75, **kwargs):
        super().__init__(**kwargs)
        self.phi = ValueTracker(0)
        self.d_phi = ValueTracker(0)
        self.length = length
        self.arrow_length = arrow_length

        self.center_dot = Dot(radius=width / 50)

        self.line = Line(ORIGIN, DOWN * length).set_color(BLUE).set_stroke(width=width)
        self.ref_line = self.line.copy()
        self.arc = Arc(length / 3, start_angle=0, angle=PI / 2, color=RED)
        self.arc_text = MathTex(r"\varphi", font_size=40).add_background_rectangle()
        self.mass = Dot(radius=width / 35).set_color(RED_E)
        self.tang_force = Arrow(ORIGIN, DOWN, color=GREEN)

        subs = VGroup(self.arc, self.line, self.tang_force)
        relatives = VGroup(self.mass)
        self.add(subs, relatives, self.center_dot)
        self.add(Rectangle(width=length * 2 + 1, height=length * 2 + 1).set_opacity(0))

        self.add_updater(self.update_self)

        # for mob in subs.submobjects:
        #     mob.add_updater(lambda m: m.shift(self.center_dot.get_center()))

    def update_self(self, dt):
        self.line.become(self.ref_line.copy().rotate_about_origin(self.phi.get_value()))
        self.arc.become(
            Arc(
                self.length / 3,
                start_angle=-PI / 2,
                angle=self.phi.get_value(),
                color=RED,
            )
        )
        self.arc_text.move_to(self.arc.get_edge_center(UP))
        self.mass.move_to(self.line.get_end())
        self.tang_force.become(
            Arrow(
                ORIGIN,
                RIGHT * self.d_phi.get_value() * self.arrow_length,
                buff=0,
                color=GREEN,
            )
            .shift(self.length * DOWN)
            .rotate_about_origin(self.phi.get_value())
        )
        self.center_dot.set_color(
            interpolate_color(GREEN, RED, np.tanh(np.abs(self.d_phi.get_value())))
        )


class TestUpdater(Scene):
    def construct(self):
        pend = Pend2()
        self.add(pend)
        self.play(
            pend.phi.animate.set_value(PI / 2),
            pend.d_phi.animate.set_value(1),
            run_time=2,
        )


import random


def circles_intersect(circle1, circle2):
    distance = np.linalg.norm(circle1.get_center() - circle2.get_center())
    return distance < circle1.get_width() / 2 + circle2.get_width() / 2


class MovingCircles(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_points = 100
        self.ps = [
            Dot(0.1).shift(random.uniform(-1, 1) * UP + random.uniform(-1, 1) * RIGHT)
            for _ in range(num_points)
        ]
        self.add(*self.ps)
        for point in self.ps:
            point.velocity = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0
        self.add_updater(self.update_self)

    @staticmethod
    def update_self(self, dt):
        for point in self.ps:
            point.shift(point.velocity)

            if point.get_center_of_mass()[0] < -2 or point.get_center_of_mass()[0] > 2:
                point.velocity = (
                    -point.velocity[0],
                    point.velocity[1],
                    point.velocity[2],
                )
            if point.get_center_of_mass()[1] < -2 or point.get_center_of_mass()[1] > 2:
                point.velocity = (
                    point.velocity[0],
                    -point.velocity[1],
                    point.velocity[2],
                )

            for other_point in self.ps:
                if point != other_point and circles_intersect(point, other_point):
                    point.velocity, other_point.velocity = (
                        other_point.velocity,
                        point.velocity,
                    )
        return self


class TestCircles(Scene):
    def construct(self):
        circles = MovingCircles()
        self.add(circles)
        self.wait(2)


class OpenGLIntro(Scene):
    def construct(self):
        surface = OpenGLSurface(
            lambda u, v: (u, v, u * np.sin(v) + v * np.cos(u)),
            u_range=(-3, 3),
            v_range=(-3, 3),
        )

        surface2 = OpenGLSurface(
            lambda u, v: (u, v * np.cos(u), u * np.sin(v) + v * np.cos(u)),
            u_range=(-3, 3),
            v_range=(-3, 3),
        )

        mesh = OpenGLSurfaceMesh(surface)

        self.play(Create(mesh))
        self.play(FadeTransform(mesh, surface))
        self.play(surface.animate.become(surface2))

        self.wait()


class TestAdd(Scene):
    def construct(self):
        c = Circle()
        circle = VGroup(*[c for i in range(1000)])
        self.add(circle)
        self.wait(1)
        print(circle.submobjects)


# from manim import manimation


# @manimation()
# def doing_stuff(self: Scene):
#     self.add(Circle())


# @manimation()
# def doing_stuff2(self: Scene):
#     self.add(Square())
#     self.play(Write(Text("Hello World")))


# print("MAIN", doing_stuff)
# print(doing_stuff.render)


class AnimateParametricFunction(Scene):
    @staticmethod
    def func(t):
        """paramatric curve"""
        return np.array(
            (np.cos(t) * np.sin(1 / 4 * t), np.sin(t) * np.sin(1 / 4 * t), 0)
        )

    def construct(self):
        ref_func = ParametricFunction(
            self.func, t_range=np.array([0, 8 * PI]), use_vectorized=True
        ).set_color(ORANGE)

        t = ValueTracker(0)
        partial_func = ref_func.get_subcurve(0, 1)
        partial_func.add_updater(
            lambda m: m.become(ref_func.get_subcurve(0, t.get_value()))
        )

        dot = Dot(color=WHITE, radius=0.05)
        self.add(dot)
        dot.add_updater(lambda m: m.move_to(partial_func.get_end()))

        self.add(partial_func)

        self.play(t.animate.set_value(1), run_time=1, rate_func=linear)
        self.wait()


class TracedPathExample(Scene):
    def construct(self):
        circ = Circle(color=RED).shift(4*LEFT)
        dot = Dot(color=RED).move_to(circ.get_start())
        rolling_circle = VGroup(circ, dot)
        trace = TracedPath(circ.get_start)
        rolling_circle.add_updater(lambda m: m.rotate(-0.3))
        self.add(trace, rolling_circle)
        self.play(rolling_circle.animate.shift(8*RIGHT), run_time=4, rate_func=linear)

class Test6(Scene):
    def construct(self) -> None:
        m = ManimBanner()
        self.play(m.create())

class Test3(Scene):
    def construct(self) -> None:
        c = Dot().shift(RIGHT)
        text = Text("Hello World").scale(3)
        self.add(c)
        paths = []

        for char in [text[i] for i in range(10)]:
            p = char.point_from_proportion(0)
            self.play(c.animate.move_to(p), run_time=0.3)
            path = TracedPath(lambda: c.get_center()).set_fill(opacity=0)
            paths.append(path)
            self.add(path)
            for i in np.linspace(0,1,1000):
                p = char.point_from_proportion(i)
                self.play(c.animate.move_to(p), run_time=0.0001)
            path.suspend_updating()
        self.wait()
        v = VGroup(*paths)
        self.play(v.animate.arrange(DOWN).set_stroke(color=RED))
        self.play(FadeOut(v, shift=LEFT))
