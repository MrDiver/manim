"""Shaders: surface shaders on 3D shapes.

A SurfaceShader is WGSL that changes a surface per pixel before
lighting: here scrolling stripes and a glowing rim. The shader is a
signal like any other, so a shape can switch to another one.
"""

from __future__ import annotations

from manimx.animation import DirectionalLight, Scene
from manimx.userland import PI, Box, FadeIn, Rotate, Sphere, Torus

from manimx import SurfaceShader

from .style import BLUE, NAVY

STRIPES = SurfaceShader("""
fn shade(surface: Surface, in: SurfaceInput) -> Surface {
    var out = surface;
    let stripe = smoothstep(0.45, 0.55, fract(in.local_position.z * 3.0 - frame.time * 0.5));
    out.base_color = mix(surface.base_color, vec3<f32>(0.95, 0.75, 0.2), stripe);
    out.roughness = mix(surface.roughness, 0.2, stripe);
    return out;
}
""")

RIM = SurfaceShader("""
fn shade(surface: Surface, in: SurfaceInput) -> Surface {
    var out = surface;
    let towards_camera = normalize(frame.camera_position.xyz - in.position);
    let rim = pow(1.0 - abs(dot(surface.normal, towards_camera)), 3.0);
    out.emissive = surface.emissive + vec3<f32>(0.3, 0.8, 1.0) * rim * 2.0;
    return out;
}
""")

CHECKER = SurfaceShader("""
fn shade(surface: Surface, in: SurfaceInput) -> Surface {
    var out = surface;
    let cell = floor(in.position.xy * 1.0);
    let dark = (i32(cell.x) + i32(cell.y)) % 2 == 0;
    out.base_color = select(surface.base_color, surface.base_color * 0.6, dark);
    return out;
}
""")


class Shaders(Scene):
    def construct(self) -> None:
        self.camera.move_to((0.0, -10.0, 4.5)).look_at((0.0, 0.0, 0.5))
        self.camera.set_field_of_view(0.6)
        self.add(DirectionalLight(direction=(0.4, 0.7, -1.0), intensity=2.5))
        floor = Box(16.0, 10.0, 0.2, color=NAVY, roughness=0.8, surface_shader=CHECKER).move_to((0.0, 2.0, -0.1))
        ball = Sphere(1.3, color=BLUE, roughness=0.6, surface_shader=STRIPES).move_to((-2.2, 0.0, 1.3))
        ring = Torus(1.1, 0.35, color="#20242c", roughness=0.4, surface_shader=RIM).move_to((2.2, 0.0, 1.3))
        self.add(floor)
        self.play(FadeIn(ball), FadeIn(ring), run_time=0.8)
        self.play(Rotate(ring, PI, axis=(1.0, 0.0, 0.0)), run_time=2.5)
        self.play(ball.animate.set_surface_shader(RIM), ring.animate.set_surface_shader(STRIPES), run_time=0.0)
        self.play(Rotate(ring, PI, axis=(0.0, 1.0, 0.0)), run_time=2.5)
