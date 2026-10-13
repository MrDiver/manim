"""Shaders: surface shaders on 3D shapes.

Shows ``SurfaceShader``: WGSL that changes a shape's surface per pixel
before lighting. Stripes that scroll with the frame time, a glowing rim
that faces away from the camera, and a switch from one shader to another
(the shader is a signal like any other).
"""

from __future__ import annotations

from manimx.animation import Scene
from manimx.userland import LEFT, PI, RIGHT, FadeIn, Rotate, Sphere, Torus

from manimx import SurfaceShader

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


class Shaders(Scene):
    def construct(self) -> None:
        self.camera.move_to((0.0, -9.0, 4.0)).look_at((0.0, 0.0, 0.0))
        self.camera.set_field_of_view(0.6)
        ball = Sphere(1.4, color="#5ea3d0", roughness=0.6, surface_shader=STRIPES).shift(LEFT * 2.2)
        ring = Torus(1.2, 0.35, color="#20242c", roughness=0.4, surface_shader=RIM).shift(RIGHT * 2.2)

        self.play(FadeIn(ball), FadeIn(ring), run_time=1.0)
        self.play(Rotate(ring, PI, axis=(1.0, 0.0, 0.0)), run_time=3.0)
        self.play(ball.animate.set_surface_shader(RIM), ring.animate.set_surface_shader(STRIPES), run_time=0.0)
        self.play(Rotate(ring, PI, axis=(0.0, 1.0, 0.0)), run_time=3.0)
