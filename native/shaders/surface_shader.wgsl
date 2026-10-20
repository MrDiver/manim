// The surface shader of a mesh pipeline: changes the material's Surface
// per pixel, before it is stored in the G-buffer (see mesh.wgsl).
//
// A SurfaceShader (manimx/robjects/material.py) replaces the line below
// with its code and a shade_surface() that calls its shade(); pipelines
// are made per shader. Without one, the surface stays as it is.

fn shade_surface(surface: Surface, in: SurfaceInput) -> Surface { return surface; }
