// Vector pipeline: fills and strokes of every Shape2D in the scene.
//
// All geometry lives in a few large storage buffers that are shared by every
// object ("vertex pulling"). The CPU encodes one indirect draw per fill and
// per stroke, and the whole opaque or transparent pass is a single
// multi_draw_indirect call. Each draw's `first_instance` points into the
// `instances` table, which tells this shader what to generate:
//
//   fill   draw: instance_count = 1, vertex_index walks the object's
//                pre-tessellated triangles in `fill_vertices`.
//   stroke draw: instance_count = number of curves, one instance per cubic
//                curve; vertex_index enumerates procedurally generated
//                ribbon / joint / cap triangles (see `stroke_vertex`).
//
// Depth ordering of flat 2D shapes: every object gets a `layer` from its
// painter's order. The fragment shader pulls the depth slightly towards the
// camera per layer, so later objects win the depth test against earlier,
// coplanar ones. This reproduces Manim's painter's order while keeping real
// 3D depth (see `layered_depth`).
//
// That bias only orders vector shapes among themselves: they have their own
// depth buffer. Occlusion by meshes is a separate, unbiased test against the
// G-buffer depth (`hidden_by_meshes`). Otherwise a late shape with a large
// layer could be pulled in front of a mesh surface it is behind.
//
// Clipping: each ClippingGroup's mask is first rendered with this same
// shader (entry point `fs_mask`) into one layer of `mask_layers`. Content
// inside the group reads its layer at its own pixel and multiplies its
// alpha by the mask's alpha. Masks of nested groups are themselves clipped
// by the enclosing mask, so one lookup covers all enclosing groups.

// ---------------------------------------------------------------------------
// Bindings (layouts mirror the numpy dtypes in vector_batch.py)
// ---------------------------------------------------------------------------

#include "frame.wgsl"

const NO_CLIP: i32 = -1;

struct Object {
    model: mat4x4<f32>,
    layer: f32,                 // fill layer; the stroke uses layer + 1
    clip: i32,                  // layer in `mask_layers`, or NO_CLIP
    flags: u32,                 // FLAG_* bits
    _pad: f32,
};

/// The stroke ribbon faces the camera (Curve3D) instead of lying in the
/// object's XY plane (Shape2D).
const FLAG_CAMERA_FACING: u32 = 1u;

/// Draw the object only while its local +z axis faces the camera (masks of
/// one-sided clipping groups, e.g. the faces of a portal cube).
const FLAG_FRONT_ONLY: u32 = 2u;

const KIND_FILL: u32 = 0u;
const KIND_STROKE: u32 = 1u;

struct Instance {
    kind: u32,
    object: u32,
    curve: u32,                 // stroke only: index into `stroke_curves`
    _pad: u32,
};

struct FillVertex {
    position: vec2<f32>,        // local 2D
    uv: vec2<f32>,              // implicit curve coordinates
    color: vec4<f32>,
    side: f32,                  // 0 solid, +1 keep control side, -1 keep chord side
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct StrokePoint {
    position: vec3<f32>,        // local; z = 0 for Shape2D
    width: f32,                 // full stroke width in world units
    color: vec4<f32>,
};

const NO_NEIGHBOR: i32 = -1;

struct StrokeCurve {
    first_point: u32,           // index of the curve's 4 control points
    prev_point: i32,            // first point of the previous curve, or NO_NEIGHBOR
    next_point: i32,            // first point of the next curve, or NO_NEIGHBOR
    start_joint: u32,           // JointType shared with the previous curve
    end_joint: u32,             // JointType shared with the next curve
    start_cap: u32,             // CapStyle, used when there is no previous curve
    end_cap: u32,               // CapStyle, used when there is no next curve
    _pad: u32,
};

@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> fill_vertices: array<FillVertex>;
@group(0) @binding(4) var<storage, read> stroke_points: array<StrokePoint>;
@group(0) @binding(5) var<storage, read> stroke_curves: array<StrokeCurve>;
@group(0) @binding(6) var mask_layers: texture_2d_array<f32>;
@group(0) @binding(7) var mesh_depth: texture_depth_multisampled_2d;

// ---------------------------------------------------------------------------
// Shared vertex output
// ---------------------------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(3) @interpolate(flat) layer: f32,
    @location(4) @interpolate(flat) clip: i32,
    // Interpolated per MSAA sample, so the curve test in the fragment shader
    // runs per sample and curved edges get the same antialiasing as polygon
    // edges.
    @location(1) @interpolate(perspective, sample) uv: vec2<f32>,
    @location(2) @interpolate(flat) side: f32,
};

fn project(world: vec3<f32>) -> vec4<f32> {
    return frame.view_projection * vec4<f32>(world, 1.0);
}

/// How far one layer is pulled towards the camera, as a fraction of the
/// depth change across one pixel.
const LAYER_SLOPE_SCALE: f32 = 1.0 / 64.0;

/// Depth of a fragment on layer `layer`.
///
/// A constant offset per layer is enough for surfaces facing the camera. On
/// a steeply tilted plane the rasterizer's sub-pixel snapping moves the
/// interpolated depth by more than that, and coplanar layers z-fight. Adding
/// a fraction of the depth slope (like a slope-scaled polygon offset) keeps
/// the layers apart at any angle.
fn layered_depth(depth: f32, layer: f32, slope: f32) -> f32 {
    return depth - layer * (frame.layer_depth_step + LAYER_SLOPE_SCALE * slope);
}

/// A position outside the clip volume; triangles made of it are discarded.
const CULLED = vec4<f32>(0.0, 0.0, 2.0, 1.0);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let instance = instances[instance_index];
    let object = objects[instance.object];
    if (object.flags & FLAG_FRONT_ONLY) != 0u && !faces_camera(object) {
        var culled: VertexOutput;
        culled.clip_position = CULLED;
        return culled;
    }
    if instance.kind == KIND_FILL {
        return fill_vertex(object, vertex_index);
    }
    return stroke_vertex(object, stroke_curves[instance.curve], vertex_index);
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

/// Vector layer passes: premultiplied color plus the layered depth.
///
/// The vector layer starts out transparent and is composited over the lit
/// 3D layer later, so colors are premultiplied by alpha: that makes MSAA
/// resolve and "over" blending exact for partially covered pixels.
/// Whether the object's plane shows its front (+z) side to the camera. A
/// plane faces the camera either everywhere or nowhere, so testing its
/// origin decides for the whole object.
fn faces_camera(object: Object) -> bool {
    let x_axis = (object.model * vec4<f32>(1.0, 0.0, 0.0, 0.0)).xyz;
    let y_axis = (object.model * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz;
    let origin = (object.model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    return dot(cross(x_axis, y_axis), towards_camera(origin)) > 0.0;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(sample_index) sample: u32) -> FragmentOutput {
    // Derivatives must be taken before any fragment may be discarded.
    let slope = fwidth(in.clip_position.z);
    if hidden_by_meshes(in.clip_position, sample, slope) {
        discard;
    }
    var out: FragmentOutput;
    out.color = premultiplied(shade(in));
    out.depth = layered_depth(in.clip_position.z, in.layer, slope);
    return out;
}

/// Whether a mesh is in front of this sample. Shapes lying on a mesh surface
/// stay visible, since vector shapes are drawn on top of meshes in painter's
/// order. Coplanar surfaces rasterized from different triangles differ by
/// the rasterizer's sub-pixel snapping, about 1/256 of a pixel's depth
/// change, so the tolerance is a small fraction of it: a larger one would
/// let shapes just behind a surface show through at grazing angles, where
/// a pixel spans a long depth range.
const MESH_DEPTH_TOLERANCE: f32 = 1.0 / 32.0;

fn hidden_by_meshes(position: vec4<f32>, sample: u32, slope: f32) -> bool {
    let mesh = textureLoad(mesh_depth, vec2<i32>(floor(position.xy)), i32(sample));
    return position.z > mesh + MESH_DEPTH_TOLERANCE * slope + 1e-7;
}

/// Mask passes render without a depth buffer: color (alpha) only.
@fragment
fn fs_mask(in: VertexOutput) -> @location(0) vec4<f32> {
    return premultiplied(shade(in));
}

fn premultiplied(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

fn shade(in: VertexOutput) -> vec4<f32> {
    // Loop-Blinn curve test: u^2 - v is negative on the chord side of the
    // quadratic and positive on the control-point side.
    if in.side != 0.0 {
        let implicit = in.uv.x * in.uv.x - in.uv.y;
        if implicit * in.side < 0.0 {
            discard;
        }
    }
    var color = in.color;
    if in.clip != NO_CLIP {
        let pixel = vec2<i32>(floor(in.clip_position.xy));
        color.a *= textureLoad(mask_layers, pixel, in.clip, 0).a;
    }
    // Fully masked fragments must not write depth and hide what is behind.
    if color.a <= 0.0 {
        discard;
    }
    return color;
}

// ---------------------------------------------------------------------------
// Fill: pre-tessellated triangles (see tessellation.py)
// ---------------------------------------------------------------------------

fn fill_vertex(object: Object, vertex_index: u32) -> VertexOutput {
    let vertex = fill_vertices[vertex_index];
    let world = (object.model * vec4<f32>(vertex.position, 0.0, 1.0)).xyz;

    var out: VertexOutput;
    out.clip_position = project(world);
    out.layer = object.layer;
    out.clip = object.clip;
    out.color = vertex.color;
    out.uv = vertex.uv;
    out.side = vertex.side;
    return out;
}

// ---------------------------------------------------------------------------
// Stroke: one instance per cubic curve, expanded into a flat ribbon. Widths
// are in world units and ignore the object's scale, like Manim's stroke
// widths.
//
// The ribbon is built around a "ribbon normal" n: its edges are offset along
// cross(n, tangent), and joint and cap fans rotate around n.
//
//   Shape2D: n is the shape's plane normal, so the ribbon lies in the plane.
//   Curve3D: n points from the curve towards the camera at every point, so
//            the ribbon always faces the viewer. Being in world units, it
//            gets thinner with distance under a perspective camera.
//
// Neighbouring curves evaluate n at their shared anchor, so they agree on
// the joint geometry.
//
// Vertex index layout of one curve instance:
//
//   [0, RIBBON)                 ribbon: SEGMENTS quads along the curve
//   [RIBBON, RIBBON + FAN_V)    start cap (only if there is no previous curve)
//   [RIBBON + FAN_V, END)       end joint with the next curve, or end cap
//
// Unused triangles are collapsed to CULLED.
// ---------------------------------------------------------------------------

const SEGMENTS: u32 = 24u;
const FAN_TRIANGLES: u32 = 8u;
const RIBBON_VERTICES: u32 = SEGMENTS * 6u;
const FAN_VERTICES: u32 = FAN_TRIANGLES * 3u;
// Keep in sync with STROKE_VERTEX_COUNT in vector_batch.py.
const STROKE_VERTEX_COUNT: u32 = RIBBON_VERTICES + 2u * FAN_VERTICES;

// Values of manimx.objects.style.JointType / CapStyle.
const JOINT_AUTO: u32 = 0u;
const JOINT_ROUND: u32 = 1u;
const JOINT_BEVEL: u32 = 2u;
const JOINT_MITER: u32 = 3u;
const CAP_ROUND: u32 = 1u;
const CAP_SQUARE: u32 = 3u;

/// Miters longer than this many half widths fall back to a bevel (Cairo's
/// default miter limit is 10).
const MITER_LIMIT: f32 = 10.0;
const PI: f32 = 3.14159265358979;

/// A cubic curve in world space with its per-control-point style.
struct Curve {
    p: array<vec3<f32>, 4>,
    width: vec4<f32>,
    color: array<vec4<f32>, 4>,
    normal: vec3<f32>,          // ribbon normal, see `oriented`
    facing: bool,               // FLAG_CAMERA_FACING
};

fn load_curve(object: Object, first_point: u32) -> Curve {
    var curve: Curve;
    for (var i = 0u; i < 4u; i++) {
        let point = stroke_points[first_point + i];
        curve.p[i] = (object.model * vec4<f32>(point.position, 1.0)).xyz;
        curve.width[i] = point.width;
        curve.color[i] = point.color;
    }
    let x_axis = (object.model * vec4<f32>(1.0, 0.0, 0.0, 0.0)).xyz;
    let y_axis = (object.model * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz;
    curve.normal = normalize(cross(x_axis, y_axis));
    curve.facing = (object.flags & FLAG_CAMERA_FACING) != 0u;
    return curve;
}

/// The curve with its ribbon normal set for a ribbon vertex at `point`.
fn oriented(curve: Curve, point: vec3<f32>) -> Curve {
    var result = curve;
    if curve.facing {
        result.normal = towards_camera(point);
    }
    return result;
}

/// Unit vector from `point` towards the viewer.
fn towards_camera(point: vec3<f32>) -> vec3<f32> {
    // Orthographic projections keep w = 1: every point sees the camera
    // along the same direction, the camera's backwards (+z) axis.
    if frame.projection[3][3] == 1.0 {
        return normalize((frame.inverse_view * vec4<f32>(0.0, 0.0, 1.0, 0.0)).xyz);
    }
    return normalize(frame.camera_position.xyz - point);
}

fn bernstein(t: f32) -> vec4<f32> {
    let s = 1.0 - t;
    return vec4<f32>(s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t);
}

fn curve_point(curve: Curve, t: f32) -> vec3<f32> {
    let b = bernstein(t);
    return b.x * curve.p[0] + b.y * curve.p[1] + b.z * curve.p[2] + b.w * curve.p[3];
}

fn curve_half_width(curve: Curve, t: f32) -> f32 {
    return 0.5 * dot(bernstein(t), curve.width);
}

fn curve_color(curve: Curve, t: f32) -> vec4<f32> {
    let b = bernstein(t);
    return b.x * curve.color[0] + b.y * curve.color[1] + b.z * curve.color[2] + b.w * curve.color[3];
}

/// Unit tangent. Handles that coincide with their anchor give a zero
/// derivative at the ends, so fall back to the next control point and
/// finally to the chord.
fn curve_tangent(curve: Curve, t: f32) -> vec3<f32> {
    let s = 1.0 - t;
    let derivative = 3.0 * (s * s * (curve.p[1] - curve.p[0])
        + 2.0 * s * t * (curve.p[2] - curve.p[1])
        + t * t * (curve.p[3] - curve.p[2]));
    if length(derivative) > 1e-9 {
        return normalize(derivative);
    }
    let fallback = select(curve.p[3] - curve.p[1], curve.p[2] - curve.p[0], t > 0.5);
    if length(fallback) > 1e-9 {
        return normalize(fallback);
    }
    return normalize(curve.p[3] - curve.p[0] + vec3<f32>(1e-9, 0.0, 0.0));
}

/// Unit vector to the left of the tangent, perpendicular to the ribbon normal.
fn left_of(curve: Curve, tangent: vec3<f32>) -> vec3<f32> {
    let left = cross(curve.normal, tangent);
    if length(left) < 1e-6 {
        // The curve points straight at the camera; any perpendicular will do.
        return normalize(cross(curve.normal, vec3<f32>(0.577, 0.577, 0.577)));
    }
    return normalize(left);
}

/// Rotate `v` (lying in the shape's plane) by `angle` around the plane normal.
fn rotate_in_plane(v: vec3<f32>, normal: vec3<f32>, angle: f32) -> vec3<f32> {
    return v * cos(angle) + cross(normal, v) * sin(angle);
}

fn is_miter(joint: u32) -> bool {
    return joint == JOINT_MITER || joint == JOINT_AUTO;
}

/// Offset direction (scaled, not unit) of the ribbon edge at a joint where
/// `own` meets `other`. Miter joints move both ribbons' corners to the shared
/// miter point; everything else keeps the curve's own normal.
fn joint_offset(own: vec3<f32>, other: vec3<f32>, joint: u32) -> vec3<f32> {
    if !is_miter(joint) {
        return own;
    }
    let bisector = own + other;
    if length(bisector) < 1e-6 {
        return own;  // a 180 degree turn has no finite miter
    }
    let miter = normalize(bisector);
    let scale = 1.0 / dot(miter, own);
    if scale > MITER_LIMIT {
        return own;  // too sharp: bevel instead
    }
    return miter * scale;
}

fn miter_succeeds(own: vec3<f32>, other: vec3<f32>, joint: u32) -> bool {
    if !is_miter(joint) {
        return false;
    }
    let bisector = own + other;
    return length(bisector) >= 1e-6 && 1.0 / dot(normalize(bisector), own) <= MITER_LIMIT;
}

fn stroke_vertex(object: Object, record: StrokeCurve, vertex_index: u32) -> VertexOutput {
    var curve = load_curve(object, record.first_point);

    var out: VertexOutput;
    out.uv = vec2<f32>(0.0);
    out.side = 0.0;
    out.layer = object.layer + 1.0;
    out.clip = object.clip;

    if vertex_index < RIBBON_VERTICES {
        let world = ribbon_position(object, curve, record, vertex_index);
        out.clip_position = project(world.xyz);
        out.color = curve_color(curve, world.w);
        return out;
    }

    let fan_vertex = (vertex_index - RIBBON_VERTICES) % FAN_VERTICES;
    let at_end = vertex_index >= RIBBON_VERTICES + FAN_VERTICES;
    let t = select(0.0, 1.0, at_end);
    let center = curve_point(curve, t);
    curve = oriented(curve, center);
    let half_width = curve_half_width(curve, t);
    let tangent = curve_tangent(curve, t);
    let left = left_of(curve, tangent);

    var offset: vec3<f32>;
    var valid: bool;
    if at_end && record.next_point != NO_NEIGHBOR {
        let next = load_curve(object, u32(record.next_point));
        let next_tangent = curve_tangent(next, 0.0);
        let result = joint_fan(curve, tangent, next_tangent, record.end_joint, fan_vertex);
        offset = result.xyz;
        valid = result.w > 0.0;
    } else if !at_end && record.prev_point != NO_NEIGHBOR {
        valid = false;  // the previous curve draws the shared joint
    } else {
        let cap = select(record.start_cap, record.end_cap, at_end);
        // Caps extend away from the curve: backwards at the start.
        let outward = select(-tangent, tangent, at_end);
        let result = cap_fan(left, outward, curve.normal, cap, fan_vertex);
        offset = result.xyz;
        valid = result.w > 0.0;
    }

    if !valid {
        out.clip_position = CULLED;
        out.color = vec4<f32>(0.0);
        return out;
    }
    out.clip_position = project(center + offset * half_width);
    out.color = curve_color(curve, t);
    return out;
}

/// World position (xyz) and curve parameter (w) of a ribbon vertex.
fn ribbon_position(object: Object, base: Curve, record: StrokeCurve, vertex_index: u32) -> vec4<f32> {
    // Two triangles per segment: (a-left, a-right, b-left), (b-left, a-right, b-right).
    let segment = vertex_index / 6u;
    let corner = vertex_index % 6u;
    let at_b = corner == 2u || corner == 3u || corner == 5u;
    let on_left = corner == 0u || corner == 2u || corner == 3u;

    let t = f32(segment + select(0u, 1u, at_b)) / f32(SEGMENTS);
    let curve = oriented(base, curve_point(base, t));
    let tangent = curve_tangent(curve, t);
    var offset = left_of(curve, tangent);

    // At the ends, share the miter point with the neighbouring curve.
    if t == 0.0 && record.prev_point != NO_NEIGHBOR {
        let previous = load_curve(object, u32(record.prev_point));
        let other = left_of(curve, curve_tangent(previous, 1.0));
        offset = joint_offset(offset, other, record.start_joint);
    } else if t == 1.0 && record.next_point != NO_NEIGHBOR {
        let next = load_curve(object, u32(record.next_point));
        let other = left_of(curve, curve_tangent(next, 0.0));
        offset = joint_offset(offset, other, record.end_joint);
    }

    let side = select(-1.0, 1.0, on_left);
    let world = curve_point(curve, t) + side * offset * curve_half_width(curve, t);
    return vec4<f32>(world, t);
}

/// Offset (xyz, in half widths) of a fan vertex filling the outer gap of a
/// joint; w = 0 marks an unused vertex.
fn joint_fan(curve: Curve, tangent: vec3<f32>, next_tangent: vec3<f32>, joint: u32, fan_vertex: u32) -> vec4<f32> {
    let own = left_of(curve, tangent);
    let other = left_of(curve, next_tangent);
    if miter_succeeds(own, other, joint) {
        return vec4<f32>(0.0);  // the ribbons already meet at the miter point
    }
    // The gap opens on the outside of the turn: right side for a left turn.
    let turns_left = dot(cross(tangent, next_tangent), curve.normal) > 0.0;
    let start = select(own, -own, turns_left);
    let end = select(other, -other, turns_left);
    let angle = atan2(dot(cross(start, end), curve.normal), dot(start, end));

    let triangle = fan_vertex / 3u;
    let corner = fan_vertex % 3u;
    if joint == JOINT_ROUND {
        return fan_corner(start, curve.normal, angle, triangle, corner);
    }
    // Bevel (and miters over the limit): a single triangle.
    if triangle > 0u {
        return vec4<f32>(0.0);
    }
    switch corner {
        case 0u: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
        case 1u: { return vec4<f32>(start, 1.0); }
        default: { return vec4<f32>(end, 1.0); }
    }
}

/// Offset (xyz, in half widths) of a cap vertex; w = 0 marks an unused vertex.
fn cap_fan(left: vec3<f32>, outward: vec3<f32>, normal: vec3<f32>, cap: u32, fan_vertex: u32) -> vec4<f32> {
    let triangle = fan_vertex / 3u;
    let corner = fan_vertex % 3u;
    if cap == CAP_ROUND {
        // Half circle from `left` through `outward` to `-left`.
        let direction = select(-1.0, 1.0, dot(cross(left, outward), normal) > 0.0);
        return fan_corner(left, normal, direction * PI, triangle, corner);
    }
    if cap == CAP_SQUARE && triangle < 2u {
        // Rectangle reaching one half width beyond the end.
        var corners = array<vec3<f32>, 6>(
            left, -left, left + outward,
            -left, -left + outward, left + outward,
        );
        return vec4<f32>(corners[fan_vertex], 1.0);
    }
    return vec4<f32>(0.0);  // butt caps add nothing
}

/// Corner of triangle `triangle` in a fan around the origin that sweeps
/// `angle` radians starting from `start`.
fn fan_corner(start: vec3<f32>, normal: vec3<f32>, angle: f32, triangle: u32, corner: u32) -> vec4<f32> {
    if corner == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let step = f32(triangle + corner - 1u) / f32(FAN_TRIANGLES);
    return vec4<f32>(rotate_in_plane(start, normal, angle * step), 1.0);
}

// ---------------------------------------------------------------------------
// Shadow maps: depth only, seen from a light (the frame uniform then holds
// the light's view, so camera-facing curves face the light). Fills keep
// their exact curve test; clipping is ignored (masks are screen space);
// translucent pixels cast a shadow from half opacity up.
// ---------------------------------------------------------------------------

@fragment
fn fs_shadow(in: VertexOutput) {
    if in.side != 0.0 && (in.uv.x * in.uv.x - in.uv.y) * in.side < 0.0 {
        discard;
    }
    if in.color.a < 0.5 {
        discard;
    }
}
