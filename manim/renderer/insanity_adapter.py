"""The InsanityAdapter: the native wgpu renderer behind the interface the
Manager and Scene still expect from a renderer.

This is a bridge, and the only place that reads mobjects for drawing. On
each frame it turns the scene's mobjects into a render object tree
(``manim._native``) and the scene's camera into a :class:`.RenderConfig`,
and asks :class:`manim._native.WgpuRenderer` for the pixels. It goes away
when scenes produce render object trees themselves.

What the native renderer cannot draw yet raises :class:`.CapabilityException`
instead of being left out: images, point clouds, background images,
colours that vary across a mesh triangle, mobjects fixed in frame or in
orientation, and filled paths that do not lie in a plane.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import quaternion
from PIL import Image

from manim import _native as native
from manim._config import config
from manim.camera import Camera, ThreeDCamera
from manim.mobject.mobject import Mobject, _AnimationBuilder
from manim.mobject.three_d.mesh import MeshMobject
from manim.mobject.types.image_mobject import (
    AbstractImageMobject,
    ImageMobjectFromCamera,
)
from manim.mobject.types.point_cloud_mobject import PMobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.renderer._execution import _RendererExecutionView
from manim.renderer.protocol import (
    Camera as RenderCamera,
)
from manim.renderer.protocol import (
    CameraViewState,
    CapabilityException,
    RenderConfig,
)
from manim.scene.scene_file_writer import SceneFileWriter
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_difference_update, list_update
from manim.utils.transform import Transform

if TYPE_CHECKING:
    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.scene.scene import Scene
    from manim.typing import RGBAPixelArray

__all__ = ["InsanityAdapter"]

ROOT_UID = 1 << 62
"""The uid of the tree's root group: above every mobject uid."""
BACKGROUND_STROKE_UID = 1 << 61
"""Added to a mobject's uid for its background stroke."""
FACE_OUTLINES_UID = 1 << 60
"""Added to a mesh's uid for the outlines of its faces."""
MESH_PART_SHIFT = 32
"""Part ``i`` of a mesh drawn in several colours has the mesh's uid plus
``i << MESH_PART_SHIFT`` (mobject uids stay far below it)."""
PLANE_TOLERANCE = 1e-6
"""How far points may lie off a plane, relative to their spread, and still
be drawn as a 2D shape in it."""


class InsanityAdapter(_RendererExecutionView):
    """Draw scenes with the native wgpu renderer.

    Has the methods the Manager and Scene call on a renderer. The native
    renderer keeps what it built between frames, so there is no cached
    image of the mobjects that do not move: every frame draws them all.
    """

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        camera_class: type[Camera] | None = None,
        camera: Camera | None = None,
        skip_animations: bool = False,
    ) -> None:
        if camera is not None and camera_class is not None:
            raise ValueError("Pass either camera or camera_class, not both.")
        self._file_writer_class = file_writer_class
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera if camera is not None else camera_cls()
        self._initialize_execution(skip_animations)
        self._pixel_size = (int(config["pixel_width"]), int(config["pixel_height"]))
        self._renderer: native.WgpuRenderer | None = None
        self._frame: RGBAPixelArray | None = None
        # The Manager resets it after each play; nothing is cached here.
        self.static_image: RGBAPixelArray | None = None
        self._closed = False

    def _ensure_open(self) -> None:
        self._ensure_raster_allowed()
        if self._closed:
            raise RuntimeError("The renderer is closed.")

    def init_scene(self, scene: Scene, session_spec: RenderSessionSpec) -> None:
        self._ensure_open()
        if hasattr(self, "_scene"):
            raise RuntimeError("This renderer is already bound to a Scene.")
        self._scene = scene

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the scene manager's file writer, creating it if needed."""
        return self._scene._get_manager().file_writer

    def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        """Delegate playback to the scene's manager; prefer Scene.play for new code."""
        scene._get_manager()._play(*args, **kwargs)

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]:
        return "wgpu", ()

    def _start_animation(self) -> None:
        self._ensure_open()

    def _prepare_animation(self, scene: Scene) -> None:
        pass

    def _present_frame(self, scene: Scene, frame_offset: float) -> None:
        pass

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None:
        pass

    def update_frame(
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs: Any,
    ) -> None:
        """Draw the scene's mobjects, or ``mobjects`` if given."""
        self._ensure_open()
        if self.skip_animations and not ignore_skipping:
            return
        if not mobjects:
            mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        self._frame = self._draw(
            self.camera,
            mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=kwargs.get("excluded_mobjects"),
        )

    def render_mobjects(
        self,
        mobjects: Iterable[Mobject],
        *,
        camera: Camera | None = None,
    ) -> None:
        """Draw the supplied mobjects."""
        self._ensure_open()
        self._frame = self._draw(self.camera if camera is None else camera, mobjects)

    def render(
        self,
        scene: Scene,
        time: float,
        moving_mobjects: Iterable[Mobject] | None = None,
    ) -> None:
        # Every frame draws all mobjects (see the class docs).
        self.update_frame(scene)

    def get_frame(self) -> RGBAPixelArray:
        """The last drawn image as RGBA, with row zero at the top."""
        self._ensure_open()
        if self._frame is None:
            self._frame = self._draw(self.camera, [])
        return self._frame.copy()

    def _get_scene_image(self, scene: Scene) -> Image.Image:
        """Draw the scene's current mobjects with a temporary renderer, so
        snapshots work before the scene renders and after it closed.
        """
        self._ensure_raster_allowed()
        renderer = InsanityAdapter(camera=self.camera)
        renderer._pixel_size = self._pixel_size
        try:
            renderer.render_mobjects(
                list_update(scene.mobjects, scene.foreground_mobjects)
            )
            return renderer.get_image()
        finally:
            renderer.close()

    def get_image(self) -> Image.Image:
        """Return the last drawn image as a PIL image."""
        return Image.fromarray(self.get_frame())

    def add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        self._ensure_open()
        self._scene._get_manager()._legacy_add_frame(frame, num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        self.add_frame(
            self.get_frame(),
            num_frames=int(duration * self._scene.session_spec.frame_rate),
        )

    def show_frame(self, scene: Scene) -> None:
        self.update_frame(scene, ignore_skipping=True)
        self.get_image().show()

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: Iterable[Mobject],
    ) -> RGBAPixelArray | None:
        return None

    def update_skipping_status(self) -> None:
        self._scene._get_manager()._update_skipping_status()

    def scene_finished(self, scene: Scene) -> None:
        self._ensure_open()
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        elif not self.num_plays:
            self.update_frame(scene)

        if output.is_still or (not self.num_plays and output.fallback_to_still):
            if self.num_plays:
                self.update_frame(scene)
            self.file_writer.save_image(self.get_frame())

    def close(self) -> None:
        """Close the renderer and release the native renderer."""
        if self._closed:
            return
        self._closed = True
        self._renderer = None
        self._frame = None
        self.static_image = None

    # -- drawing --------------------------------------------------------------

    def _draw(
        self,
        camera: Camera,
        mobjects: Iterable[Mobject],
        *,
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> RGBAPixelArray:
        """Render ``mobjects`` as seen by ``camera``."""
        if camera.background_image is not None:
            raise CapabilityException(
                "The wgpu renderer cannot draw a camera background image."
            )
        members = _members(
            mobjects, camera.use_z_index, include_submobjects, excluded_mobjects
        )
        nodes = []
        views = []
        for mobject in members:
            nodes += _nodes(mobject)
            if isinstance(mobject, ImageMobjectFromCamera):
                views.append(_view_state(mobject))
        width, height = self._pixel_size
        render_config = RenderConfig(
            pixel_width=width,
            pixel_height=height,
            background_color=camera.background_color,
            background_opacity=camera.background_opacity,
            camera=_render_camera(camera),
            camera_views=tuple(views),
        )
        if self._renderer is None:
            self._renderer = native.WgpuRenderer()
        root = native.RGroup(uid=ROOT_UID, children=nodes)
        return self._renderer.render(root, render_config)


def _members(
    mobjects: Iterable[Mobject],
    use_z_index: bool,
    include_submobjects: bool,
    excluded_mobjects: list[Mobject] | None,
) -> list[Mobject]:
    """The mobjects to draw, in drawing order (as the Cairo renderer drew them)."""
    if not include_submobjects:
        return list(mobjects)
    members = extract_mobject_family_members(
        mobjects, use_z_index=use_z_index, only_those_with_points=True
    )
    if excluded_mobjects:
        members = list_difference_update(
            members,
            extract_mobject_family_members(excluded_mobjects, use_z_index=use_z_index),
        )
    return members


def _nodes(mobject: Mobject) -> list[native.RNode]:
    """The render objects that draw one mobject (without its submobjects)."""
    if mobject.fixed_in_frame or mobject.fixed_orientation is not None:
        raise CapabilityException(
            "The wgpu renderer cannot draw mobjects fixed in frame or in orientation."
        )
    if isinstance(mobject, VMobject):
        return _path_nodes(mobject)
    if isinstance(mobject, MeshMobject):
        return _mesh_nodes(mobject)
    if isinstance(mobject, ImageMobjectFromCamera):
        return [_view_node(mobject)]
    if isinstance(mobject, (AbstractImageMobject, PMobject)):
        raise CapabilityException(
            f"The wgpu renderer cannot draw a {type(mobject).__name__} yet."
        )
    # A plain Mobject has nothing to draw.
    return []


# -- paths ----------------------------------------------------------------------


def _path_nodes(vmobject: VMobject) -> list[native.RNode]:
    points = np.asarray(vmobject.points, dtype=float)
    # Cairo drew only whole curves (four points each).
    points = points[: len(points) // 4 * 4]
    if len(points) == 0:
        return []
    attributes: dict[str, Any] = vmobject.__dict__
    fill_rgbas = attributes.get("fill_rgbas", _NO_COLOR)
    stroke = _stroke(
        attributes.get("stroke_rgbas", _NO_COLOR),
        vmobject.get_stroke_width(),
        vmobject,
    )
    background_width = vmobject.get_stroke_width(background=True)
    background = (
        _stroke(
            attributes.get("background_stroke_rgbas", _NO_COLOR),
            background_width,
            vmobject,
        )
        if background_width > 0
        else None
    )
    filled = bool(np.any(np.asarray(fill_rgbas)[:, 3] > 0))
    plane = _plane(points)
    if plane is None:
        if filled:
            raise CapabilityException(
                "The wgpu renderer cannot fill a path that does not lie in a plane."
            )
        nodes = [
            native.RCurve3D(uid=vmobject.uid, points=points, stroke=stroke),
        ]
        if background is not None:
            nodes.insert(
                0,
                native.RCurve3D(
                    uid=vmobject.uid + BACKGROUND_STROKE_UID,
                    points=points,
                    stroke=background,
                ),
            )
        return nodes
    transform, local = plane
    nodes = [
        native.RShape2D(
            transform,
            uid=vmobject.uid,
            points=local,
            fill=_fill(fill_rgbas),
            stroke=stroke,
        )
    ]
    if background is not None:
        # Cairo drew the background stroke under the fill.
        nodes.insert(
            0,
            native.RShape2D(
                transform,
                uid=vmobject.uid + BACKGROUND_STROKE_UID,
                points=local,
                stroke=background,
            ),
        )
    return nodes


def _fill(rgbas: np.ndarray) -> native.Fill:
    rgbas = np.asarray(rgbas, dtype=float)
    return native.Fill(color=rgbas[:, :3], opacity=rgbas[:, 3])


def _stroke(rgbas: np.ndarray, width: float, vmobject: VMobject) -> native.Stroke:
    rgbas = np.asarray(rgbas, dtype=float)
    return native.Stroke(
        color=rgbas[:, :3],
        opacity=rgbas[:, 3],
        width=float(width),
        joint=vmobject.joint_type.value,
        cap=vmobject.cap_style.value,
    )


def _plane(points: np.ndarray) -> tuple[Transform, np.ndarray] | None:
    """Where the points lie in a plane: the plane's placement and the points
    in its local XY coordinates; None when they do not lie in one.
    """
    if len(points) == 0:
        return Transform(), np.zeros((0, 2))
    z = points[:, 2]
    spread = float(np.ptp(points, axis=0).max())
    if float(np.ptp(z)) <= PLANE_TOLERANCE * max(spread, 1.0):
        return Transform(position=(0.0, 0.0, float(z[0]))), points[:, :2]
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
    if singular[2] > PLANE_TOLERANCE * max(singular[0], 1.0):
        return None
    u, v = axes[0], axes[1]
    w = np.cross(u, v)
    rotation = quaternion.from_rotation_matrix(np.column_stack([u, v, w]))
    local = (points - center) @ np.column_stack([u, v])
    return Transform(position=tuple(center), rotation=rotation), local


_NO_COLOR = np.zeros((1, 4))
_NO_COLOR.flags.writeable = False


# -- meshes ---------------------------------------------------------------------


def _mesh_nodes(mesh: MeshMobject) -> list[native.RNode]:
    """One RShape3D per colour of the mesh (a material has one colour), and
    the outlines of its faces.
    """
    rgbas = np.asarray(mesh.rgbas, dtype=float)
    points = np.asarray(mesh.points, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    normals = _vertex_normals(points, triangles)
    has_texture = mesh.texture.size > 0 and len(mesh.uvs) == len(points)
    uvs = np.asarray(mesh.uvs, dtype=float) if has_texture else None
    texture = native.Texture(mesh.texture) if has_texture else None
    parts = (
        [(rgbas[0], triangles)]
        if len(rgbas) == 1
        else _triangles_by_colour(rgbas, triangles)
    )
    nodes: list[native.RNode] = []
    for index, (rgba, part) in enumerate(parts):
        data = native.MeshData(points, part.ravel(), normals, uvs=uvs)
        material = native.PbrMaterial(
            base_color=rgba[:3],
            opacity=float(rgba[3]),
            base_color_texture=texture,
        )
        uid = mesh.uid + (index << MESH_PART_SHIFT)
        nodes.append(native.RShape3D(uid=uid, mesh=data, material=material))
    if mesh.stroke_width > 0:
        nodes += _face_outlines(mesh, points)
    return nodes


def _triangles_by_colour(
    rgbas: np.ndarray, triangles: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The triangles of a mesh with a colour per vertex, grouped by colour;
    each triangle's three vertices must have the same colour (as on the
    faces of a checkerboard surface).
    """
    corner_colours = rgbas[triangles]
    if not np.all(corner_colours == corner_colours[:, :1]):
        raise CapabilityException(
            "The wgpu renderer cannot draw colours that vary across a triangle yet."
        )
    colours, which = np.unique(corner_colours[:, 0], axis=0, return_inverse=True)
    return [(colour, triangles[which == i]) for i, colour in enumerate(colours)]


def _vertex_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Per vertex, the area-weighted mean of the normals of its triangles."""
    normals = np.zeros_like(points)
    if len(triangles):
        a, b, c = (points[triangles[:, i]] for i in range(3))
        face_normals = np.cross(b - a, c - a)
        for i in range(3):
            np.add.at(normals, triangles[:, i], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    normals[lengths[:, 0] == 0] = (0.0, 0.0, 1.0)
    return normals


def _face_outlines(mesh: MeshMobject, points: np.ndarray) -> list[native.RCurve3D]:
    """The outlines of the mesh's faces, each a closed path of straight
    curves: one RCurve3D per outline colour (one for all faces, or one per
    face).
    """
    faces = np.asarray(mesh.get_faces(), dtype=np.int64)
    starts = points[faces]
    ends = points[np.roll(faces, -1, axis=1)]
    curves = np.stack(
        [starts, starts + (ends - starts) / 3, starts + 2 * (ends - starts) / 3, ends],
        axis=2,
    )
    stroke_rgbas = np.asarray(mesh.stroke_rgbas, dtype=float)
    if len(stroke_rgbas) == 1:
        parts = [(stroke_rgbas[0], curves)]
    else:
        colours, which = np.unique(stroke_rgbas, axis=0, return_inverse=True)
        parts = [(colour, curves[which == i]) for i, colour in enumerate(colours)]
    return [
        native.RCurve3D(
            uid=mesh.uid + FACE_OUTLINES_UID + (index << MESH_PART_SHIFT),
            points=part.reshape(-1, 3),
            stroke=native.Stroke(
                color=rgba[:3], opacity=rgba[3], width=float(mesh.stroke_width)
            ),
        )
        for index, (rgba, part) in enumerate(parts)
    ]


# -- camera views ---------------------------------------------------------------


def _view_frame(view: ImageMobjectFromCamera) -> tuple[Transform, float, float]:
    """The placement, width and height of a camera view's rectangle (its
    points are the corners: upper left, upper right, lower left, lower right).
    """
    upper_left, upper_right, lower_left, _ = np.asarray(view.points, dtype=float)
    across = upper_right - upper_left
    down = lower_left - upper_left
    width = float(np.linalg.norm(across))
    height = float(np.linalg.norm(down))
    u = across / width
    v = -down / height
    rotation = quaternion.from_rotation_matrix(np.column_stack([u, v, np.cross(u, v)]))
    center = np.asarray(view.points, dtype=float).mean(axis=0)
    return Transform(position=tuple(center), rotation=rotation), width, height


def _view_node(view: ImageMobjectFromCamera) -> native.RShape2D:
    transform, width, height = _view_frame(view)
    x, y = width / 2, height / 2
    corners = [(-x, -y), (x, -y), (x, y), (-x, y)]
    points = []
    for index in range(4):
        a, b = np.array(corners[index]), np.array(corners[(index + 1) % 4])
        points += [a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]
    return native.RShape2D(
        transform,
        uid=view.uid,
        points=np.array(points),
        fill=native.Fill(color=[1.0, 1.0, 1.0], opacity=1.0),
        stroke=native.Stroke(width=0.0),
    )


def _view_state(view: ImageMobjectFromCamera) -> CameraViewState:
    _, width, height = _view_frame(view)
    return CameraViewState(view.uid, _render_camera(view.camera), width, height)


# -- the camera -----------------------------------------------------------------


def _render_camera(camera: Camera) -> RenderCamera:
    """The renderer's snapshot of a scene camera.

    A :class:`ThreeDCamera` rotates the scene by its rotation matrix about
    the frame center, and projects it towards a viewer ``focal_distance``
    in front of the frame (see :meth:`.ThreeDCamera.project_points`): a
    perspective camera there, which shows ``frame_height / zoom`` at the
    frame center. Every other camera is orthographic.
    """
    center = np.asarray(camera.frame_center, dtype=float)
    if not isinstance(camera, ThreeDCamera):
        return RenderCamera(
            transform=Transform(position=tuple(center + (0.0, 0.0, 11.0))),
            frame_height=float(camera.frame_height),
        )
    focal_distance = float(camera.get_focal_distance())
    shown_height = float(camera.frame_height) / float(camera.get_zoom())
    to_world = np.asarray(camera.get_rotation_matrix(), dtype=float).T
    position = center + to_world @ np.array([0.0, 0.0, focal_distance])
    return RenderCamera(
        transform=Transform(
            position=tuple(position),
            rotation=quaternion.from_rotation_matrix(to_world),
        ),
        frame_height=shown_height,
        field_of_view=2 * float(np.arctan(shown_height / 2 / focal_distance)),
        far=max(100.0, 10 * focal_distance),
        focus_distance=focal_distance,
    )
