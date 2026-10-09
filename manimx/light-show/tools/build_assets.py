"""Build the glTF models in ``assets/``: a moving head, a speaker and a mirror ball.

    uv run python tools/build_assets.py

The models are ordinary glTF 2.0 files (Y-up, front +Z, metres), the kind
Blender exports; the scenes load them with ``load_gltf`` like any other
model. They are generated here so the project ships no third-party art.
Every part an animation needs is its own named glTF node, and every
material is named, so a scene finds them with ``find("head")`` and
``find_all(tags="material:Lens")``:

* ``moving_head.glb``: ``base`` (the bracket on the truss) > ``yoke``
  (pans about its Y axis) > ``head`` (tilts about its X axis, the lens
  looks along -Y, straight down when hanging) > ``lens``
* ``speaker.glb``: ``cabinet`` with ``woofer`` and ``tweeter`` (a
  speaker's front is +Z)
* ``mirror_ball.glb``: ``ball``, about a thousand flat mirror tiles
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygltflib
from manimx.userland.geometry import box_mesh, surface_mesh, torus_mesh

ASSETS = Path(__file__).resolve().parent.parent / "assets"


@dataclass
class Part:
    """Triangles in a node's own space, all with one material."""

    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    material: str


@dataclass
class ModelNode:
    name: str
    parts: list[Part] = field(default_factory=list)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    children: list[ModelNode] = field(default_factory=list)


@dataclass
class Material:
    color: tuple[float, float, float]  # linear, as glTF stores it
    metallic: float
    roughness: float
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)


# -- shapes in glTF space ------------------------------------------------------


def _part(mesh, material: str, offset=(0.0, 0.0, 0.0), z_to_y: bool = False) -> Part:
    """A part from a manimx mesh, moved by ``offset``. Meshes of
    ``userland.geometry`` are built around +z; ``z_to_y`` turns that axis
    onto +y (glTF's up)."""
    positions, normals = (
        np.array(mesh.positions, np.float64),
        np.array(mesh.normals, np.float64),
    )
    if z_to_y:
        turn = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # z -> y, y -> -z
        positions, normals = positions @ turn, normals @ turn
    return Part(positions + offset, normals, np.array(mesh.indices, np.uint32), material)


def box(size, material: str, offset=(0.0, 0.0, 0.0)) -> Part:
    return _part(box_mesh(*size), material, offset)


def cylinder(
    radius: float,
    length: float,
    material: str,
    offset=(0.0, 0.0, 0.0),
    segments: int = 40,
) -> list[Part]:
    """A closed cylinder along y, centered at ``offset``."""
    tube = surface_mesh(
        lambda u, v: (radius * math.cos(u), radius * math.sin(u), v),
        (0.0, math.tau),
        (-length / 2, length / 2),
        (segments, 1),
    )
    parts = [_part(tube, material, offset, z_to_y=True)]
    for side in (-1.0, 1.0):
        parts.append(
            disc(
                radius,
                material,
                (offset[0], offset[1] + side * length / 2, offset[2]),
                facing=side,
                segments=segments,
            )
        )
    return parts


def disc(radius: float, material: str, offset, facing: float, segments: int = 40) -> Part:
    """A flat disc in the xz plane facing +y (``facing`` 1) or -y (-1)."""
    angles = np.linspace(0.0, math.tau, segments, endpoint=False)
    rim = np.column_stack([radius * np.cos(angles), np.zeros(segments), radius * np.sin(angles)])
    positions = np.vstack([[0.0, 0.0, 0.0], rim]) + offset
    normals = np.tile([0.0, facing, 0.0], (segments + 1, 1))
    triangles = [(0, 1 + (i + 1) % segments, 1 + i) if facing > 0 else (0, 1 + i, 1 + (i + 1) % segments) for i in range(segments)]
    return Part(positions, normals, np.array(triangles, np.uint32).reshape(-1), material)


def cone(radius: float, depth: float, material: str, offset, segments: int = 40) -> Part:
    """A speaker cone: a shallow funnel opening towards +z."""
    mesh = surface_mesh(
        lambda u, v: (
            v * radius * math.cos(u),
            v * radius * math.sin(u),
            depth * (v - 1.0),
        ),
        (0.0, math.tau),
        (0.15, 1.0),
        (segments, 4),
    )
    return _part(mesh, material, offset)


def ring(major: float, minor: float, material: str, offset) -> Part:
    """A torus around z (a speaker's surround, facing +z)."""
    return _part(torus_mesh(major, minor, 48, 12), material, offset)


# -- the models ------------------------------------------------------------------


def moving_head() -> tuple[ModelNode, dict[str, Material]]:
    materials = {
        "Housing": Material((0.02, 0.02, 0.025), metallic=0.3, roughness=0.45),
        "Trim": Material((0.6, 0.6, 0.62), metallic=1.0, roughness=0.3),
        "Lens": Material((0.05, 0.05, 0.05), metallic=0.0, roughness=0.05, emissive=(1.0, 1.0, 1.0)),
    }
    lens = ModelNode("lens", [disc(0.16, "Lens", (0.0, -0.236, 0.0), facing=-1.0)])
    head = ModelNode(
        "head",
        [
            *cylinder(0.2, 0.46, "Housing"),
            _part(torus_mesh(0.19, 0.025, 48, 12), "Trim", (0.0, -0.22, 0.0), z_to_y=True),
        ],
        translation=(0.0, -0.44, 0.0),
        children=[lens],
    )
    yoke = ModelNode(
        "yoke",
        [
            box((0.62, 0.06, 0.2), "Housing", (0.0, -0.03, 0.0)),
            box((0.06, 0.5, 0.2), "Housing", (-0.28, -0.28, 0.0)),
            box((0.06, 0.5, 0.2), "Housing", (0.28, -0.28, 0.0)),
            *cylinder(0.05, 0.12, "Trim", (0.0, -0.03, 0.0)),
        ],
        translation=(0.0, -0.18, 0.0),
        children=[head],
    )
    base = ModelNode(
        "base",
        [
            box((0.5, 0.16, 0.42), "Housing", (0.0, -0.1, 0.0)),
            box((0.12, 0.04, 0.5), "Trim", (0.0, 0.0, 0.0)),
        ],
        children=[yoke],
    )
    return base, materials


def speaker() -> tuple[ModelNode, dict[str, Material]]:
    materials = {
        "Cabinet": Material((0.012, 0.012, 0.014), metallic=0.0, roughness=0.85),
        "Cone": Material((0.03, 0.03, 0.035), metallic=0.0, roughness=0.6),
        "Surround": Material((0.005, 0.005, 0.005), metallic=0.0, roughness=0.9),
        "Trim": Material((0.6, 0.6, 0.62), metallic=1.0, roughness=0.3),
    }
    front = 0.5
    woofer = ModelNode(
        "woofer",
        [
            cone(0.36, 0.12, "Cone", (0.0, 0.0, 0.0)),
            ring(0.37, 0.03, "Surround", (0.0, 0.0, 0.0)),
            ring(0.42, 0.02, "Trim", (0.0, 0.0, 0.0)),
        ],
        translation=(0.0, -0.25, front),
    )
    tweeter = ModelNode(
        "tweeter",
        [
            cone(0.12, 0.05, "Cone", (0.0, 0.0, 0.0)),
            ring(0.14, 0.015, "Trim", (0.0, 0.0, 0.0)),
        ],
        translation=(0.0, 0.52, front),
    )
    cabinet = ModelNode(
        "cabinet",
        [box((1.2, 1.6, 1.0), "Cabinet", (0.0, 0.0, 0.0))],
        children=[woofer, tweeter],
    )
    return cabinet, materials


def mirror_ball(radius: float = 0.7, rings: int = 22) -> tuple[ModelNode, dict[str, Material]]:
    """Flat square tiles on a sphere with small gaps, each tile its own
    four vertices (flat normals), so every tile catches the light on its own."""
    materials = {
        "Mirror": Material((0.9, 0.9, 0.92), metallic=1.0, roughness=0.12),
        "Trim": Material((0.6, 0.6, 0.62), 1.0, 0.3),
    }
    positions, normals, indices = [], [], []
    for ring_index in range(1, rings - 1):
        phi0, phi1 = math.pi * ring_index / rings, math.pi * (ring_index + 1) / rings
        segments = max(6, round(2 * rings * math.sin((phi0 + phi1) / 2)))
        for segment in range(segments):
            theta0, theta1 = (
                math.tau * segment / segments,
                math.tau * (segment + 1) / segments,
            )
            corners = [
                _on_sphere(radius, p, t)
                for p, t in (
                    (phi0, theta0),
                    (phi0, theta1),
                    (phi1, theta1),
                    (phi1, theta0),
                )
            ]
            center = np.mean(corners, axis=0)
            corners = [center + 0.88 * (c - center) for c in corners]  # the gaps between tiles
            normal = center / np.linalg.norm(center)
            base = len(positions)
            positions += corners
            normals += [normal] * 4
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    tiles = Part(np.array(positions), np.array(normals), np.array(indices, np.uint32), "Mirror")
    caps = [disc(0.12, "Trim", (0.0, side * radius * 0.985, 0.0), facing=side) for side in (-1.0, 1.0)]
    hanger = cylinder(0.015, 2.0, "Trim", (0.0, radius + 1.0, 0.0), segments=8)
    return ModelNode("ball", [tiles, *caps, *hanger]), materials


def _on_sphere(radius: float, phi: float, theta: float) -> np.ndarray:
    """A point at polar angle ``phi`` from +y."""
    return radius * np.array(
        [
            math.sin(phi) * math.cos(theta),
            math.cos(phi),
            math.sin(phi) * math.sin(theta),
        ]
    )


# -- writing GLB -------------------------------------------------------------------


def write_glb(path: Path, root: ModelNode, materials: dict[str, Material]) -> None:
    gltf = pygltflib.GLTF2(asset=pygltflib.Asset(generator="light-show tools/build_assets.py"))
    blob = bytearray()
    names = list(materials)
    for name in names:
        material = materials[name]
        gltf.materials.append(
            pygltflib.Material(
                name=name,
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=[*material.color, 1.0],
                    metallicFactor=material.metallic,
                    roughnessFactor=material.roughness,
                ),
                emissiveFactor=list(material.emissive),
            )
        )

    def view(data: np.ndarray, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=data.nbytes, target=target))
        blob.extend(data.tobytes())
        return len(gltf.bufferViews) - 1

    def accessor(data: np.ndarray, kind: str, target: int, bounds: bool = False) -> int:
        component = pygltflib.UNSIGNED_INT if data.dtype == np.uint32 else pygltflib.FLOAT
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=view(data, target),
                componentType=component,
                count=len(data),
                type=kind,
                min=data.min(axis=0).tolist() if bounds else None,
                max=data.max(axis=0).tolist() if bounds else None,
            )
        )
        return len(gltf.accessors) - 1

    def add(node: ModelNode) -> int:
        mesh_index = None
        if node.parts:
            primitives = []
            for part in node.parts:
                positions = part.positions.astype(np.float32)
                normals = (part.normals / np.linalg.norm(part.normals, axis=1, keepdims=True)).astype(np.float32)
                primitives.append(
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(
                            POSITION=accessor(
                                positions,
                                pygltflib.VEC3,
                                pygltflib.ARRAY_BUFFER,
                                bounds=True,
                            ),
                            NORMAL=accessor(normals, pygltflib.VEC3, pygltflib.ARRAY_BUFFER),
                        ),
                        indices=accessor(
                            part.indices.astype(np.uint32),
                            pygltflib.SCALAR,
                            pygltflib.ELEMENT_ARRAY_BUFFER,
                        ),
                        material=names.index(part.material),
                    )
                )
            gltf.meshes.append(pygltflib.Mesh(name=node.name, primitives=primitives))
            mesh_index = len(gltf.meshes) - 1
        children = [add(child) for child in node.children]
        gltf.nodes.append(
            pygltflib.Node(
                name=node.name,
                mesh=mesh_index,
                children=children,
                translation=list(node.translation),
            )
        )
        return len(gltf.nodes) - 1

    gltf.scenes = [pygltflib.Scene(nodes=[add(root)])]
    gltf.scene = 0
    gltf.buffers = [pygltflib.Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(path))


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for name, (root, materials) in {
        "moving_head": moving_head(),
        "speaker": speaker(),
        "mirror_ball": mirror_ball(),
    }.items():
        path = ASSETS / f"{name}.glb"
        write_glb(path, root, materials)
        print(f"{path.relative_to(ASSETS.parent)}  {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
