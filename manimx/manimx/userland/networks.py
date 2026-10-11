"""Graphs in the sense of networks: vertices joined by edges.

:class:`Graph` is a Group of an edge per pair (a Line, or an Arrow in a
:class:`DiGraph`) under a vertex per name (a Dot, or a Dot with a Typst
label). The edges are connected to the vertices (``Line.connect``), so
moving a vertex, or the whole layout with :meth:`Graph.change_layout`,
carries its edges along.

Layouts place the vertices in the graph's own plane, scaled to fit
``layout_scale`` (the largest distance from the center):

* ``"circular"``: evenly around a circle, in the given order;
* ``"shell"``: circles inside each other, one per list of vertices in
  ``shells`` (the first innermost);
* ``"spring"``: Fruchterman and Reingold's force-directed placement
  (https://doi.org/10.1002/spe.4380211102), from a seeded start;
* ``"tree"``: layers by distance from ``root`` (top down), each vertex
  centered over its children, leaves evenly spaced;
* ``"random"``: seeded random positions;
* a dict from vertex to position, kept as given (not scaled).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from ..animation import Group, Node
from ..animation.nodes.node import ABS, ChangeList, animatable
from .lines import Arrow, Line
from .shapes2d import Dot
from .typst import Typst

Vertex = Hashable
Layout = str | Mapping[Vertex, Sequence[float]]
LAYOUTS = ("circular", "shell", "spring", "tree", "random")


class Graph(Group):
    """``vertices`` joined by ``edges`` (pairs of vertices), placed by
    ``layout`` (see the module docs).

    ``vertices[v]`` is vertex ``v``'s node and ``edges[(u, v)]`` the edge's
    (``graph[v]`` for short). Vertices are Dots of ``vertex_radius`` styled
    by ``vertex_style``; with ``labels`` (True for every vertex's name, or a
    dict of markup per vertex) each is a Group of a larger Dot and its
    Typst label (``label_style``, e.g. ``{"color": ...}``, and
    ``label_preamble``). Edges are styled by ``edge_style``.
    """

    def __init__(
        self,
        vertices: Sequence[Vertex],
        edges: Sequence[tuple[Vertex, Vertex]],
        layout: Layout = "spring",
        layout_scale: float = 2.0,
        root: Vertex | None = None,
        shells: Sequence[Sequence[Vertex]] | None = None,
        seed: int = 0,
        vertex_radius: float | None = None,
        vertex_style: Mapping[str, Any] | None = None,
        labels: bool | Mapping[Vertex, str] = False,
        label_style: Mapping[str, Any] | None = None,
        label_preamble: str = "",
        edge_style: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.vertex_names = list(dict.fromkeys(vertices))
        unknown = [pair for pair in edges if pair[0] not in self.vertex_names or pair[1] not in self.vertex_names]
        if unknown:
            raise ValueError(f"edges {unknown} join vertices that are not in the graph")
        self.edge_pairs = list(dict.fromkeys(tuple(pair) for pair in edges))
        self.layout_scale, self.seed, self.root, self.shells = float(layout_scale), seed, root, shells
        if labels is True:
            labels = {vertex: str(vertex) for vertex in self.vertex_names}
        labels = dict(labels or {})
        radius = vertex_radius if vertex_radius is not None else (0.25 if labels else 0.08)
        self.vertex_radius = float(radius)
        self.vertices: dict[Vertex, Node] = {}
        for vertex in self.vertex_names:
            dot = Dot(self.vertex_radius, f"{self.name}.{vertex}", **dict(vertex_style or {}))
            if vertex in labels:
                label = Typst(labels[vertex], label_preamble, height=self.vertex_radius, **dict(label_style or {}))
                self.vertices[vertex] = Group(dot, label, name=f"{self.name}.vertex.{vertex}")
            else:
                self.vertices[vertex] = dot
        self.edges: dict[tuple[Vertex, Vertex], Node] = {pair: self._edge(pair, dict(edge_style or {})) for pair in self.edge_pairs}
        positions = self.layout_positions(layout)
        for vertex, node in self.vertices.items():
            node.move_to(positions[vertex])
        for pair, edge in self.edges.items():
            edge.connect(self.vertices[pair[0]], self.vertices[pair[1]])
        # Edges first: the vertices are drawn over their ends.
        self.add(*self.edges.values(), *self.vertices.values())

    def _edge(self, pair: tuple[Vertex, Vertex], style: dict[str, Any]) -> Any:
        return Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), name=f"{self.name}.edge.{pair[0]}.{pair[1]}", **style)

    def __getitem__(self, vertex: Vertex) -> Node:
        return self.vertices[vertex]

    def layout_positions(self, layout: Layout) -> dict[Vertex, np.ndarray]:
        """Where ``layout`` puts every vertex (3D, z = 0)."""
        if isinstance(layout, Mapping):
            missing = [vertex for vertex in self.vertex_names if vertex not in layout]
            if missing:
                raise ValueError(f"the layout has no position for {missing}")
            return {vertex: _point(layout[vertex]) for vertex in self.vertex_names}
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}: one of {', '.join(LAYOUTS)}, or a dict of positions")
        count = len(self.vertex_names)
        if layout == "circular":
            places = _circle(count)
        elif layout == "shell":
            places = self._shell()
        elif layout == "spring":
            places = _spring(count, self._edge_indices(), self.seed)
        elif layout == "tree":
            places = self._tree()
        else:
            places = np.random.default_rng(self.seed).uniform(-1.0, 1.0, (count, 2))
        # Circles are centered on the origin already; their bounding box
        # (with an odd count) is not.
        fitted = _fit(places, self.layout_scale, center=layout not in ("circular", "shell"))
        return {vertex: _point(place) for vertex, place in zip(self.vertex_names, fitted, strict=True)}

    @animatable
    def change_layout(self, layout: Layout) -> ChangeList:
        """Move the vertices to ``layout`` (the edges follow):
        ``self.play(graph.animate.change_layout("circular"))``."""
        changes: ChangeList = []
        for vertex, position in self.layout_positions(layout).items():
            axes = self.vertices[vertex].position.axes  # type: ignore[attr-defined]
            changes += [(axis, ABS, float(component)) for axis, component in zip(axes, position, strict=True)]
        return changes

    def _edge_indices(self) -> list[tuple[int, int]]:
        index = {vertex: number for number, vertex in enumerate(self.vertex_names)}
        return [(index[u], index[v]) for u, v in self.edge_pairs if u != v]

    def _shell(self) -> np.ndarray:
        shells = [list(shell) for shell in (self.shells or [self.vertex_names])]
        listed = {vertex for shell in shells for vertex in shell}
        rest = [vertex for vertex in self.vertex_names if vertex not in listed]
        if rest:
            shells.append(rest)
        places: dict[Vertex, np.ndarray] = {}
        for ring, shell in enumerate(shells):
            radius = 0.0 if ring == 0 and len(shell) == 1 else ring + 1.0
            places.update(zip(shell, _circle(len(shell)) * radius, strict=True))
        return np.array([places[vertex] for vertex in self.vertex_names])

    def _tree(self) -> np.ndarray:
        """Layers by distance from the root, top down; leaves evenly spaced
        left to right, each parent centered over its children."""
        root = self.root if self.root is not None else self.vertex_names[0]
        if root not in self.vertex_names:
            raise ValueError(f"the root {root!r} is not a vertex")
        neighbors: dict[Vertex, list[Vertex]] = {vertex: [] for vertex in self.vertex_names}
        for u, v in self.edge_pairs:
            neighbors[u].append(v)
            neighbors[v].append(u)
        children: dict[Vertex, list[Vertex]] = {}
        depth = {root: 0}
        order = [root]
        for vertex in order:  # breadth first
            children[vertex] = [n for n in neighbors[vertex] if n not in depth]
            for child in children[vertex]:
                depth[child] = depth[vertex] + 1
                order.append(child)
        unreached = [vertex for vertex in self.vertex_names if vertex not in depth]
        if unreached:
            raise ValueError(f"a tree layout needs every vertex connected to the root; not {unreached}")
        x: dict[Vertex, float] = {}
        next_leaf = [0.0]

        def place(vertex: Vertex) -> float:
            if not children[vertex]:
                x[vertex], next_leaf[0] = next_leaf[0], next_leaf[0] + 1.0
            else:
                below = [place(child) for child in children[vertex]]
                x[vertex] = (below[0] + below[-1]) / 2
            return x[vertex]

        place(root)
        return np.array([[x[vertex], -float(depth[vertex])] for vertex in self.vertex_names])


class DiGraph(Graph):
    """A Graph whose edges are Arrows from the first vertex of each pair
    to the second, ending at the vertices' edges (``vertex_radius``)."""

    def _edge(self, pair: tuple[Vertex, Vertex], style: dict[str, Any]) -> Any:
        style.setdefault("buff", self.vertex_radius)
        return Arrow((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), name=f"{self.name}.edge.{pair[0]}.{pair[1]}", **style)


def _circle(count: int) -> np.ndarray:
    """``count`` points evenly around the unit circle, the first at the top."""
    angles = math.pi / 2 - 2 * math.pi * np.arange(count) / max(count, 1)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) if count > 1 else np.zeros((count, 2))


def _spring(count: int, edges: list[tuple[int, int]], seed: int, iterations: int = 200) -> np.ndarray:
    """Fruchterman and Reingold's force-directed placement in a unit
    square: neighbors attract by d^2 / k, every pair repels by k^2 / d,
    and a temperature that cools linearly caps each step."""
    rng = np.random.default_rng(seed)
    places = rng.uniform(-0.5, 0.5, (count, 2))
    if count < 2:
        return np.zeros((count, 2))
    k = math.sqrt(1.0 / count)
    temperature = 0.1
    for step in range(iterations):
        delta = places[:, None, :] - places[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-6)
        force = (delta / distance[..., None] * (k * k / distance)[..., None]).sum(axis=1)
        for u, v in edges:
            pull = delta[u, v] * distance[u, v] / k
            force[u] -= pull
            force[v] += pull
        length = np.maximum(np.linalg.norm(force, axis=1), 1e-9)
        places += force / length[:, None] * np.minimum(length, temperature * (1.0 - step / iterations))[:, None]
    return places


def _fit(places: np.ndarray, scale: float, center: bool = True) -> np.ndarray:
    """``places`` scaled so the farthest is ``scale`` from the origin;
    with ``center``, their bounding box centered on it first."""
    if len(places) == 0:
        return places
    centered = places - (places.min(axis=0) + places.max(axis=0)) / 2 if center else places
    reach = np.linalg.norm(centered, axis=1).max()
    return centered * (scale / reach) if reach > 1e-12 else centered


def _point(place: Sequence[float]) -> np.ndarray:
    point = np.zeros(3)
    values = np.asarray(place, dtype=np.float64)
    point[: len(values)] = values
    return point


__all__ = ["DiGraph", "Graph"]
