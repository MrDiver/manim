"""Scene nodes: the objects a scene is made of. Every property of a node
is a signal (manimx.animation.signals), so animations can change it.

| Module | What it holds |
|---|---|
| ``node`` | ``Node``, the base of every node, and the mixins for position, rotation and scale |
| ``group`` | ``Group``: a node that only holds others |
| ``clipping_group`` | ``ClippingGroup``: content shown only inside a mask |
| ``shape2d`` | ``Shape2D``: a vector shape in its local plane |
| ``curve3d`` | ``Curve3D``: a stroke-only path in 3D |
| ``shape3d`` | ``Shape3D``: a lit triangle mesh |
| ``particles`` | ``Particles``: many copies of one mesh |
| ``light`` | ``DirectionalLight``, ``PointLight``, ``SpotLight`` |
| ``camera`` | ``Camera``: what a scene is seen through |
| ``camera_view`` | ``CameraView``: another camera's picture on a plane |
| ``style`` | fill and stroke values of the nodes that have them |

Everything here is re-exported by :mod:`manimx.animation`.
"""
