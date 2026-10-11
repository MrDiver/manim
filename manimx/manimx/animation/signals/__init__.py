"""Signals: the values every animatable property is made of, and how
animations change them over time.

| Module | What it holds |
|---|---|
| ``signal`` | ``Signal``, ``RootSignal`` (a value animations write), ``DerivedSignal`` (computed from others), ``Vector3Signal``, ``TimeSignals`` |
| ``kinds`` | ``SignalKind``: how values of one type interpolate and combine (``SCALAR``, ``COLOR``, ``PATH``, ...) |
| ``signal_tracks`` | ``SignalTrack``: the animations touching one root signal, combined over time |
| ``contributions`` | what an animation does to a signal, described as data for the core |
| ``rates`` | the default rate functions ``linear`` and ``smooth`` |
| ``paths`` | operations on cubic Bezier point arrays (the part of a path drawn so far) |

Everything here is re-exported by :mod:`manimx.animation`.
"""
