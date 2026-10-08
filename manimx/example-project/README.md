# example-project

A [manimx](https://github.com/ManimCommunity/manim) project.

```sh
uv sync                              # install manimx into .venv
uv run manimx generate scene Intro   # add a scene (file, registry, timeline)
uv run manimx render --preview       # quick low-resolution video of the timeline
uv run manimx render                 # full quality
```

Scenes live in `scenes/`, the timeline and render settings in `manimx.json`.
The walkthrough behind this project is [GETTING_STARTED.md](../GETTING_STARTED.md).

- `Intro`: the Manim logo assembling itself (animation basics)
- `Bounce`: balls with a hand-written simulation, an animated wall, stereo plops with reverb
- `Outro`: the animation template, as `manimx g scene outro` creates it
