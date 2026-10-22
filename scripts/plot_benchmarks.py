"""Plot the results of scripts/benchmark_render.py.

    python scripts/plot_benchmarks.py

Reads ``benchmarks/results/*.json``, keeps the newest run of each
configuration (renderer, GPU, power mode) and writes to
``benchmarks/plots/``:

- ``moving.png``: milliseconds per frame of the scenes that move;
- ``draw.png``: the part of it the renderer itself draws (without
  turning mobjects into what it draws);
- ``first_frame.png``: seconds until the first frame of the still scenes;
- ``memory.png``: peak memory of the process per scene.

Each plot shows one panel per power mode, the renderers as bars.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
PLOTS = ROOT / "benchmarks" / "plots"

# Categorical slots 1-3 of a validated colour-blind-safe palette, in a
# fixed order: each configuration keeps its colour in every plot.
SERIES = {
    ("InsanityAdapter", "high-performance"): ("native, dedicated GPU", "#2a78d6"),
    ("InsanityAdapter", "low-power"): ("native, integrated GPU", "#eb6834"),
    ("CairoRenderer", "default"): ("Cairo", "#1baf7a"),
}
POWER_MODES = {"performance": "performance mode", "low-power": "energy-saving mode"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load() -> dict[tuple[str, str, str], dict]:
    """The newest run per (renderer, GPU, power mode)."""
    runs: dict[tuple[str, str, str], dict] = {}
    for path in sorted(RESULTS.glob("*.json")):  # names start with the time
        run = json.loads(path.read_text())
        renderers = {scene["renderer"] for scene in run["scenes"].values()}
        if len(renderers) != 1:
            continue
        runs[(renderers.pop(), run["gpu"], run["power"])] = run
    return runs


def plot(
    runs: dict,
    name: str,
    title: str,
    unit: str,
    scenes: list[str],
    value,
    powers: list[str],
    log: bool,
) -> None:
    figure, panels = plt.subplots(
        1,
        len(powers),
        figsize=(6.2 * len(powers), 0.55 * len(scenes) + 1.6),
        sharex=True,
        squeeze=False,
        facecolor=SURFACE,
    )
    height = 0.8 / len(SERIES)
    largest = max(
        value(numbers) or 0
        for run in runs.values()
        for scene, numbers in run["scenes"].items()
        if scene in scenes
    )
    for panel, power in zip(panels[0], powers, strict=True):
        panel.set_facecolor(SURFACE)
        for index, ((renderer, gpu), (label, colour)) in enumerate(SERIES.items()):
            run = runs.get((renderer, gpu, power))
            if run is None:
                continue
            for row, scene in enumerate(scenes):
                numbers = run["scenes"].get(scene)
                if numbers is None or value(numbers) is None:
                    continue
                y = row + (index - (len(SERIES) - 1) / 2) * height
                amount = value(numbers)
                panel.barh(
                    y,
                    amount,
                    height=height * 0.85,
                    color=colour,
                    label=label if row == 0 else None,
                )
                text = f"{amount:.0f}" if amount >= 10 else f"{amount:.2g}"
                at = amount * 1.06 if log else amount + 0.01 * largest
                panel.text(
                    at, y, text, va="center", ha="left", fontsize=7.5, color=MUTED
                )
        panel.set_yticks(range(len(scenes)), scenes, fontsize=9, color=INK)
        panel.invert_yaxis()
        if log:
            panel.set_xscale("log")
        panel.set_xlabel(unit, fontsize=9, color=MUTED)
        panel.set_title(
            POWER_MODES.get(power, power), fontsize=10, color=INK, loc="left"
        )
        panel.grid(axis="x", color=GRID, linewidth=0.8)
        panel.set_axisbelow(True)
        for side in ("top", "right", "left"):
            panel.spines[side].set_visible(False)
        panel.spines["bottom"].set_color(GRID)
        panel.tick_params(axis="x", colors=MUTED, labelsize=8)
        panel.tick_params(axis="y", length=0)
    handles, labels = panels[0][0].get_legend_handles_labels()
    # Title and legend take a fixed 0.75 inch at the top, whatever the height.
    inch = 1 / figure.get_figheight()
    figure.suptitle(
        title, x=0.01, y=1 - 0.08 * inch, ha="left", va="top", fontsize=12, color=INK
    )
    figure.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.005, 1 - 0.38 * inch),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.75 * inch))
    figure.savefig(PLOTS / name, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {(PLOTS / name).relative_to(ROOT)}")


def main() -> None:
    runs = load()
    PLOTS.mkdir(parents=True, exist_ok=True)
    scenes = list(next(iter(runs.values()))["scenes"])
    moving = [scene for scene in scenes if "still" not in scene]
    still = [scene for scene in scenes if "still" in scene]
    powers = [power for power in POWER_MODES if any(key[2] == power for key in runs)]
    plot(
        runs,
        "moving.png",
        "Time per frame, scenes that move (lower is better)",
        "ms per frame (log scale)",
        moving,
        lambda numbers: 1000 * numbers["seconds"] / max(numbers["frames"], 1),
        powers,
        log=True,
    )
    plot(
        runs,
        "draw.png",
        "Time the renderer itself draws, per frame (lower is better)",
        "ms per frame (log scale)",
        moving,
        lambda numbers: (
            1000 * numbers["draw_seconds"] / max(numbers["frames"], 1)
            if "draw_seconds" in numbers
            else None
        ),
        powers,
        log=True,
    )
    plot(
        runs,
        "first_frame.png",
        "Time until the first frame, still scenes (lower is better)",
        "seconds",
        still,
        lambda numbers: numbers["seconds"],
        powers,
        log=False,
    )
    plot(
        runs,
        "memory.png",
        "Peak memory of the process (lower is better)",
        "MiB",
        scenes,
        lambda numbers: numbers["peak_mib"],
        powers[:1],
        log=False,
    )


if __name__ == "__main__":
    main()
