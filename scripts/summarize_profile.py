"""Summarize a py-spy profile (speedscope format) as text.

    python scripts/summarize_profile.py benchmarks/py-spy/10000_squares_moving.json
    python scripts/summarize_profile.py PROFILE --top 40 --match insanity --match _native

Prints the functions with the most samples, inclusive (the function and
everything it calls) and self (the function alone), as a share of all
samples. ``--match`` keeps only functions whose name or file contains one
of the given words. For the whole picture, open the file at
https://www.speedscope.app.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--match", action="append", default=[])
    arguments = parser.parse_args()

    data = json.loads(arguments.profile.read_text())
    frames = data["shared"]["frames"]
    inclusive: collections.Counter[int] = collections.Counter()
    own: collections.Counter[int] = collections.Counter()
    total = 0
    for profile in data["profiles"]:
        for stack, weight in zip(profile["samples"], profile["weights"], strict=True):
            if not stack:  # a thread sampled outside Python
                continue
            total += weight
            own[stack[-1]] += weight
            for frame in set(stack):
                inclusive[frame] += weight

    def label(index: int) -> str:
        frame = frames[index]
        file = Path(frame.get("file") or "?").name
        return f"{frame['name']} ({file}:{frame.get('line', '?')})"

    def shown(index: int) -> bool:
        text = label(index) + (frames[index].get("file") or "")
        return not arguments.match or any(word in text for word in arguments.match)

    unit = data["profiles"][0].get("unit", "samples") if data["profiles"] else "samples"
    print(f"{arguments.profile}: {total:.1f} {unit} sampled")
    for title, counter in (("inclusive", inclusive), ("self", own)):
        print(f"\n{title}:")
        rows = [
            (index, weight) for index, weight in counter.most_common() if shown(index)
        ]
        for index, weight in rows[: arguments.top]:
            print(f"{100 * weight / total:6.1f}%  {label(index)}")


if __name__ == "__main__":
    main()
