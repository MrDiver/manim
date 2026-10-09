# benchmark

A frozen copy of `example-project` (as of commit 2bd043e1), the project
`scripts/benchmark_project.py` measures and checks reference frames
against. `example-project` is for trying things out and changes; this
copy only changes together with a new baseline in PERFORMANCE.md.

```sh
uv run python scripts/benchmark_project.py tests/projects/benchmark
uv run python scripts/benchmark_project.py tests/projects/benchmark --check-reference /tmp/ref.npz
```
