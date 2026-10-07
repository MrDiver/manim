# manimx Test TODO

## Syrupy visual snapshots

- [x] Add `syrupy` to test dependencies.
- [x] Enable pytest marker `visual`.
- [x] Add `tests/test_visual.py`.
- [x] Generate `tests/__snapshots__/test_visual.ambr`.
- [x] Replace static SVG fixture with real `manimx` renderer output.
- [x] Render deterministic scenes to PNG and compare with reference images.
- [ ] Keep fixture input deterministic: fixed size, camera, colors, seed, and frame.
- [ ] Test same input twice and confirm identical output.
- [ ] Change one shape property and confirm snapshot failure.
- [ ] Update snapshots only after intentional visual review.
- [ ] Add one visual test per shape.
- [ ] Add visual tests for create, fade, transform, and morph animation endpoints.

### How Syrupy works

- Pytest injects the `snapshot` fixture into the test.
- `assert value == snapshot` uses the test function name as snapshot key.
- Syrupy stores the serialized value in `tests/__snapshots__/test_visual.ambr`.
- Next run serializes current `value` and compares it with that stored value.
- Missing snapshot fails the test until created with `--snapshot-update`.
- Changed output fails the test and shows received versus stored output.
- `--snapshot-update` accepts current output as the new baseline. Review diff first.

Commands:

```bash
uv run pytest -m visual
uv run pytest -m visual --snapshot-update
```

## State tests

- [x] Add `state` pytest marker.
- [x] Add state fixture in `tests/conftest.py`.
- [x] Test copied state does not mutate original state.
- [x] Test object identity exists.
- [ ] Move state model from fixture into `manimx` package code.
- [ ] Add immutable `SceneSnapshot`.
- [ ] Add `state_at(time)` tests.
- [ ] Add `state_at(frame, frame_rate)` tests.
- [ ] Add create and delete object tests.
- [ ] Add overlapping animation tests.
- [ ] Add morph state tests.
- [ ] Confirm repeated queries return equal state.
- [ ] Confirm query order does not change state.

Command:

```bash
uv run pytest -m state
```

## Coverage and mutation

- [x] Enforce 90% coverage in `pyproject.toml`.
- [x] Add `mutmut`.
- [x] Configure mutmut to mutate only `manimx/`.
- [x] Disable coverage gate inside mutmut runs.
- [ ] Add production code before increasing mutation scope.
- [ ] Require all surviving mutants to have an explicit reason.

Commands:

```bash
uv run pytest
uv run mutmut run
uv run mutmut results
```
