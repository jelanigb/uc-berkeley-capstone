## Tests

Run the unit test suite from the repo root:

```bash
scripts/run_tests.sh           # all tests
scripts/run_tests.sh -v        # verbose
scripts/run_tests.sh -k engineer   # filter by name
```

Any extra args are forwarded to `pytest`.

### Pre-push hook

A `pre-push` git hook runs the suite automatically and aborts the push on
failure. Hooks are version-controlled under `scripts/hooks/`. After cloning,
point git at that directory once:

```bash
git config core.hooksPath scripts/hooks
```

Bypass in a pinch with `git push --no-verify`.
