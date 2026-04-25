#!/usr/bin/env bash
# Run the full unit test suite. Invocable from anywhere — cd's to the
# repo root first. Forwards any extra args to pytest, e.g.:
#   scripts/run_tests.sh -k feature_engineer
#   scripts/run_tests.sh -v
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec python -m pytest tests/ "$@"
