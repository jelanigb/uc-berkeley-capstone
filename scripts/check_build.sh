#!/usr/bin/env bash
# check_build.sh — verify pipeline imports and stage wiring for all scenarios.
# Run from repo root: bash scripts/check_build.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src/capstone"

python - <<'EOF'
import sys

# ── imports ───────────────────────────────────────────────────────────────────
try:
    from pipeline.factory import PipelineFactory, VALID_STAGE_NAMES_
    from pipeline.version_config import VersionConfig
except ImportError as e:
    print(f"FAIL  Import error: {e}", file=sys.stderr)
    sys.exit(1)

# ── expected stages per scenario ──────────────────────────────────────────────
EXPECTED = {
    "full_run": {
        "loader", "preprocessor", "raw_snapshotter", "engineer", "splitter",
        "scaler", "augmenter", "final_snapshotter", "trainer",
        "hyperparam_snapshotter", "model_snapshotter", "model_loader",
        "validator", "validation_results_snapshotter",
    },
    "retrain_existing_data": {
        "loader", "preprocessor", "raw_snapshotter", "engineer", "splitter",
        "scaler", "augmenter", "final_snapshotter", "trainer",
        "hyperparam_snapshotter", "model_snapshotter", "model_loader",
        "validator", "validation_results_snapshotter",
    },
    "tune_hyperparams": {
        "loader", "preprocessor", "raw_snapshotter", "engineer", "splitter",
        "scaler", "augmenter", "final_snapshotter", "trainer",
        "hyperparam_snapshotter", "model_snapshotter", "model_loader",
        "validator", "validation_results_snapshotter",
    },
    "validate_current": {
        "loader", "preprocessor", "engineer", "splitter", "scaler",
        "model_loader", "validator", "validation_results_snapshotter",
    },
    "retro_validate": {
        "loader", "preprocessor", "engineer", "splitter", "scaler",
        "model_loader", "validator", "validation_results_snapshotter",
    },
}

# ── build config (no GCS mutation) ────────────────────────────────────────────
try:
    config = VersionConfig.load().build()
except Exception as e:
    print(f"FAIL  VersionConfig.load().build() raised: {e}", file=sys.stderr)
    sys.exit(1)

# ── instantiate and check each scenario ──────────────────────────────────────
failures = []

scenarios = {
    "full_run":              lambda: PipelineFactory.full_run(config),
    "retrain_existing_data": lambda: PipelineFactory.retrain_existing_data(config),
    "tune_hyperparams":      lambda: PipelineFactory.tune_hyperparams(config),
    "validate_current":      lambda: PipelineFactory.validate_current(config),
    "retro_validate":        lambda: PipelineFactory.retro_validate(config, model_versions=["v1.0"]),
}

for name, factory_fn in scenarios.items():
    try:
        stages = factory_fn()
    except Exception as e:
        print(f"FAIL  {name}: instantiation raised: {e}")
        failures.append(name)
        continue

    present = {n for n in VALID_STAGE_NAMES_ if n in stages.__dict__}
    expected = EXPECTED[name]

    missing = expected - present
    extra = present - expected

    status = "OK  "
    notes = []
    if missing:
        status = "FAIL"
        notes.append(f"missing={sorted(missing)}")
    if extra:
        status = "WARN"
        notes.append(f"extra={sorted(extra)}")

    note_str = "  " + ", ".join(notes) if notes else ""
    print(f"{status}  {name:<26} stages={sorted(present)}{note_str}")

    if missing:
        failures.append(name)

if failures:
    print(f"\nFAIL  {len(failures)} scenario(s) failed: {failures}", file=sys.stderr)
    sys.exit(1)

print("\nOK")
EOF
