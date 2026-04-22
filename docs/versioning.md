# Versioning Design

## Overview

Three entities are versioned independently, each with its own `major.minor`
counter in GCS at `config/versions.json`:

1. **Base Data** — feature-engineered final-dataset parquet (+ raw pull)
2. **Model** — trained model artifacts (model.pkl, scaler.pkl, metadata, features)
3. **Hyperparameters** — the param set fed to GridSearch / RandomizedSearch

Each version bumps independently based on which `SnapshotConfig` flags are
set in the notebook run.

## Versioning Philosophy

### Base Data — `v{major}.{minor}_{suffix}`
Example: `v3.2_mixed_80real`, `v1.0_real`

- **Major** — structural/schema change (new columns, renamed columns, changed types).
- **Minor** — new data pull using the same schema.

Stored in GCS under `snapshots/`. Files: `snapshots_{version_tag}_{rowcount}rows_{timestamp}.parquet`
with a `_meta.json` sidecar.

Note: prior to April 2026 the inverse convention was used (major = new pull,
minor = schema change). Historical `v1.x`–`v3.1` snapshots are grandfathered
under the old convention; new snapshots follow the rules above.

### Model — `v{major}.{minor}`
Example: `v3.2`, `v4.0_rf` (suffix `_rf` / `_lr_l1` / `_xgb` / `_ensemble` is
added by `save_model`; the base version is shared across all four models in
one training run).

- **Major** — trained on a new data version.
- **Minor** — same data, adjusted model properties or hyperparameters.

Stored in GCS under `models/{version_tag}_{model_type}/` as `model.pkl`,
`scaler.pkl`, `feature_cols.json`, `metadata.json`. Each metadata file records
the `data_snapshot` and `hyperparam_version` used, preserving the link back
to the specific data + params combination.

### Hyperparameters — `v{major}.{minor}`
Example: `v1.0`, `v1.3`, `v2.0`

- **Major** — a new parameter added to the grid.
- **Minor** — values adjusted for an existing parameter
  (e.g. `n_factors: [1,2,4]` → `[1,2,4,8]`).

Stored in GCS under `hyperparams/{version_tag}.json`. The file contains the
params dict for all four model classes (LR, RF, XGB, VotingClassifier) plus
a `search_config` record of the search strategy used to discover them.

As of the 2026-04-21 rebaseline, hyperparams starts at **v1.0**. The previous
shared-counter `v3.2` snapshot lives at that tag in GCS for history; its params
were re-saved as `v1.0` under the new scheme.

## GCS Layout

```
gs://maduros-dolce-capstone-data/
├── config/
│   └── versions.json                  # single source of truth; nested per-entity
├── snapshots/
│   ├── snapshots_v3.1_mixed_80real_965rows_*.parquet + _meta.json
│   ├── baselines_v3.1_real_*.parquet  + _meta.json
│   └── medians_v3.1_real_*.parquet    + _meta.json
├── models/
│   └── v3.2_rf/
│       ├── model.pkl
│       ├── scaler.pkl
│       ├── feature_cols.json
│       └── metadata.json
└── hyperparams/
    ├── v1.0.json
    └── v1.1.json
```

### `versions.json` schema

```json
{
  "data":        { "major": 3, "minor": 1, "raw_suffix": "real", "final_suffix": "mixed_80real" },
  "model":       { "major": 3, "minor": 1 },
  "hyperparams": { "major": 1, "minor": 0 },
  "last_updated":        "2026-04-21T12:00:00",
  "last_snapshot_types": ["models", "hyperparams"]
}
```

`SnapshotConfig.load()` auto-migrates two legacy shapes on first run:
(1) the flat schema `{"version": "3.1", ...}` → nested per-entity, and
(2) the earlier `data.mixed_suffix` key → `data.final_suffix` (a code-side
rename of the final-dataset suffix). Both migrations are lazy — GCS is only
rewritten when the next `commit()` runs.

## SnapshotConfig API

The config cell is the only place the user edits between runs. Typical usage:

```python
from utils.snapshot_config import SnapshotConfig

config = (
    SnapshotConfig.load()
    .snapshot_models()
    .snapshot_hyperparams()
    .use_data_version("3.1")    # reuse existing data, bump only model + hyperparams
    .build()
)
# ... notebook runs snapshots ...
config.commit()                  # at the end, persists bumped versions to GCS
```

### Bump-triggering methods

| Method                                 | Entity bumped | Bump size |
|----------------------------------------|---------------|-----------|
| `.snapshot_raw()` / `.snapshot_final()` | data          | minor     |
| `.snapshot_schema_change()`             | data          | major     |
| `.snapshot_models()`                    | model         | minor     |
| `.snapshot_models_new_data()`           | model         | major     |
| `.snapshot_hyperparams()`               | hyperparams   | minor     |
| `.snapshot_hyperparams_new_grid()`      | hyperparams   | major     |

Major bumps must be set **explicitly** — there is no auto-upgrade from a
data bump in the same session to a model major bump. This keeps bump behavior
fully user-controlled.

`.snapshot_schema_change()` is a modifier: it only has effect alongside
`.snapshot_raw()` or `.snapshot_final()`. Calling it alone still causes a data
major bump but makes no data write, which is rarely useful.

### Pinning methods

- `.use_data_version("3.1")` — pin data to an existing snapshot.
- `.use_model_version("3.1")` — pin model (rare).
- `.use_hyperparam_version("1.0")` — pin hyperparams.

Pinning errors if combined with the corresponding write flag (pinning reuses
an existing snapshot, which a write would overwrite).

### Presets

| Preset                | Flags set                                        | Result                       |
|-----------------------|--------------------------------------------------|------------------------------|
| `"dry_run"`           | none                                             | no bumps, reads only         |
| `"model_tuning"`      | tune + models + hyperparams                      | model minor, hyperparams minor |
| `"feature_engineering"` | final + tune + models + hyperparams            | data minor, model minor, hyperparams minor |
| `"new_raw_data"`      | raw + final + tune + models_new_data + hyperparams | data minor, model **major**, hyperparams minor |

Presets default to minor bumps except where noted. Chain
`.snapshot_schema_change()` / `.snapshot_hyperparams_new_grid()` after a preset
to upgrade to additional major bumps.

### Four bump scenarios

```python
# (a) Pull fresh data with same schema, retrain on it
SnapshotConfig.load() \
    .snapshot_raw() \
    .snapshot_final() \
    .snapshot_models_new_data() \
    .snapshot_hyperparams() \
    .build()
# → data v3.1 → v3.2, model v3.1 → v4.0, hyperparams v1.0 → v1.1

# (b) Schema change (new feature column added)
SnapshotConfig.load() \
    .snapshot_raw() \
    .snapshot_final() \
    .snapshot_schema_change() \
    .snapshot_models_new_data() \
    .build()
# → data v3.1 → v4.0, model v3.1 → v4.0

# (c) Tweak model properties against existing data
SnapshotConfig.load() \
    .snapshot_models() \
    .use_data_version("3.1") \
    .build()
# → data pinned to v3.1, model v3.1 → v3.2

# (d) Expand hyperparam grid (new param added)
SnapshotConfig.load() \
    .snapshot_hyperparams_new_grid() \
    .tune() \
    .build()
# → hyperparams v1.0 → v2.0
```

## Raw BigQuery data — not stored as parquet

The raw BQ pull is not archived as parquet. Rationale:

- The BQ `video_snapshots` table is the immutable archive (append-only,
  keyed by snapshot date).
- Old raw parquet can't exactly reproduce an old mixed snapshot anyway — the
  intermediate feature-engineering code evolves.
- Mixed snapshots (`snapshots/*.parquet`) preserve the exact training inputs.

If the BQ table ever becomes mutable, revisit this decision — a parquet
archive at each data major bump becomes worthwhile insurance.
