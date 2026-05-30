# Notebook Sequence

This folder holds the Capstone report notebooks as a numbered sequence. Each one
is a thin sequencer over the `pipeline/` package — it constructs a
`VersionConfig`, asks `PipelineFactory` for the right stages, and runs them in
order. All real logic lives in the `.py` files under `src/capstone/`.

| # | Notebook | Run it when… | Reads | Writes (persisting runs only) |
|---|----------|--------------|-------|----------------------------------|
| 01 | [`01_eda_raw_data.ipynb`](./01_eda_raw_data.ipynb) | only new data was collected | latest GCS raw snapshot | figures |
| 02 | [`02_feature_engineering_eda.ipynb`](./02_feature_engineering_eda.ipynb) | features changed or data refreshed | latest GCS raw snapshot | engineered splits, figures |
| 03 | [`03_model_training_results.ipynb`](./03_model_training_results.ipynb) | a model was added/changed | latest snapshot (or fresh BQ pull) | model + split snapshots, validation results, version bump |
| 04 | [`04_hyperparameter_tuning.ipynb`](./04_hyperparameter_tuning.ipynb) | optimizing the top models | latest snapshot | hyperparam + model snapshots, version bump |
| 05 | [`05_final_model_selection.ipynb`](./05_final_model_selection.ipynb) | choosing/justifying the final model | current model version from GCS | validation results |

## Typical workflow

- New/changed model → re-run **03**.
- Feature change → re-run **02**, then **03**.
- New data → see the section below — **03 must run first**.
- Comfortable with **03** → re-run **04** (tune the top models).
- Comfortable with **04** → re-run **05** (lock in the final model).

## Pulling new data

Pulling new data is a special case because of a versioning constraint: per
[`docs/versioning.md`](../../../../docs/versioning.md), training on a new data
version requires a **model major version bump** (`snapshot_models_new_data()`).
That bump belongs in the training notebook (03), not in the pre-modeling EDA
notebooks (01, 02). Introducing a model snapshot into 01 or 02 would be wrong —
those notebooks don't train.

Because of this, **notebook 03 must run first** when you pull new data. The
correct sequence is:

**Step 1 — Run notebook 03 in the new-data scenario.** In the config cell set
`RUN_MODE = "NEW_DATA"`. This pulls from BigQuery and persists a data minor bump
(+ baselines), the engineered splits, and a model MAJOR bump (retrained on new
data) — a data pull is always a real run, never a dry preview:

```python
RUN_MODE = "NEW_DATA"
# ...
config = (
    VersionConfig.load(use_synthetic=False)
    .snapshot_raw()             # BQ pull -> data minor bump + baselines bump
    .snapshot_final()           # persist engineered splits
    .snapshot_models_new_data() # model MAJOR bump (retrained on new data)
    .dry_run(False)
    .build()
)
stages = PipelineFactory.full_run(config)
```

Run the notebook to completion including `config.commit()` at the bottom. This
pulls from BigQuery, writes the new raw and engineered split snapshots to GCS,
trains and snapshots all models under the new major model version, and updates
`versions.json` so every subsequent `VersionConfig.load()` call sees the new
version numbers.

**Step 2 — Re-run notebooks 01 and 02** for updated EDA views with
`IS_DRY_RUN = True`. Because `VersionConfig.load()` now reads the bumped versions
from GCS, both notebooks automatically pick up the new snapshot. Keep
`IS_DRY_RUN = True` here — no additional writes are needed, and for notebook 02
in particular this is important: 03 already persisted the engineered splits and
bumped the data version, so an `IS_DRY_RUN = False` re-run of 02 would
**erroneously bump the data version a second time**. Only set 02 to
`IS_DRY_RUN = False` when the feature-engineering code itself changed.

**Step 3 — Run 04 and 05** as normal.

| Step | Notebook | Config |
| --- | --- | --- |
| 1 | **03** | `RUN_MODE = "NEW_DATA"` |
| 2 | 01, 02 | `IS_DRY_RUN = True` (re-run for updated EDA) |
| 3 | 04, 05 | `IS_DRY_RUN = False` when ready to persist |

After step 1, set `RUN_MODE` back to `"OLD_MODEL_OLD_DATA"` (or
`"NEW_MODEL_OLD_DATA"` when persisting a model change) so subsequent runs reuse
the committed snapshot instead of triggering another BQ pull.

## Conventions shared across the sequence

**`IS_DRY_RUN` guardrail.** Every notebook **except 03** defines
`IS_DRY_RUN = True` in its config cell and passes it into
`VersionConfig.dry_run(IS_DRY_RUN)` (notebook 03 has three scenarios rather than a
write/no-write toggle, so it uses a single `RUN_MODE` switch instead — see below).
While `True`,
all write operations are skipped — the snapshotter stages and `config.commit()`
print a "writing skipped" message instead of touching GCS, figures are not saved
to disk, and `build()` reports every version as unchanged (no version bump is
computed or shown). Because the guardrail is enforced centrally, the config cell
chains its `snapshot_*()` methods **unconditionally** — `dry_run()` alone decides
whether they take effect, so no cell branches on the flag. Flipping the one
switch to `False` is all it takes to persist a run.

**`.skip_eda()` (notebook 02 only).** Chain `.skip_eda()` onto the config builder
(e.g. `...dry_run(IS_DRY_RUN).skip_eda().build()`) to skip the post-engineering
EDA plots (the slow part) when you only need to rebuild or snapshot the
engineered data. Omit it (the default) to run the EDA. The EDA cells read it back
via `config.is_skip_eda`.

**Run modes (notebook 03).** A single `RUN_MODE` switch picks the scenario:
`"OLD_MODEL_OLD_DATA"` (default — reuse the committed GCS snapshot, view only, no
writes), `"NEW_MODEL_OLD_DATA"` (reuse the snapshot, persist a model minor bump),
or `"NEW_DATA"` (fresh BigQuery pull + retrain + persist everything). The training
and evaluation steps are identical across all three; a data pull is always a real
run, never a dry preview.

**Per-model tuning (notebook 04).** Set `MODELS_TO_TUNE` to the explicit subset
to search (e.g. `["xgb"]`). Models left out keep their current snapshot
parameters; all four families are still trained and evaluated.

**Evaluation metrics.** ROC-AUC is the primary selection metric. Every metrics
table leads with the **global (macro)** `precision_total` / `recall_total` /
`f1_total` — the unweighted mean across the above/below classes
(`average="macro"`) — followed by the per-class breakdown.

**Locked validation holdout.** A 30% holdout was created once at v4.0 and locked
by video ID in GCS, enabling apples-to-apples comparison across all model
versions. An **out-of-bounds (OOB)** holdout (unseen verticals) is planned for a
future pass; notebook 05 includes a guarded placeholder for it.

## Older notebooks

Superseded notebooks are kept under [`old/`](./old/) for reference and are not
part of the graded sequence.
