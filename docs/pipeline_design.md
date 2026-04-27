# Pipeline Architecture Design
**UC Berkeley Capstone — YouTube Engagement Prediction**
*Draft — April 2026*

---

## Overview

The pipeline is restructured around three layers:

- **`VersionConfig`** — builder-pattern config carrying version intent (which versions
  to read, which bumps to write) plus run-wide flags (`use_synthetic`, search config).
  Renamed from `RunConfig` so it isn't confused with `PipelineRun`; the bulk of what it
  carries is versioning, even though a few non-version flags also live here.
- **`PipelineRun`** — typed dataclass carrying all state between stages (the data carrier)
- **Stage classes** — one class per CRISP-DM phase, each responsible for one concern

A **`PipelineFactory`** assembles the right combination of stages and injected logic for
each run scenario. The notebook is a thin sequencer — it constructs config, creates a
`PipelineRun`, asks the factory for stages, and calls `run()` in order. All core logic
lives in `.py` files outside the notebook.

---

## Structure

```
pipeline/
├── version_config.py               # VersionConfig builder (replaces utils/snapshot_config.py)
├── pipeline_run.py                 # PipelineRun dataclass — typed state carrier
├── factory.py                      # PipelineFactory + PipelineStages container
│
└── stages/
    ├── data_loader.py              # DataLoader (read-only: BQ for full_run, GCS otherwise)
    ├── raw_snapshotter.py          # RawSnapshotter (write-side counterpart to DataLoader)
    ├── data_preprocessor.py        # DataPreprocessor (pivot + baseline-join + cleanup)
    ├── feature_engineer.py         # FeatureEngineerLogic + FeatureEngineer
    ├── data_splitter.py            # DataSplitter (load-only holdout filter + train/test split)
    ├── scaler.py                   # Scaler (fit StandardScaler on X_train, transform all splits)
    ├── eda.py                      # Module-level plot functions (no class)
    ├── synthetic_augmenter.py      # SyntheticAugmenter (train split only)
    ├── final_snapshotter.py        # FinalSnapshotter (writes per-split X/y to GCS)
    ├── model_trainer.py            # ModelTrainerLogic + ModelTrainer
    ├── model_loader.py             # ModelLoader (load saved models from GCS)
    ├── model_snapshotter.py        # ModelSnapshotter (write-side counterpart to ModelLoader)
    ├── hyperparam_snapshotter.py   # HyperparamSnapshotter (saves search results)
    ├── validator.py                # ValidatorLogic + Validator
    │                               # RetroValidatorLogic (swappable via DI)
    └── validation_results_snapshotter.py  # ValidationResultsSnapshotter (append-only JSONL)

data_processing/                    # May be reorganized; pure transformation functions
utils/                              # May be reorganized; snapshot_data, snapshot_model, etc.
data_collection/                    # OFF LIMITS — harvester, baselines, discovery untouched
scripts/
    create_validation_set.py        # One-time holdout creation script (run before first training)
    check_build.sh                  # Import + wiring verification for all 5 scenarios
```

---

## Core Concepts

### PipelineRun — state carrier

A typed dataclass. Populated progressively as stages run. No transformation logic —
only state and introspection methods. The notebook holds one instance per run; stages
read from and write back to it.

```python
@dataclass
class PipelineRun:
    config: VersionConfig

    # Raw data (populated by DataLoader)
    df_videos:       Optional[pd.DataFrame] = None
    df_baselines:    Optional[pd.DataFrame] = None
    df_medians:      Optional[pd.DataFrame] = None

    # Cleaned wide-format data (populated by DataPreprocessor)
    df_clean:        Optional[pd.DataFrame] = None   # 1 row per video, post-pivot

    # Fully engineered data (populated by FeatureEngineer)
    df_engineered:   Optional[pd.DataFrame] = None   # 1 row per video, target computed

    # Splits (populated by DataSplitter)
    df_train:        Optional[pd.DataFrame] = None   # real only; augmented downstream
    df_test:         Optional[pd.DataFrame] = None   # real only, always
    df_val:          Optional[pd.DataFrame] = None   # real only, locked forever

    # Unscaled features (populated by DataSplitter; X_val_unscaled retained by Scaler)
    X_train:         Optional[pd.DataFrame] = None
    X_test:          Optional[pd.DataFrame] = None
    X_val:           Optional[pd.DataFrame] = None
    X_val_unscaled:  Optional[pd.DataFrame] = None   # pre-Scaler val features for Validator
    y_train:         Optional[pd.Series]    = None
    y_test:          Optional[pd.Series]    = None
    y_val:           Optional[pd.Series]    = None

    # Model artifacts (populated by ModelTrainer or ModelLoader)
    models:          dict = field(default_factory=dict)

    # Evaluation results (populated by Validator)
    results:         dict = field(default_factory=dict)

    def assert_ready_for(self, stage: Stage) -> None: ...
    def summary(self) -> None: ...
```

### Stage Classes — outer/inner split

Each stage with non-trivial logic has two sibling classes in one file. The outer class
handles orchestration (skip logic, snapshotting, logging). The inner `Logic` class holds
the core transformation and is unit-testable with no dependencies on config or GCS.

`DataLoader`, `DataPreprocessor`, `DataSplitter`, and `Scaler` have no `Logic` companion
— they are thin by nature. `EDA` is a module of plain functions with no class.

### DataPreprocessor — pivot + baseline-join + cleanup

`DataPreprocessor` runs immediately after `DataLoader` and before `FeatureEngineer`.
It is a thin wrapper around `build_clean_dataset` from `data_processing/data_cleanup.py`,
which chains: `pivot_snapshots` (drops incomplete-triplet videos, produces one wide row
per video) → `join_baselines` (attaches channel median baselines) → `clean_data`
(structural cleanup). Reads `run.df_videos` + `run.df_medians`, writes `run.df_clean`.

`pivot_snapshots` drops any video that does not have all three poll labels (`upload`,
`24h`, `7d`). After preprocessing, `df_clean` contains exactly those videos and the
`view_count_upload`, `baseline_median_*` columns that `FeatureEngineer` requires.

### FeatureEngineer — single-df input/output, no scaling

`FeatureEngineer` runs on `run.df_clean` (post-pivot, one row per video) and produces
`run.df_engineered`. It computes the `above_baseline` target, all ratio/growth features,
and categorical encodings via `engineer_features` from `data_processing/feature_engineering.py`.

**Narrowed responsibility vs. prior design:** `FeatureEngineer` no longer splits data,
fits a scaler, or exposes `transform_external`. Those concerns belong to `DataSplitter`
(X/y derivation) and `Scaler` (StandardScaler fit/transform) respectively.

`FeatureEngineerLogic.engineer(df, label)` is kept as a public method so
`SyntheticAugmenter` can engineer synthetic rows through the same chain.

`derive_feature_cols(df)` is a module-level function (single source of truth for
feature column selection by exclusion). `DataSplitter` imports it to derive `feature_cols`
without duplicating the exclusion logic.

### DataSplitter — load-only, post-engineering

`DataSplitter` is **load-only**: it assumes the validation holdout record already
exists in GCS at `splits/validation_ids.json`. If the record does not exist, the stage
raises with a clear pointer to `scripts/create_validation_set.py`, which must be run
once before any pipeline scenario.

`DataSplitter` operates on `run.df_engineered` (post-engineering, one row per video,
target column `above_baseline` present). It:

1. Loads the locked `video_id` list from GCS.
2. Filters to produce `df_val` (holdout videos) and `df_remaining` (all others).
3. Stratifies `df_remaining` into `df_train` / `df_test` (80/20) using the 18-cell key.
4. Derives `feature_cols = derive_feature_cols(df)` and splits each partition into
   unscaled `X_*` / `y_*` matrices on `run`.

**Stratification key (18 cells):** `vertical × tier × above_baseline` — the same key
used by `create_holdout()` at holdout-creation time, so stratification is consistent
across all runs and the holdout creation. Computed by the module-level `stratify_key(df)`
function, which is the single source of truth for both.

**Holdout creation** is deliberately NOT handled by `DataSplitter`. Run
`scripts/create_validation_set.py` once before the first training run.

**Load path notes:** the train/test pool excludes ALL recorded video_ids — not just
surviving ones. A video that disappears today and reappears later must never flip into
train, or it would leak between val and train across runs. If some recorded ids are
missing from `df_engineered` (deleted/private), `df_val` shrinks and a warning is
printed; the stored record is left untouched.

**Split proportions** (after first holdout creation):

| Split | Proportion of total | Notes |
|---|---|---|
| Validation | 30% | Fixed forever |
| Train | ~56% | Augmented with synthetic rows downstream |
| Test | ~14% | Real data only; re-split from remaining 70% on each run |

### Scaler — fit on X_train, transform all splits

`Scaler` runs immediately after `DataSplitter`. It fits a `StandardScaler` on
`run.X_train` and applies the same fitted transform to `X_train`, `X_test`, and `X_val`.
Before overwriting `X_val`, it captures the unscaled features as `run.X_val_unscaled`
so `Validator` can apply each loaded model's own historical scaler for a fair comparison.

Exposes `self.scaler_` and `self.feature_cols_` for downstream consumers:
- `SyntheticAugmenter` calls `scaler.transform(X_synth)` to scale synth rows.
- `ModelTrainer` stamps `entry["scaler"] = scaler.scaler_` onto each `run.models` entry
  so the scaler is saved alongside the model and reproduced at inference time.

### SyntheticAugmenter — train split only

Runs after `Scaler`, before `ModelTrainer`. Generates synthetic rows from `run.df_train`
(the real, post-engineering, unscaled training set) via `generate_synthetic_data`, then
engineers them through the same `FeatureEngineerLogic` chain and scales them using the
injected `Scaler`'s fitted `StandardScaler`. Appends to `run.X_train` / `run.y_train`
only. Test and validation splits are never touched. Is a no-op when
`config.use_synthetic = False`.

**Constructor injection:** `SyntheticAugmenter` takes `scaler: Scaler` (for
`scaler_.transform` and `feature_cols_`) and `logic: FeatureEngineerLogic` (for the
engineering chain). The factory wires both dependencies.

**Row-count formula:** `synth = floor(real / target_real_pct) - real`, where
`target_real_pct` is read from `config.target_real_pct`.

### ModelLoader — load saved model artifacts from GCS

Symmetric counterpart to `DataLoader` for model artifacts. Loads the family of
models tied to a base version string by auto-discovering all
`models/<base_version>_*/` directories in GCS, then reading each one's `model.pkl`,
`scaler.pkl`, `feature_cols.json`, and `metadata.json`. Populates `run.models`
keyed by the per-model suffix (e.g. `{"lr_l1": ..., "rf": ..., "xgb": ...,
"ensemble": ...}`).

The driving requirement is reproducibility: pairing a past model version with a
past data version must produce identical results, so loading must be a first-class
pipeline stage (not a side effect inside `Validator`).

**Scenarios that use it:**

- `validate_current` — loads `config.model_version` so `Validator` can score the
  current family against the locked validation set without retraining.
- `retro_validate` — loads each base version in `model_versions`, populating
  `run.models` with all of them so `RetroValidatorLogic` can replay them all
  against the same locked validation set.
- `retrain_existing_data` / `tune_hyperparams` — wired into the factory but
  optional in practice; callers that don't need a loaded model simply do not
  invoke `stages.model_loader.run(run)`.

### Snapshotter family — write-side counterparts

The pipeline has a symmetric write-side family of stages that persist artifacts to
GCS, each gated by its corresponding `take_snapshot_*` flag on `VersionConfig`.
Snapshotters never modify `PipelineRun` state — they read what previous stages
populated and write it out. When their flag is False, `run()` is a no-op.

| Snapshotter | Reads from `PipelineRun` | Writes to GCS at | Gate flag |
|---|---|---|---|
| `RawSnapshotter` | `df_videos`, `df_baselines`, `df_medians` | `snapshots/snapshots_<raw_version>_*.parquet` | `config.take_snapshot_raw` |
| `FinalSnapshotter` | `X_train`, `y_train`, `X_test`, `y_test`, `X_val`, `y_val` | six per-split parquet files under `snapshots/<final_version>_*` | `config.take_snapshot_final` |
| `ModelSnapshotter` | `run.models` (each model + scaler + feature_cols) | `models/<model_version>_<model_name>/` | `config.take_snapshot_models` |
| `HyperparamSnapshotter` | best params from `run.models` | `hyperparams/<hyperparam_version>/` | `config.take_snapshot_hyperparams` |
| `ValidationResultsSnapshotter` | `run.results` | `models/<model_version>/validation_results.jsonl` (append-only) | always runs |

**Why per-split for `FinalSnapshotter`:** the pipeline splits before engineering and
augmentation, so there is no single combined dataframe. Saving each split separately
preserves the structural boundary that the pipeline already enforces — synthetic rows
live in `X_train` only, val is locked, etc.

**Why these are stages, not utility calls:** lifting them into stage classes makes the
write side composable (one `.run(run)` call, automatic gating, centralized logging).
The stages delegate to existing utilities.

### PipelineFactory — wiring and scenario assembly

The single place where logic classes are instantiated and injected into stages.
`PipelineStages` is a **plain class** (not a dataclass) — absent stages are not set as
instance attributes and `__getattr__` raises a descriptive error on access.

Training scenarios share one `FeatureEngineerLogic` instance (injected into both
`FeatureEngineer` and `SyntheticAugmenter`) and one `Scaler` instance (injected into
both `SyntheticAugmenter` and `ModelTrainer`).

---

## Stage Sequence

**Training scenarios** (`full_run`, `retrain_existing_data`, `tune_hyperparams`):

```
DataLoader               ← read-only: BQ for full_run, GCS otherwise
    ↓
RawSnapshotter           ← gated by config.take_snapshot_raw; no-op otherwise
    ↓
DataPreprocessor         ← pivot + baseline-join + cleanup → run.df_clean
    ↓
FeatureEngineer          ← engineer_features on df_clean → run.df_engineered
    ↓
DataSplitter             ← load holdout; filter + split → df_train/test/val + X/y matrices
    ↓
Scaler                   ← fit on X_train; transform all splits; capture X_val_unscaled
    ↓
EDA (pre/post)           ← real data only; individual plot functions, each callable
    ↓
SyntheticAugmenter       ← appends synthetic rows to X_train / y_train only
    ↓
FinalSnapshotter         ← gated by config.take_snapshot_final; saves per-split parquet
    ↓
ModelTrainer             ← trains on X_train (real + synthetic); evaluates on X_test
    ↓
HyperparamSnapshotter    ← gated by config.take_snapshot_hyperparams
    ↓
ModelSnapshotter         ← gated by config.take_snapshot_models
    ↓
Validator                ← evaluates on X_val_unscaled + each model's own scaler
    ↓
ValidationResultsSnapshotter ← appends run.results to JSONL sidecar in GCS
```

**Validation-only scenarios** (`validate_current`, `retro_validate`):

```
DataLoader
    ↓
DataPreprocessor         ← pivot + baseline-join + cleanup → run.df_clean
    ↓
FeatureEngineer          ← engineer_features → run.df_engineered
    ↓
DataSplitter             ← load holdout; produce df_train/test/val + X/y matrices
    ↓
Scaler                   ← fit on X_train; transform splits; capture X_val_unscaled
    ↓
ModelLoader              ← reads model(s) from GCS; populates run.models
    ↓
Validator                ← evaluates on X_val_unscaled + each model's own scaler
    ↓
ValidationResultsSnapshotter ← appends run.results to JSONL sidecar in GCS
```

**BQ vs GCS by scenario:**

- `full_run` only — `DataLoader` reads from BigQuery
- All other scenarios — `DataLoader` reads from GCS parquet snapshot
- All scenarios — `DataSplitter` loads holdout IDs from GCS (errors if not found)
- `validate_current` / `retro_validate` — `ModelLoader` present, `ModelTrainer` absent
- `full_run` / `retrain_existing_data` / `tune_hyperparams` — `ModelTrainer` present;
  `ModelLoader` also wired for optional warm-start / baseline-comparison use

---

## Notebook Sequencing

```python
from pipeline.version_config import VersionConfig
from pipeline.pipeline_run import PipelineRun
from pipeline.factory import PipelineFactory
from pipeline.stages import eda

config = (
    VersionConfig.load(use_synthetic=True)
    .snapshot_raw()
    .snapshot_final()
    .snapshot_models_new_data()
    .build()
)
run = PipelineRun(config)
stages = PipelineFactory.full_run(config)

stages.loader.run(run)
stages.raw_snapshotter.run(run)        # no-op if config.take_snapshot_raw = False
stages.preprocessor.run(run)
stages.engineer.run(run)
stages.splitter.run(run)
stages.scaler.run(run)
run.summary()

# EDA — real data only
eda.plot_label_rates(run)
eda.plot_engagement_distribution(run)
eda.plot_feature_correlations(run)

stages.augmenter.run(run)              # no-op if config.use_synthetic = False
stages.final_snapshotter.run(run)      # no-op if config.take_snapshot_final = False
stages.trainer.run(run)
stages.hyperparam_snapshotter.run(run) # no-op if config.take_snapshot_hyperparams = False
stages.model_snapshotter.run(run)      # no-op if config.take_snapshot_models = False
stages.validator.run(run)
stages.validation_results_snapshotter.run(run)

config.commit()
```

---

## Run Scenarios

All training scenarios share the same active stage list — they differ in source
of truth (BQ vs GCS), what gets snapshotted, and tuning behavior.

| Scenario | Factory method | BQ load | Stages active |
|---|---|---|---|
| Full run (new data) | `full_run` | ✓ | loader → raw_snap → preprocessor → engineer → splitter → scaler → augment → final_snap → trainer → hyperparam_snap → model_snap → validator → val_results_snap (+ model_loader available) |
| Retrain, same data | `retrain_existing_data` | ✗ | same as full_run (loads parquet from GCS) |
| Hyperparameter tuning | `tune_hyperparams` | ✗ | same as full_run (search enabled via `config.tune_models`) |
| Validate current models | `validate_current` | ✗ | loader → preprocessor → engineer → splitter → scaler → model_loader → validator → val_results_snap |
| Replay saved models | `retro_validate` | ✗ | loader → preprocessor → engineer → splitter → scaler → model_loader → validator → val_results_snap (`RetroValidatorLogic` injected; loads multiple versions) |

---

## Interfaces

### `pipeline/version_config.py`
```python
class VersionConfig:
    # Builder methods (all return self)
    @classmethod
    def load(cls) -> "VersionConfig": ...
    def snapshot_raw(self, suffix: str = None) -> "VersionConfig": ...
    def snapshot_final(self, suffix: str = None) -> "VersionConfig": ...
    def snapshot_schema_change(self) -> "VersionConfig": ...
    def snapshot_models(self) -> "VersionConfig": ...
    def snapshot_models_new_data(self) -> "VersionConfig": ...
    def snapshot_hyperparams(self) -> "VersionConfig": ...
    def snapshot_hyperparams_new_grid(self) -> "VersionConfig": ...
    def tune(self, strategy: str = "random", n_iter: int = 50,
             cv: int = 5, scoring: str = "roc_auc") -> "VersionConfig": ...
    def use_data_version(self, version: str) -> "VersionConfig": ...
    def use_hyperparam_version(self, version: str) -> "VersionConfig": ...
    def build(self) -> "VersionConfig": ...
    def commit(self): ...

    # Flag properties
    @property def take_snapshot_raw(self) -> bool: ...
    @property def take_snapshot_final(self) -> bool: ...
    @property def take_snapshot_models(self) -> bool: ...
    @property def take_snapshot_hyperparams(self) -> bool: ...
    @property def tune_models(self) -> bool: ...
    @property def use_synthetic(self) -> bool: ...
    @property def target_real_pct(self) -> float: ...  # e.g. 0.8; drives final_version suffix

    # Version strings (set by build())
    raw_version: str        # e.g. "v3.1_real"
    final_version: str      # e.g. "v3.1_100real" or "v3.1_mixed_80real"
    model_version: str      # e.g. "v3.1"
    hyperparam_version: str # e.g. "v1.0"
```

### `pipeline/pipeline_run.py`
```python
class Stage(Enum):
    LOAD = "load"
    PREPROCESS = "preprocess"
    ENGINEER = "engineer"
    SPLIT = "split"
    SCALE = "scale"
    AUGMENT = "augment"
    TRAIN = "train"
    VALIDATE = "validate"

@dataclass
class PipelineRun:
    config: VersionConfig
    df_videos:      Optional[pd.DataFrame] = None
    df_baselines:   Optional[pd.DataFrame] = None
    df_medians:     Optional[pd.DataFrame] = None
    df_clean:       Optional[pd.DataFrame] = None   # post-DataPreprocessor
    df_engineered:  Optional[pd.DataFrame] = None   # post-FeatureEngineer
    df_train:       Optional[pd.DataFrame] = None
    df_test:        Optional[pd.DataFrame] = None
    df_val:         Optional[pd.DataFrame] = None
    X_train:        Optional[pd.DataFrame] = None
    X_test:         Optional[pd.DataFrame] = None
    X_val:          Optional[pd.DataFrame] = None
    X_val_unscaled: Optional[pd.DataFrame] = None   # captured by Scaler before transform
    y_train:        Optional[pd.Series]    = None
    y_test:         Optional[pd.Series]    = None
    y_val:          Optional[pd.Series]    = None
    models:         dict = field(default_factory=dict)
    results:        dict = field(default_factory=dict)

    def assert_ready_for(self, stage: Stage) -> None: ...
    def summary(self) -> None: ...
```

### `pipeline/factory.py`
```python
class PipelineStages:
    """Plain class — absent stages are not set as attributes; __getattr__ raises."""

    VALID_STAGE_NAMES = (
        "loader", "preprocessor", "raw_snapshotter",
        "engineer",
        "splitter",
        "scaler",
        "augmenter", "final_snapshotter",
        "trainer", "hyperparam_snapshotter", "model_snapshotter",
        "model_loader",
        "validator",
        "validation_results_snapshotter",
    )

    def __init__(self, scenario: str, **stages): ...
    def __getattr__(self, name: str): ...  # raises descriptive error for absent stages

class PipelineFactory:
    @staticmethod
    def full_run(config: VersionConfig) -> PipelineStages: ...
    @staticmethod
    def retrain_existing_data(config: VersionConfig) -> PipelineStages: ...
    @staticmethod
    def tune_hyperparams(config: VersionConfig) -> PipelineStages: ...
    @staticmethod
    def validate_current(config: VersionConfig) -> PipelineStages: ...
    @staticmethod
    def retro_validate(config: VersionConfig, model_versions: list[str]) -> PipelineStages: ...
```

### `pipeline/stages/data_loader.py`
```python
class DataLoader:
    """Read-only. Populates run.df_videos / df_baselines / df_medians.

    BQ vs GCS is decided by the `source` constructor argument:
      SOURCE_BQ  → BigQuery (full_run)
      SOURCE_GCS → GCS parquet at config.raw_version (all other scenarios)

    Writing back to GCS is the job of RawSnapshotter, not this stage.
    """
    SOURCE_BQ = "bq"
    SOURCE_GCS = "gcs"

    def __init__(self, config: VersionConfig, source: str = SOURCE_GCS): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/data_preprocessor.py`
```python
class DataPreprocessor:
    """Thin wrapper around build_clean_dataset (pivot + baseline-join + cleanup).

    Reads run.df_videos + run.df_medians.
    Writes run.df_clean (one wide row per video; only complete-triplet videos retained).
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/feature_engineer.py`
```python
def derive_feature_cols(df: pd.DataFrame) -> list:
    """Module-level. Returns feature column list by exclusion (single source of truth).
    Imported by DataSplitter to derive feature_cols without duplicating logic.
    """
    ...

class FeatureEngineerLogic:
    """Core transformation — testable with just DataFrames, no config or GCS.

    engineer(df, label) runs the full engineer_features chain on a single DataFrame.
    Used by both FeatureEngineer (on df_clean) and SyntheticAugmenter (on synth rows).
    """
    def engineer(self, df: pd.DataFrame, label: str = "") -> pd.DataFrame: ...

class FeatureEngineer:
    """Stage — runs FeatureEngineerLogic on run.df_clean, writes run.df_engineered."""
    def __init__(self, config: VersionConfig, logic: FeatureEngineerLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/data_splitter.py`
```python
def stratify_key(df: pd.DataFrame) -> pd.Series:
    """18-cell stratification key: vertical × tier × above_baseline.
    Single source of truth used by both DataSplitter (runtime) and create_holdout
    (one-time creation). Requires vertical, tier, and above_baseline columns.
    """
    ...

def create_holdout(
    df_engineered: pd.DataFrame,
    frac: float,
    store: "HoldoutStore",
    seed: int = 42,
    confirm: Callable[[], bool] = None,
) -> dict:
    """Stratified holdout split for one-time validation set creation.
    Used by scripts/create_validation_set.py. Not called during normal pipeline runs.
    Raises if any stratification cell has fewer than 2 rows.
    """
    ...

class HoldoutStore:
    """Persistence interface for the locked validation video_id record."""
    def exists(self) -> bool: ...
    def load(self) -> dict: ...
    def save(self, payload: dict) -> None: ...
    def location(self) -> str: ...

class GcsHoldoutStore(HoldoutStore):
    """Default production backend: splits/validation_ids.json in GCS."""
    DEFAULT_PATH = "splits/validation_ids.json"
    ...

class InMemoryHoldoutStore(HoldoutStore):
    """In-memory store for dry-run / testing. Never touches GCS. exists() → False."""
    ...

class DataSplitter:
    """Stage — load-only holdout filter + stratified train/test split.

    Raises RuntimeError (with pointer to create_validation_set.py) if holdout
    does not exist. Operates on run.df_engineered (post-engineering).
    Derives feature_cols via derive_feature_cols and populates all X/y matrices.
    """
    TEST_FRAC_OF_REMAINING = 0.20

    def __init__(self, config: VersionConfig, store: HoldoutStore = None, seed: int = 42): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/scaler.py`
```python
class Scaler:
    """Stage — fit StandardScaler on X_train, transform all splits.

    Exposes scaler_ and feature_cols_ for SyntheticAugmenter and ModelTrainer.
    Captures run.X_val_unscaled before transforming X_val so Validator can
    apply each loaded model's own historical scaler.
    """
    def __init__(self, config: VersionConfig): ...

    def run(self, run: PipelineRun) -> PipelineRun: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to an external DataFrame (e.g. synth rows)."""
        ...

    scaler_: StandardScaler       # set after run()
    feature_cols_: list           # set after run()
```

### `pipeline/stages/synthetic_augmenter.py`
```python
class SyntheticAugmenter:
    """Generates synthetic rows from run.df_train, engineers + scales them,
    and appends to run.X_train / run.y_train only.

    Constructor-injected with Scaler (for fitted scaler and feature_cols) and
    FeatureEngineerLogic (for the engineering chain).
    """
    def __init__(
        self,
        config: VersionConfig,
        scaler: "Scaler",
        logic: FeatureEngineerLogic = None,
        seed: int = 42,
    ): ...

    def run(self, run: PipelineRun) -> PipelineRun:
        # no-op if config.use_synthetic is False
        # otherwise:
        #   1. generate_synthetic_data(run.df_train, num_rows=...)
        #   2. logic.engineer(df_synth) → df_synth_eng
        #   3. X_synth = scaler.transform(df_synth_eng[feature_cols_])
        #   4. concat onto run.X_train / run.y_train
        ...
```

### `pipeline/stages/model_trainer.py`
```python
class ModelTrainerLogic:
    def run(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame,  y_test: pd.Series,
        config: VersionConfig,
        num_synth_rows: int = 0,
    ) -> dict:  # {model_name: partial_entry_dict}
        ...

class ModelTrainer:
    """Constructor-injected with Scaler to stamp scaler_ and feature_cols_ onto
    each run.models entry for correct inference at load time.
    """
    def __init__(self, config: VersionConfig, scaler: "Scaler", logic: ModelTrainerLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/model_loader.py`
```python
class ModelLoader:
    """Loads families of saved model artifacts from GCS by base version string."""

    def __init__(
        self,
        config: VersionConfig,
        versions: list[str] = None,
    ):
        # If `versions` is None, defaults to [config.model_version].
        ...

    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/validator.py`
```python
class ValidatorLogic:
    """Evaluates current trained models against the validation set."""
    def run(
        self,
        models: dict,
        X_val_unscaled: pd.DataFrame,
        y_val: pd.Series,
        config: VersionConfig,
    ) -> dict: ...

class RetroValidatorLogic:
    """Evaluates multiple saved model versions against the validation set.

    Loading is handled upstream by ModelLoader; this class consumes run.models
    and produces a per-version metrics breakdown for cross-version comparison.
    """
    def run(
        self,
        models: dict,
        X_val_unscaled: pd.DataFrame,
        y_val: pd.Series,
        config: VersionConfig,
    ) -> dict: ...

class Validator:
    def __init__(self, config: VersionConfig, logic: ValidatorLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/validation_results_snapshotter.py`
```python
class ValidationResultsSnapshotter:
    """Appends run.results as one JSON line to
    gs://<bucket>/models/<model_version>/validation_results.jsonl.
    Always runs (no gate flag). Raises if run.results is empty.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/raw_snapshotter.py`
```python
class RawSnapshotter:
    """Persists run.df_videos / df_baselines / df_medians to GCS at
    config.raw_version. No-op when config.take_snapshot_raw is False.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/final_snapshotter.py`
```python
class FinalSnapshotter:
    """Persists the six modeling artifacts (X_train, y_train, X_test, y_test,
    X_val, y_val) as separate parquet files at config.final_version, plus a
    metadata sidecar. No-op when config.take_snapshot_final is False.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/model_snapshotter.py`
```python
class ModelSnapshotter:
    """Persists each fitted model in run.models to GCS under
    models/<model_version>_<model_name>/. No-op when
    config.take_snapshot_models is False.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/hyperparam_snapshotter.py`
```python
class HyperparamSnapshotter:
    """Persists the hyperparameters used (or discovered via search) at
    config.hyperparam_version. No-op when config.take_snapshot_hyperparams is False.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

---

## Implementation Notes

**File operations:**

- `utils/snapshot_config.py` → `pipeline/version_config.py` via `git mv`. Rewrite the
  contents entirely; preserve the builder pattern and all existing public method names.
  Do not leave a stub or import alias in `utils/`.
- `data_collection/` is off-limits. Do not touch any file under that directory.
- `data_processing/` and `utils/` may be reorganized as needed to support the new
  pipeline package.

**PipelineStages absent-stage behavior:**

- `PipelineStages` is a plain class, not a dataclass. Stages absent from a given
  scenario are never set as instance attributes — Python's `__getattr__` then fires
  on access and raises a descriptive error. Do not use a dataclass with `None`
  defaults (lookup would succeed and `__getattr__` would never run) and do not
  override `__getattribute__` (recursion risk).
- Each instance carries a `scenario: str` attribute so error messages reference it.
- Error message format: `"ModelTrainer is not part of this pipeline scenario
  (validate_current). Check your PipelineFactory method."`
- Validate that all kwargs passed to `__init__` are recognized stage names; reject
  typos with a clear error.

**DataSplitter GCS path:**
- Bucket: `maduros-dolce-capstone-data`
- Full path: `gs://maduros-dolce-capstone-data/splits/validation_ids.json`
- Use `google-cloud-storage` (already a project dependency). Do not use `gsutil`.

**First-time holdout creation:**
- Run `python scripts/create_validation_set.py --yes` once before any pipeline scenario.
- The script runs Load → DataPreprocessor → FeatureEngineer (using the actual stage
  classes), then calls `create_holdout()` with `GcsHoldoutStore`.
- Dry-run mode (default, or `--dry-run`) calls `create_holdout()` with
  `InMemoryHoldoutStore` and prints the split plan without writing anything.
- Verify with `bash scripts/check_build.sh` after any pipeline code change.

**`pivot_snapshots` and incomplete triplets:**
- `DataPreprocessor` calls `build_clean_dataset`, which calls `pivot_snapshots`.
- `pivot_snapshots` drops any video not present in all three poll labels (`upload`,
  `24h`, `7d`). After preprocessing, `df_clean` contains only complete-triplet videos.
- The BQ `WHERE poll_label IS NOT NULL` filter is preserved in `DataLoader`'s BQ query
  to exclude old broken rows that still exist in BigQuery.

**VersionConfig `final_version` naming:**
- `final_version` is computed dynamically from `use_synthetic_` and `target_real_pct_`.
- `use_synthetic=False` → `final_version = "v3.1_100real"`
- `use_synthetic=True, target_real_pct=0.8` → `final_version = "v3.1_mixed_80real"`
- `target_real_pct` lives on `VersionConfig` (not `SyntheticAugmenter`), so the naming
  is consistent even when `SyntheticAugmenter` is not in the active scenario.

**Unit tests:**
- Write unit tests for all `Logic` classes using `pytest`.
- Place tests in `tests/pipeline/stages/` mirroring the stage file structure.
  Use `<module>_test.py` naming (suffix, not prefix).
- Tests must not require GCS access or a BigQuery connection.

**Existing `data_processing/` functions:**

- `build_clean_dataset` (`data_cleanup.py`) — called by `DataPreprocessor`.
- `pivot_snapshots` (`data_cleanup.py`) — called inside `build_clean_dataset`.
- `engineer_features` (`feature_engineering.py`) — called by `FeatureEngineerLogic.engineer`.
- `generate_synthetic_data` (`synthetic_data.py`) — called by `SyntheticAugmenter`.
- `combine_real_and_synthetic` is no longer used — the pipeline appends synth
  to `X_train` / `y_train` directly.

**Existing `utils/` snapshot helpers:**

- `RawSnapshotter` → `utils.snapshot_data` helpers.
- `FinalSnapshotter` → a per-split helper inside `utils.snapshot_data`.
- `ModelSnapshotter` → `utils.snapshot_model.save_model` (once per model in `run.models`).
- `HyperparamSnapshotter` → `utils.snapshot_hyperparameters.save_hyperparams`.
- `ValidationResultsSnapshotter` → `utils.snapshot_model.save_validation_results`
  (append-only JSONL; one JSON line per run).

---

## Design Principles

- **The notebook sequences; it does not implement.** All logic below the stage `run()`
  call lives in `.py` files.
- **Logic classes are independently testable.** `FeatureEngineerLogic().engineer(df)`
  requires no config, no GCS, no network.
- **The factory names scenarios.** New run types are new factory methods.
- **Cleanup and engineering always precede splitting.** `DataPreprocessor` and
  `FeatureEngineer` run on the full dataset before `DataSplitter` partitions it.
  This eliminates cross-split leakage and ensures the 18-cell stratification key
  (which requires `above_baseline`) is available at split time.
- **The validation set is immutable.** Created once via `scripts/create_validation_set.py`,
  persisted by video ID to GCS, never retrained against.
- **Synthetic rows never enter test or validation.** `SyntheticAugmenter` writes only
  to `run.X_train` / `run.y_train`. The boundary is enforced structurally.
- **Scaling is a separate stage.** `Scaler` sits between `DataSplitter` and
  `SyntheticAugmenter` / `ModelTrainer`. Unscaled `X_val` is retained as
  `X_val_unscaled` so Validator can apply each loaded model's own historical scaler.
- **BigQuery access is exclusive to `full_run`.** All other scenarios load from GCS
  parquet, keeping quota consumption predictable and non-BQ runs fast.
- **Reads and writes are separate stages.** Loaders never write; snapshotters never
  read external state for transformation.
- **Validation results are persisted.** `ValidationResultsSnapshotter` appends each
  run's metrics to an append-only JSONL sidecar, enabling historical tracking across
  model versions without a separate database.
