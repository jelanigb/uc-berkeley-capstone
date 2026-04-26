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
├── pipeline_run.py             # PipelineRun dataclass — typed state carrier
├── factory.py                  # PipelineFactory + PipelineStages container
│
└── stages/
    ├── data_loader.py          # DataLoader (read-only: BQ for full_run, GCS otherwise)
    ├── raw_snapshotter.py      # RawSnapshotter (write-side counterpart to DataLoader)
    ├── data_splitter.py        # DataSplitter (create or load holdout)
    ├── eda.py                  # Module-level plot functions (no class)
    ├── feature_engineer.py     # FeatureEngineerLogic + FeatureEngineer
    ├── synthetic_augmenter.py  # SyntheticAugmenter (train split only)
    ├── final_snapshotter.py    # FinalSnapshotter (writes per-split X/y to GCS)
    ├── model_trainer.py        # ModelTrainerLogic + ModelTrainer
    ├── model_loader.py         # ModelLoader (load saved models from GCS)
    ├── model_snapshotter.py    # ModelSnapshotter (write-side counterpart to ModelLoader)
    ├── hyperparam_snapshotter.py # HyperparamSnapshotter (saves search results)
    └── validator.py            # ValidatorLogic + Validator
                                # RetroValidatorLogic (swappable via DI)

data_processing/                # May be reorganized; pure transformation functions
utils/                          # May be reorganized; snapshot_data, snapshot_model, etc.
data_collection/                # OFF LIMITS — harvester, baselines, discovery untouched
project_pipeline.ipynb          # New working notebook (does not replace existing)
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
    df_videos:      Optional[pd.DataFrame] = None
    df_baselines:   Optional[pd.DataFrame] = None
    df_medians:     Optional[pd.DataFrame] = None

    # Splits (populated by DataSplitter — real data only)
    df_train:       Optional[pd.DataFrame] = None   # augmented with synthetic downstream
    df_test:        Optional[pd.DataFrame] = None   # real data only, always
    df_val:         Optional[pd.DataFrame] = None   # real data only, locked forever

    # Engineered features (populated by FeatureEngineer)
    df_model:       Optional[pd.DataFrame] = None   # full engineered train set, real only (EDA)
    X_train:        Optional[pd.DataFrame] = None
    X_test:         Optional[pd.DataFrame] = None
    X_val:          Optional[pd.DataFrame] = None
    y_train:        Optional[pd.Series]    = None
    y_test:         Optional[pd.Series]    = None
    y_val:          Optional[pd.Series]    = None

    # Model artifacts (populated by ModelTrainer)
    models:         dict = field(default_factory=dict)

    # Evaluation results (populated by Validator)
    results:        dict = field(default_factory=dict)

    def assert_ready_for(self, stage: str): ...  # fails fast with a clear message
    def summary(self): ...                        # prints current state of all fields
```

### Stage Classes — outer/inner split

Each stage with non-trivial logic has two sibling classes in one file. The outer class
handles orchestration (skip logic, snapshotting, logging). The inner `Logic` class holds
the core transformation and is unit-testable with no dependencies on config or GCS.

```python
# feature_engineer.py

class FeatureEngineerLogic:
    """Core transformation — testable with just DataFrames, no config or GCS."""
    def run(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # fit any stateful transforms on df_train only
        # apply same fitted transforms to df_test and df_val
        return df_train_eng, df_test_eng, df_val_eng

class FeatureEngineer:
    """Orchestration wrapper — the only thing the notebook imports."""
    def __init__(self, config: VersionConfig, logic: FeatureEngineerLogic = None):
        self.logic = logic or FeatureEngineerLogic()

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for("engineer")
        df_train, df_test, df_val = self.logic.run(
            run.df_train, run.df_test, run.df_val
        )
        run.X_train, run.y_train = self._split_xy(df_train)
        run.X_test,  run.y_test  = self._split_xy(df_test)
        run.X_val,   run.y_val   = self._split_xy(df_val)
        run.df_model = df_train  # real train only, for post-engineering EDA
        return run
```

`DataLoader` and `DataSplitter` have no `Logic` companion — they are thin by nature.
`SyntheticAugmenter` is also thin: it delegates to `generate_synthetic_data` from
`data_processing/synthetic_data.py` for row generation, then reuses the injected
`FeatureEngineer.transform_external` method to align the synth rows with `X_train`.
`combine_real_and_synthetic` is no longer used — the new pipeline appends synth
to `X_train` / `y_train` directly rather than building a combined real+synth frame.
`EDA` is a module of plain functions with no class.

### DataSplitter — three-way stratified holdout

`DataSplitter` runs on real data only, before feature engineering, and stratifies on
**vertical × tier cell** (9 cells). The `above_baseline` label is not yet available
at split time; cell-level stratification is sufficient since the target is a
within-channel median split and is approximately balanced within each cell by design.

**Split proportions:**

| Split | Proportion of total | Est. rows (16,204 today) | Notes |
|---|---|---|---|
| Validation | 30% | ~4,860 | Fixed forever after first `full_run` |
| Train | 56% | ~9,070 | Augmented with synthetic rows downstream |
| Test | 14% | ~2,270 | Real data only; resampled each run |

The 70% non-holdout pool is split 80/20 into train and test on every run.

**Two code paths:**

- **Create** (first `full_run` ever): splits the dataset, persists holdout video IDs
  to GCS at `gs://maduros-dolce-capstone-data/splits/validation_ids.json` alongside
  split metadata. Writes `df_val`, `df_train`, `df_test` to `PipelineRun`.
- **Load** (every subsequent run): reads persisted IDs from GCS, partitions current
  dataset by membership. Videos not in the holdout ID list form the train/test pool,
  re-split 80/20 as data accrues. Warns (does not error) if loaded validation set
  is smaller than the count recorded at creation time.

`validation_ids.json` schema:
```json
{
  "created_at": "2026-04-23T...",
  "seed": 42,
  "total_val_rows": 4860,
  "rows_per_cell": { "Education_S": 340, "Education_M": 490, ... },
  "video_ids": ["abc123", "def456", ...]
}
```

### SyntheticAugmenter — train split only

Runs after `FeatureEngineer`, before `ModelTrainer`. Generates synthetic rows via
`GaussianCopulaSynthesizer` (delegates to `data_processing.synthetic_data`) and
appends them to `run.X_train` / `run.y_train` only. Test and validation splits
are never touched. Is a no-op when `config.use_synthetic = False`.

**Coupling to `FeatureEngineer`:** synthetic rows must end up shaped exactly like
the real `X_train` — same engineered columns, same categorical encoding, scaled
by the same fitted `StandardScaler`. Re-implementing that logic here would risk
silent drift, so `SyntheticAugmenter` is constructor-injected with the
`FeatureEngineer` instance and reuses its public `transform_external(df)` method
to engineer + scale synth rows after generation. The factory is responsible for
wiring this dependency.

Generation works against `run.df_model` (real, engineered, unscaled — populated by
`FeatureEngineer` for EDA) since the SDV synthesizer needs the channel and
baseline context that's already been computed by that stage.

### ModelLoader — load saved model artifacts from GCS

Symmetric counterpart to `DataLoader` for model artifacts. Loads the family of
models tied to a base version string by auto-discovering all
`models/<base_version>_*/` directories in GCS, then reading each one's `model.pkl`,
`scaler.pkl`, `feature_cols.json`, and `metadata.json`. Populates `run.models`
keyed by the per-model suffix (e.g. `{"lr_l1": ..., "rf": ..., "xgb": ...,
"ensemble": ...}`).

The "base version" pattern is dictated by how `ModelSnapshotter` (via the existing
`save_model` utility) names artifacts: a single bump produces multiple model
directories like `v3.1_lr_l1`, `v3.1_rf`, `v3.1_xgb`, `v3.1_ensemble`. Loading
`v3.1` means loading all four.

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
  optional in practice; callers that don't need a loaded model simply don't
  invoke `stages.model_loader.run(run)`.

**Position in the sequence:** runs after `FeatureEngineer` (so `X_val` / `y_val`
exist) and before `Validator`. Does not depend on `ModelTrainer` and is mutually
exclusive with it in `validate_current` / `retro_validate`.

### Snapshotter family — write-side counterparts

The pipeline has a symmetric write-side family of stages that persist artifacts to
GCS, each gated by its corresponding `take_snapshot_*` flag on `VersionConfig`.
Snapshotters never modify `PipelineRun` state — they read what previous stages
populated and write it out. When their flag is False, `run()` is a no-op (the
notebook always calls them; the gate lives inside the stage).

| Snapshotter | Reads from `PipelineRun` | Writes to GCS at | Gate flag |
|---|---|---|---|
| `RawSnapshotter` | `df_videos`, `df_baselines`, `df_medians` | `snapshots/snapshots_<raw_version>_*.parquet` and friends | `config.take_snapshot_raw` |
| `FinalSnapshotter` | `X_train`, `y_train`, `X_test`, `y_test`, `X_val`, `y_val` | six per-split parquet files under `snapshots/<final_version>_*` | `config.take_snapshot_final` |
| `ModelSnapshotter` | `run.models` (each model + its associated scaler / feature_cols) | `models/<model_version>_<model_name>/` | `config.take_snapshot_models` |
| `HyperparamSnapshotter` | best params from `run.models` (or from search results) | `hyperparams/<hyperparam_version>/` | `config.take_snapshot_hyperparams` |

**Why per-split for `FinalSnapshotter`:** the old notebook saved one combined
`df_combined` (real + synthetic, post-engineering). The new pipeline splits
*before* engineering and augmentation, so there's no longer a single dataframe
that captures the modeling table. Saving each split separately preserves the
structural boundary that the pipeline already enforces — synthetic rows live in
`X_train` only, val is locked, etc. — and lets retro replays load only the parts
they need (e.g., `X_val` + `y_val` alone for revalidation against an old data version).

**Why these are stages, not utility calls:** the existing `save_*` utilities work,
but lifting them into stage classes makes the write side composable with the rest
of the pipeline (one `.run(run)` call per phase, automatic gating, one place to
centralize logging and error handling). The stages delegate to the existing
utilities — they don't reimplement the disk/GCS logic.

**Position in the sequence:** each snapshotter sits immediately after the stage
whose output it persists (see the Stage Sequence section below).

Implementation lands in a future coding pass; this section documents the
design SOT.

### PipelineFactory — wiring and scenario assembly

The single place where logic classes are instantiated and injected into stages.
`PipelineStages` is a **plain class** (not a dataclass) — when a stage is absent
for a given scenario, the corresponding attribute is simply not set on the
instance. Python's `__getattr__` is then invoked on access and raises a
descriptive error:

```
ModelTrainer is not part of this pipeline scenario (validate_current).
Check your PipelineFactory method.
```

Going non-dataclass for two reasons: (1) a dataclass field set to `None` is a
successful attribute lookup, so `__getattr__` would never fire for it, and
overriding `__getattribute__` instead carries a real recursion risk; (2)
`PipelineStages` needs to carry the scenario name so the error message can
reference the factory method that built it — `scenario` is just an `__init__`
argument on a plain class.

```python
class PipelineFactory:

    @staticmethod
    def full_run(config: VersionConfig) -> PipelineStages:
        """Load fresh data from BQ, split, engineer, augment, train, validate."""

    @staticmethod
    def retrain_existing_data(config: VersionConfig) -> PipelineStages:
        """Load from GCS parquet snapshot (no BQ). Re-split, engineer, augment,
        train, validate."""

    @staticmethod
    def tune_hyperparams(config: VersionConfig) -> PipelineStages:
        """Retrain with hyperparameter search enabled on existing data snapshot."""

    @staticmethod
    def validate_current(config: VersionConfig) -> PipelineStages:
        """Validation stage only. No trainer — accessing stages.trainer raises."""

    @staticmethod
    def retro_validate(
        config: VersionConfig,
        model_versions: list[str],
    ) -> PipelineStages:
        """Reload saved model versions from GCS, replay against fixed validation set.
        Injects RetroValidatorLogic in place of ValidatorLogic."""
```

---

## Stage Sequence

**Training scenarios** (`full_run`, `retrain_existing_data`, `tune_hyperparams`):

```
DataLoader               ← read-only: BQ for full_run, GCS otherwise
    ↓
RawSnapshotter           ← gated by config.take_snapshot_raw; no-op otherwise
    ↓
DataSplitter             ← create or load holdout; writes df_train / df_test / df_val
    ↓
EDA (pre-engineering)    ← real data only; individual plot functions, each callable
    ↓
FeatureEngineer          ← fits on df_train (real only); transforms all three splits
    ↓
EDA (post-engineering)   ← real data only; velocity, correlation, scatter plots
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
Validator                ← evaluates on X_val (real only, locked); records results
```

**Validation-only scenarios** (`validate_current`, `retro_validate`):

```
DataLoader
    ↓
DataSplitter            ← loads existing holdout; writes df_train / df_test / df_val
    ↓
FeatureEngineer         ← transforms all three splits (no fit needed beyond train)
    ↓
ModelLoader             ← reads model(s) from GCS; populates run.models
    ↓
Validator               ← evaluates loaded models on X_val; records results
```

**BQ vs GCS by scenario:**

- `full_run` only — `DataLoader` reads from BigQuery
- All other scenarios — `DataLoader` reads from GCS parquet snapshot
- All scenarios — `DataSplitter` reads holdout IDs from GCS (or creates on first run)
- `validate_current` / `retro_validate` — `ModelLoader` present, `ModelTrainer` absent
  from `PipelineStages` (accessing `stages.trainer` raises)
- `full_run` / `retrain_existing_data` / `tune_hyperparams` — `ModelTrainer` present;
  `ModelLoader` also wired in for optional warm-start / baseline-comparison use

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
stages.splitter.run(run)
run.summary()

# EDA — pre-engineering (real data only)
eda.plot_label_rates(run)
eda.plot_engagement_distribution(run)
eda.plot_segment_heatmap(run)

stages.feature_engineer.run(run)

# EDA — post-engineering (real data only)
eda.plot_velocity_distributions(run)
eda.plot_feature_correlations(run)

stages.augmenter.run(run)              # no-op if config.use_synthetic = False
stages.final_snapshotter.run(run)      # no-op if config.take_snapshot_final = False
stages.trainer.run(run)
stages.hyperparam_snapshotter.run(run) # no-op if config.take_snapshot_hyperparams = False
stages.model_snapshotter.run(run)      # no-op if config.take_snapshot_models = False
stages.validator.run(run)

config.commit()
```

---

## Run Scenarios

All training scenarios share the same active stage list — they differ in source
of truth (BQ vs GCS), what gets snapshotted, and tuning behavior. Snapshotters
are wired into every training scenario; their gates (`config.take_snapshot_*`)
decide whether they actually write.

| Scenario | Factory method | BQ load | Stages active | Notes |
|---|---|---|---|---|
| Full run (new data) | `full_run` | ✓ | loader → raw_snap → splitter → engineer → augment → final_snap → trainer → hyperparam_snap → model_snap → validator (+ model_loader available) | Creates holdout on first run ever |
| Retrain, same data | `retrain_existing_data` | ✗ | loader → raw_snap → splitter → engineer → augment → final_snap → trainer → hyperparam_snap → model_snap → validator (+ model_loader available) | Loads parquet from GCS |
| Hyperparameter tuning | `tune_hyperparams` | ✗ | loader → raw_snap → splitter → engineer → augment → final_snap → trainer → hyperparam_snap → model_snap → validator (+ model_loader available) | Search enabled via `config.tune_models` |
| Validate current models | `validate_current` | ✗ | loader → splitter → engineer → model_loader → validator | No trainer; no snapshotters (read-only) |
| Replay saved models | `retro_validate` | ✗ | loader → splitter → engineer → model_loader → validator | `RetroValidatorLogic` injected; loads multiple model versions |

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
    @property def use_synthetic(self) -> bool: ...  # NEW

    # Version strings (set by build())
    raw_version: str
    final_version: str
    model_version: str
    hyperparam_version: str
```

### `pipeline/pipeline_run.py`
```python
@dataclass
class PipelineRun:
    config: VersionConfig
    df_videos:   Optional[pd.DataFrame] = None
    df_baselines: Optional[pd.DataFrame] = None
    df_medians:  Optional[pd.DataFrame] = None
    df_train:    Optional[pd.DataFrame] = None
    df_test:     Optional[pd.DataFrame] = None
    df_val:      Optional[pd.DataFrame] = None
    df_model:    Optional[pd.DataFrame] = None
    X_train:     Optional[pd.DataFrame] = None
    X_test:      Optional[pd.DataFrame] = None
    X_val:       Optional[pd.DataFrame] = None
    y_train:     Optional[pd.Series]    = None
    y_test:      Optional[pd.Series]    = None
    y_val:       Optional[pd.Series]    = None
    models:      dict = field(default_factory=dict)
    results:     dict = field(default_factory=dict)

    def assert_ready_for(self, stage: str) -> None: ...
    def summary(self) -> None: ...
```

### `pipeline/factory.py`
```python
class PipelineStages:
    """Plain class — absent stages are not set as attributes; __getattr__ raises."""

    VALID_STAGE_NAMES = (
        "loader", "raw_snapshotter",
        "splitter",
        "feature_engineer",
        "augmenter", "final_snapshotter",
        "trainer", "hyperparam_snapshotter", "model_snapshotter",
        "model_loader",
        "validator",
    )

    def __init__(self, scenario: str, **stages):
        # Sets only the stages provided; absent ones are not attributes at all,
        # so __getattr__ fires on access and produces a descriptive error.
        ...

    def __getattr__(self, name: str):
        # Raises descriptive error when an absent stage is accessed.
        ...

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

    BQ vs GCS is decided by `config.scenario`:
      - full_run                         → BigQuery (video_snapshots + baselines + medians)
      - retrain_existing_data, tune_*,
        validate_current, retro_validate → GCS parquet at config.raw_version

    Writing back to GCS is the job of `RawSnapshotter`, not this stage.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/data_splitter.py`
```python
class DataSplitter:
    GCS_VALIDATION_IDS_PATH = "splits/validation_ids.json"

    def __init__(self, config: VersionConfig, seed: int = 42): ...
    def run(self, run: PipelineRun) -> PipelineRun:
        # create path: splits, persists to GCS, populates run.df_train/test/val
        # load path: reads IDs from GCS, partitions, warns if val set shrank
        ...
```

### `pipeline/stages/eda.py`
```python
# Module-level constants
FIGSIZE_WIDE = (12, 4)
FIGSIZE_STANDARD = (11, 5)
PALETTE = "Set2"

# Pre-engineering plots (operate on run.df_train, real data only)
def plot_label_rates(run: PipelineRun, figsize=FIGSIZE_WIDE) -> None: ...
def plot_engagement_distribution(run: PipelineRun, figsize=FIGSIZE_STANDARD) -> None: ...
def plot_segment_heatmap(run: PipelineRun, figsize=(6, 4)) -> None: ...

# Post-engineering plots (operate on run.df_model, real data only)
def plot_velocity_distributions(run: PipelineRun, figsize=(15, 4)) -> None: ...
def plot_feature_correlations(run: PipelineRun, figsize=(28, 12)) -> None: ...
def plot_feature_scatter(run: PipelineRun, figsize=(15, 6)) -> None: ...
```

### `pipeline/stages/feature_engineer.py`
```python
class FeatureEngineerLogic:
    def run(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...

class FeatureEngineer:
    def __init__(self, config: VersionConfig, logic: FeatureEngineerLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
    def transform_external(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply already-fitted engineering + scaling to a new DataFrame.
        Used by SyntheticAugmenter to align synth rows with X_train. Errors
        if called before run() has fit the scaler.
        """
        ...
    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: ...
```

### `pipeline/stages/synthetic_augmenter.py`
```python
class SyntheticAugmenter:
    """Generates synthetic rows from run.df_model, engineers + scales them
    via the injected FeatureEngineer, and appends to run.X_train / run.y_train.

    Constructor-injected with FeatureEngineer to reuse its fitted scaler and
    feature_cols — re-implementing those here would risk silent drift.
    """
    def __init__(
        self,
        config: VersionConfig,
        feature_engineer: "FeatureEngineer",
        seed: int = 42,
    ): ...

    def run(self, run: PipelineRun) -> PipelineRun:
        # no-op if config.use_synthetic is False
        # otherwise:
        #   1. generate_synthetic_data(run.df_model)
        #   2. feature_engineer.transform_external(df_synth) → X_synth, y_synth
        #   3. concat onto run.X_train / run.y_train
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
    ) -> dict:  # {model_name: fitted_model}
        ...

class ModelTrainer:
    def __init__(self, config: VersionConfig, logic: ModelTrainerLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/model_loader.py`

```python
class ModelLoader:
    """Loads families of saved model artifacts from GCS by base version string.

    For each base version (e.g. "v3.1"), auto-discovers all `models/<base>_*/`
    directories in GCS and loads each. Populates run.models keyed by per-model
    suffix for the single-version case, or by f"{base_version}/{model_name}"
    when loading multiple base versions (retro_validate).
    """

    def __init__(
        self,
        config: VersionConfig,
        versions: list[str] = None,
    ):
        # If `versions` is None, defaults to [config.model_version].
        # validate_current passes None; retro_validate passes a list of base
        # versions (e.g. ["v3.0", "v3.1", "v3.2"]).
        ...

    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/raw_snapshotter.py`
```python
class RawSnapshotter:
    """Persists run.df_videos / df_baselines / df_medians to GCS at
    config.raw_version. No-op when config.take_snapshot_raw is False.
    Delegates to existing utils.snapshot_data utilities.
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
    config.take_snapshot_models is False. Delegates to utils.snapshot_model.save_model.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/hyperparam_snapshotter.py`
```python
class HyperparamSnapshotter:
    """Persists the hyperparameters used (or discovered via search) at
    config.hyperparam_version. No-op when config.take_snapshot_hyperparams
    is False. Delegates to utils.snapshot_hyperparameters.save_hyperparams.
    """
    def __init__(self, config: VersionConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

### `pipeline/stages/validator.py`
```python
class ValidatorLogic:
    """Evaluates current trained models against the validation set."""
    def run(
        self,
        models: dict,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: VersionConfig,
    ) -> dict: ...  # {model_name: metrics_dict}

class RetroValidatorLogic:
    """Evaluates multiple saved model versions against the validation set.

    Loading is handled upstream by `ModelLoader`; this class consumes the
    already-populated `run.models` dict and produces a per-version metrics
    breakdown suitable for cross-version comparison.
    """
    def run(
        self,
        models: dict,           # populated by ModelLoader with multiple versions
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: VersionConfig,
    ) -> dict: ...

class Validator:
    def __init__(self, config: VersionConfig, logic: ValidatorLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

---

## Implementation Notes

These are constraints for Claude Code, not design rationale.

**File operations:**

- `utils/snapshot_config.py` → `pipeline/version_config.py` via `git mv`. Rewrite the
  contents entirely; preserve the builder pattern and all existing public method names.
  Do not leave a stub or import alias in `utils/`.
- The intermediate name `pipeline/run_config.py` (used during the early refactor passes)
  must also be renamed to `pipeline/version_config.py`, with the class renamed
  `RunConfig` → `VersionConfig`. Update all imports across `pipeline/`, `tests/`, and
  the notebook. The rename is purely cosmetic; no logic changes.
- New working notebook: `project_pipeline.ipynb` at the project root. Do not modify
  or delete `data_analysis.ipynb`.
- `data_collection/` is off-limits. Do not touch any file under that directory.
- `data_processing/` and `utils/` may be reorganized as needed to support the new
  pipeline package.

**PipelineStages absent-stage behavior:**

- `PipelineStages` is a plain class, not a dataclass. Stages absent from a given
  scenario are never set as instance attributes — Python's `__getattr__` then fires
  on access and raises a descriptive error. Do not use a dataclass with `None`
  defaults (lookup would succeed and `__getattr__` would never run) and do not
  override `__getattribute__` (recursion risk).
- Each instance carries a `scenario: str` attribute (the factory method name) so
  the error message can reference it.
- Error message must name the stage and the factory method, e.g.:
  `"ModelTrainer is not part of this pipeline scenario (validate_current). Check your PipelineFactory method."`
- Validate that all kwargs passed to `__init__` are recognized stage names; reject
  typos with a clear error rather than silently storing an unknown attribute.

**DataSplitter GCS path:**
- Bucket: `maduros-dolce-capstone-data`
- Full path: `gs://maduros-dolce-capstone-data/splits/validation_ids.json`
- Use `google-cloud-storage` (already a project dependency). Do not use `gsutil`.

**VersionConfig additions vs. existing SnapshotConfig:**
- Add `use_synthetic: bool = True` as a new flag property.
- All existing version-bumping logic, GCS read/write, and builder method names must
  be preserved exactly — downstream GCS artifacts depend on the version string format.

**Unit tests:**
- Write unit tests for all `Logic` classes (`FeatureEngineerLogic`, `ModelTrainerLogic`,
  `ValidatorLogic`, `RetroValidatorLogic`) using `pytest`.
- Place tests in `tests/pipeline/stages/` mirroring the stage file structure.
- Tests must not require GCS access or a BigQuery connection — use minimal DataFrames
  constructed inline.
- Do not write tests for outer orchestration classes (`FeatureEngineer`, `ModelTrainer`,
  etc.) at this stage.

**Existing `data_processing/` functions:**

- `data_processing/feature_engineering.py` and `data_processing/data_cleanup.py`
  are the source of truth for transformation logic. `FeatureEngineerLogic` should
  delegate to these functions, not reimplement them.
- `data_processing/synthetic_data.py` is the source of truth for synthetic generation.
  `SyntheticAugmenter` delegates to `generate_synthetic_data` from that module
  (`combine_real_and_synthetic` is no longer used — the new pipeline appends synth
  to `X_train` / `y_train` directly via `FeatureEngineer.transform_external`).

**Existing `utils/` snapshot helpers:**

- The Snapshotter family delegates to existing utilities; do not duplicate the
  GCS / disk logic.
- `RawSnapshotter` → `utils.snapshot_data.snapshot_video_data` and `snapshot_baselines`.
- `FinalSnapshotter` → a *new* per-split helper inside `utils.snapshot_data` (the
  existing `save_snapshot` saves a single combined frame; we now save six per-split
  parquet files plus a metadata sidecar). Keep `save_snapshot` available for
  backwards compatibility but do not use it from the new pipeline.
- `ModelSnapshotter` → `utils.snapshot_model.save_model`, called once per fitted
  model in `run.models`.
- `HyperparamSnapshotter` → `utils.snapshot_hyperparameters.save_hyperparams`.

**Snapshotter gating and idempotency:**

- Each snapshotter checks its `config.take_snapshot_*` flag at the top of `run()`
  and returns immediately when False. The notebook always calls every snapshotter;
  the gate lives inside the stage so the notebook stays scenario-agnostic.
- Snapshotters never modify `PipelineRun` state — they only read what previous
  stages populated and write to GCS.
- A snapshotter must fail loudly (raise) if its required inputs are missing — e.g.,
  `ModelSnapshotter` raises if `run.models` is empty when its flag is set.

---

## Design Principles

- **The notebook sequences; it does not implement.** All logic below the stage `run()`
  call lives in `.py` files.
- **Logic classes are independently testable.** `FeatureEngineerLogic().run(df, df, df)`
  requires no config, no GCS, no network.
- **The factory names scenarios.** New run types are new factory methods — not new
  notebook cells or new conditionals.
- **EDA always reflects real data.** Synthetic augmentation is placed after all EDA
  stages so plots are never contaminated by SDV-generated rows.
- **The validation set is immutable.** Created once, persisted by video ID to GCS,
  never retrained against. Enables apples-to-apples comparison across all model versions
  past and future.
- **Synthetic rows never enter test or validation.** `SyntheticAugmenter` writes only
  to `run.X_train` / `run.y_train`. The boundary is enforced structurally, not by a
  filter at evaluation time.
- **EDA is callable, not prescribed.** Plot functions are called individually in the
  notebook; any can be commented out. Shared aesthetics live as module-level constants
  in `eda.py`.
- **BigQuery access is exclusive to `full_run`.** All other scenarios load from GCS
  parquet, keeping quota consumption predictable and non-BQ runs fast.
- **Reads and writes are separate stages.** Loaders never write; snapshotters never
  read external state for transformation. Each loader has a write-side counterpart
  (DataLoader↔RawSnapshotter, ModelLoader↔ModelSnapshotter) so the symmetry is
  visible in the stage list and the notebook reads top-to-bottom without scenario
  branching.
