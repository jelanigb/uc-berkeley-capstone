# Pipeline Architecture Design
**UC Berkeley Capstone — YouTube Engagement Prediction**
*Draft — April 2026*

---

## Overview

The pipeline is restructured around three layers:

- **`RunConfig`** — builder-pattern orchestration config (what to do this run)
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
├── run_config.py               # RunConfig builder (replaces utils/snapshot_config.py)
├── pipeline_run.py             # PipelineRun dataclass — typed state carrier
├── factory.py                  # PipelineFactory + PipelineStages container
│
└── stages/
    ├── data_loader.py          # DataLoader
    ├── data_splitter.py        # DataSplitter (create or load holdout)
    ├── eda.py                  # Module-level plot functions (no class)
    ├── feature_engineer.py     # FeatureEngineerLogic + FeatureEngineer
    ├── synthetic_augmenter.py  # SyntheticAugmenter (train split only)
    ├── model_trainer.py        # ModelTrainerLogic + ModelTrainer
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
    config: RunConfig

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
    def __init__(self, config: RunConfig, logic: FeatureEngineerLogic = None):
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
`SyntheticAugmenter` is also thin: it delegates to the existing `generate_synthetic_data`
and `combine_real_and_synthetic` functions in `data_processing/synthetic_data.py`.
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
`GaussianCopulaSynthesizer` and appends them to `run.X_train` / `run.y_train` only.
Test and validation splits are never touched. Is a no-op when `config.use_synthetic
= False`.

### PipelineFactory — wiring and scenario assembly

The single place where logic classes are instantiated and injected into stages.
`PipelineStages` is a dataclass with `Optional` stage fields. When a stage is absent
for a given scenario, its field is `None`. `PipelineStages` overrides `__getattr__`
to intercept access to `None` fields and raise a descriptive error:

```
ModelTrainer is not part of this pipeline scenario (validate_current).
Check your PipelineFactory method.
```

```python
class PipelineFactory:

    @staticmethod
    def full_run(config: RunConfig) -> PipelineStages:
        """Load fresh data from BQ, split, engineer, augment, train, validate."""

    @staticmethod
    def retrain_existing_data(config: RunConfig) -> PipelineStages:
        """Load from GCS parquet snapshot (no BQ). Re-split, engineer, augment,
        train, validate."""

    @staticmethod
    def tune_hyperparams(config: RunConfig) -> PipelineStages:
        """Retrain with hyperparameter search enabled on existing data snapshot."""

    @staticmethod
    def validate_current(config: RunConfig) -> PipelineStages:
        """Validation stage only. No trainer — accessing stages.trainer raises."""

    @staticmethod
    def retro_validate(
        config: RunConfig,
        model_versions: list[str],
    ) -> PipelineStages:
        """Reload saved model versions from GCS, replay against fixed validation set.
        Injects RetroValidatorLogic in place of ValidatorLogic."""
```

---

## Stage Sequence

```
DataLoader
    ↓
DataSplitter            ← create or load holdout; writes df_train / df_test / df_val
    ↓
EDA (pre-engineering)   ← real data only; individual plot functions, each callable
    ↓
FeatureEngineer         ← fits on df_train (real only); transforms all three splits
    ↓
EDA (post-engineering)  ← real data only; velocity, correlation, scatter plots
    ↓
SyntheticAugmenter      ← appends synthetic rows to X_train / y_train only
    ↓
ModelTrainer            ← trains on X_train (real + synthetic); evaluates on X_test
    ↓
Validator               ← evaluates on X_val (real only, locked); records results
```

**BQ vs GCS by scenario:**

- `full_run` only — `DataLoader` reads from BigQuery
- All other scenarios — `DataLoader` reads from GCS parquet snapshot
- All scenarios — `DataSplitter` reads holdout IDs from GCS (or creates on first run)
- `validate_current` / `retro_validate` — `ModelTrainer` absent from `PipelineStages`

---

## Notebook Sequencing

```python
from pipeline.run_config import RunConfig
from pipeline.pipeline_run import PipelineRun
from pipeline.factory import PipelineFactory
from pipeline.stages import eda

config = RunConfig.load().snapshot_models_new_data().build()
run = PipelineRun(config)
stages = PipelineFactory.full_run(config)

stages.loader.run(run)
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

stages.augmenter.run(run)    # no-op if config.use_synthetic = False
stages.trainer.run(run)
stages.validator.run(run)

config.commit()
```

---

## Run Scenarios

| Scenario | Factory method | BQ load | Stages active | Notes |
|---|---|---|---|---|
| Full run (new data) | `full_run` | ✓ | all | Creates holdout on first run ever |
| Retrain, same data | `retrain_existing_data` | ✗ | all | Loads parquet from GCS |
| Hyperparameter tuning | `tune_hyperparams` | ✗ | splitter → engineer → augment → train → validate | Search enabled in trainer logic |
| Validate current models | `validate_current` | ✗ | splitter → engineer → validate | No trainer stage |
| Replay saved models | `retro_validate` | ✗ | splitter → engineer → validate | `RetroValidatorLogic` injected |

---

## Interfaces

### `pipeline/run_config.py`
```python
class RunConfig:
    # Builder methods (all return self)
    @classmethod
    def load(cls) -> "RunConfig": ...
    def snapshot_raw(self, suffix: str = None) -> "RunConfig": ...
    def snapshot_final(self, suffix: str = None) -> "RunConfig": ...
    def snapshot_schema_change(self) -> "RunConfig": ...
    def snapshot_models(self) -> "RunConfig": ...
    def snapshot_models_new_data(self) -> "RunConfig": ...
    def snapshot_hyperparams(self) -> "RunConfig": ...
    def snapshot_hyperparams_new_grid(self) -> "RunConfig": ...
    def tune(self, strategy: str = "random", n_iter: int = 50,
             cv: int = 5, scoring: str = "roc_auc") -> "RunConfig": ...
    def use_data_version(self, version: str) -> "RunConfig": ...
    def use_hyperparam_version(self, version: str) -> "RunConfig": ...
    def build(self) -> "RunConfig": ...
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
    config: RunConfig
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
@dataclass
class PipelineStages:
    loader:           Optional[DataLoader]          = None
    splitter:         Optional[DataSplitter]        = None
    feature_engineer: Optional[FeatureEngineer]     = None
    augmenter:        Optional[SyntheticAugmenter]  = None
    trainer:          Optional[ModelTrainer]         = None
    validator:        Optional[Validator]            = None

    def __getattr__(self, name: str):
        # Raises descriptive error when a None stage is accessed

class PipelineFactory:
    @staticmethod
    def full_run(config: RunConfig) -> PipelineStages: ...
    @staticmethod
    def retrain_existing_data(config: RunConfig) -> PipelineStages: ...
    @staticmethod
    def tune_hyperparams(config: RunConfig) -> PipelineStages: ...
    @staticmethod
    def validate_current(config: RunConfig) -> PipelineStages: ...
    @staticmethod
    def retro_validate(config: RunConfig, model_versions: list[str]) -> PipelineStages: ...
```

### `pipeline/stages/data_loader.py`
```python
class DataLoader:
    def __init__(self, config: RunConfig): ...
    def run(self, run: PipelineRun) -> PipelineRun:
        # full_run: reads from BigQuery (video_snapshots + baselines + medians)
        # all others: reads from GCS parquet at config.raw_version
        ...
```

### `pipeline/stages/data_splitter.py`
```python
class DataSplitter:
    GCS_VALIDATION_IDS_PATH = "splits/validation_ids.json"

    def __init__(self, config: RunConfig, seed: int = 42): ...
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
    def __init__(self, config: RunConfig, logic: FeatureEngineerLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: ...
```

### `pipeline/stages/synthetic_augmenter.py`
```python
class SyntheticAugmenter:
    def __init__(self, config: RunConfig, seed: int = 42): ...
    def run(self, run: PipelineRun) -> PipelineRun:
        # no-op if config.use_synthetic is False
        # otherwise: generates synthetic rows, appends to run.X_train / run.y_train only
        ...
```

### `pipeline/stages/model_trainer.py`
```python
class ModelTrainerLogic:
    def run(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame,  y_test: pd.Series,
        config: RunConfig,
    ) -> dict:  # {model_name: fitted_model}
        ...

class ModelTrainer:
    def __init__(self, config: RunConfig, logic: ModelTrainerLogic = None): ...
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
        config: RunConfig,
    ) -> dict: ...  # {model_name: metrics_dict}

class RetroValidatorLogic:
    """Loads saved model versions from GCS and evaluates them against validation set."""
    def __init__(self, versions: list[str]): ...
    def run(
        self,
        models: dict,           # ignored — loads from GCS instead
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: RunConfig,
    ) -> dict: ...

class Validator:
    def __init__(self, config: RunConfig, logic: ValidatorLogic = None): ...
    def run(self, run: PipelineRun) -> PipelineRun: ...
```

---

## Implementation Notes

These are constraints for Claude Code, not design rationale.

**File operations:**
- `utils/snapshot_config.py` → `pipeline/run_config.py` via `git mv`. Rewrite the
  contents entirely; preserve the builder pattern and all existing public method names.
  Do not leave a stub or import alias in `utils/`.
- New working notebook: `project_pipeline.ipynb` at the project root. Do not modify
  or delete `data_analysis.ipynb`.
- `data_collection/` is off-limits. Do not touch any file under that directory.
- `data_processing/` and `utils/` may be reorganized as needed to support the new
  pipeline package.

**PipelineStages absent-stage behavior:**
- Implement via `__getattr__` override, not by raising in field accessors.
- Error message must name the stage and the factory method, e.g.:
  `"ModelTrainer is not part of this pipeline scenario (validate_current). Check your PipelineFactory method."`

**DataSplitter GCS path:**
- Bucket: `maduros-dolce-capstone-data`
- Full path: `gs://maduros-dolce-capstone-data/splits/validation_ids.json`
- Use `google-cloud-storage` (already a project dependency). Do not use `gsutil`.

**RunConfig additions vs. existing SnapshotConfig:**
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
  `SyntheticAugmenter` delegates to `generate_synthetic_data` and
  `combine_real_and_synthetic` from that module.

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
