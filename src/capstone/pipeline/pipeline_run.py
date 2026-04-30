"""
PipelineRun — typed state carrier for a single end-to-end pipeline run.

No transformation logic lives here. Stages read from and write back to a
single `PipelineRun` instance; the notebook holds that one instance and
sequences the stage calls. `assert_ready_for(...)` lets each stage fail
fast — with a clear message — if a prior stage was skipped.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from pipeline.version_config import VersionConfig


class Stage(Enum):
    """Stage identifiers for `PipelineRun.assert_ready_for`.

    Named singular (not `Stages`) to avoid shadowing the `stages` binding
    returned by `PipelineFactory.<scenario>(config)` in the notebook.
    """
    LOAD = "load"
    PREPROCESS = "preprocess"
    ENGINEER = "engineer"
    SPLIT = "split"
    SCALE = "scale"
    EDA_PRE = "eda_pre"
    EDA_POST = "eda_post"
    AUGMENT = "augment"
    TRAIN = "train"
    VALIDATE = "validate"


# Fields each stage expects to already be populated before it runs.
REQUIRED_FIELDS_ = {
    Stage.LOAD: [],
    Stage.PREPROCESS: ["df_videos", "df_medians"],
    Stage.ENGINEER: ["df_clean"],
    Stage.SPLIT: ["df_engineered"],
    Stage.SCALE: ["X_train", "X_test", "X_val"],
    Stage.EDA_PRE: ["df_train"],
    Stage.EDA_POST: ["df_train"],
    Stage.AUGMENT: ["X_train", "y_train", "df_train"],
    Stage.TRAIN: ["X_train", "y_train", "X_test", "y_test"],
    Stage.VALIDATE: ["X_val", "X_val_unscaled", "y_val"],
}

SUMMARY_FIELDS_ = [
    "df_videos", "df_baselines", "df_medians",
    "df_clean", "df_engineered",
    "df_train", "df_test", "df_val",
    "X_train", "X_test", "X_val", "X_val_unscaled",
    "y_train", "y_test", "y_val",
    "models", "results",
]


@dataclass
class PipelineRun:
    config: VersionConfig

    # Raw data (populated by DataLoader)
    df_videos: Optional[pd.DataFrame] = None
    df_baselines: Optional[pd.DataFrame] = None
    df_medians: Optional[pd.DataFrame] = None

    # Preprocessed data (populated by DataPreprocessor)
    # Wide format: 1 row per complete-triplet video, baseline columns joined.
    df_clean: Optional[pd.DataFrame] = None

    # Engineered data (populated by FeatureEngineer)
    # All derived features present; target column above_baseline computed.
    df_engineered: Optional[pd.DataFrame] = None

    # Splits (populated by DataSplitter — real data only, post-engineering)
    df_train: Optional[pd.DataFrame] = None   # augmented with synthetic downstream
    df_test: Optional[pd.DataFrame] = None    # real data only, always
    df_val: Optional[pd.DataFrame] = None     # real data only, locked forever

    # Feature matrices (X unscaled, produced by DataSplitter; scaled in-place by Scaler)
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    X_val: Optional[pd.DataFrame] = None
    # Unscaled val features — post-engineering, pre-StandardScaler.
    # Validator uses this so each loaded model can apply its own saved scaler,
    # which differs from the current Scaler's fit when loading historical models.
    X_val_unscaled: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    y_val: Optional[pd.Series] = None

    # Synthetic row count (set by SyntheticAugmenter; read by ModelTrainer)
    num_synth_rows: int = 0

    # Model artifacts (populated by ModelTrainer)
    models: dict = field(default_factory=dict)

    # Evaluation results (populated by Validator)
    results: dict = field(default_factory=dict)

    # EDA state (set via eda.set_df / eda.set_fig_size / eda.set_palette)
    active_eda_df: Optional[pd.DataFrame] = None
    eda_config: dict = field(default_factory=lambda: {
        "fig_size": (10, 6),
        "palette": "viridis",
    })

    def assert_ready_for(self, stage: Stage) -> None:
        """Fail fast if the fields required by `stage` are not yet populated."""
        if not isinstance(stage, Stage):
            raise TypeError(
                f"Exepcted a Stage enum value, got "
                f"{type(stage).__name__}={stage!r}. "
                f"Import with `from pipeline.pipeline_run import Stage`."
            )
        missing = [f for f in REQUIRED_FIELDS_[stage] if getattr(self, f) is None]
        if missing:
            raise RuntimeError(
                f"PipelineRun not ready for stage '{stage.value}': "
                f"missing field(s) {missing}. "
                f"Did you run the prior stage?"
            )

    def summary(self) -> None:
        """Print a plain-text populated/None readout of every state field."""
        print(f"PipelineRun(config={self.config!r})")
        for name in SUMMARY_FIELDS_:
            val = getattr(self, name)
            print(f"  {name:13s}  {describe_field_(val)}")

    def __repr__(self) -> str:
        populated = [
            n for n in SUMMARY_FIELDS_
            if (v := getattr(self, n)) is not None
            and not (isinstance(v, (dict, list)) and not v)
        ]
        return (
            f"PipelineRun(model_version={self.config.model_version!r}, "
            f"num_synth_rows={self.num_synth_rows}, "
            f"populated={populated})"
        )


def describe_field_(val) -> str:
    if val is None:
        return "None"
    if isinstance(val, pd.DataFrame):
        return f"populated  DataFrame shape={val.shape}"
    if isinstance(val, pd.Series):
        return f"populated  Series length={len(val)}"
    if isinstance(val, dict):
        if not val:
            return "empty dict"
        return f"populated  keys={list(val.keys())}"
    return f"populated  {type(val).__name__}"
