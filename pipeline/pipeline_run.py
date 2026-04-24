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

from pipeline.run_config import RunConfig


class Stage(Enum):
    """Stage identifiers for `PipelineRun.assert_ready_for`.

    Named singular (not `Stages`) to avoid shadowing the `stages` binding
    returned by `PipelineFactory.<scenario>(config)` in the notebook.
    """
    LOAD = "load"
    SPLIT = "split"
    EDA_PRE = "eda_pre"
    ENGINEER = "engineer"
    EDA_POST = "eda_post"
    AUGMENT = "augment"
    TRAIN = "train"
    VALIDATE = "validate"


# Fields each stage expects to already be populated before it runs.
REQUIRED_FIELDS_ = {
    Stage.LOAD: [],
    Stage.SPLIT: ["df_videos"],
    Stage.EDA_PRE: ["df_train"],
    Stage.ENGINEER: ["df_train", "df_test", "df_val"],
    Stage.EDA_POST: ["df_model"],
    Stage.AUGMENT: ["X_train", "y_train"],
    Stage.TRAIN: ["X_train", "y_train", "X_test", "y_test"],
    Stage.VALIDATE: ["X_val", "y_val"],
}

SUMMARY_FIELDS_ = [
    "df_videos", "df_baselines", "df_medians",
    "df_train", "df_test", "df_val",
    "df_model",
    "X_train", "X_test", "X_val",
    "y_train", "y_test", "y_val",
    "models", "results",
]


@dataclass
class PipelineRun:
    config: RunConfig

    # Raw data (populated by DataLoader)
    df_videos: Optional[pd.DataFrame] = None
    df_baselines: Optional[pd.DataFrame] = None
    df_medians: Optional[pd.DataFrame] = None

    # Splits (populated by DataSplitter — real data only)
    df_train: Optional[pd.DataFrame] = None   # augmented with synthetic downstream
    df_test: Optional[pd.DataFrame] = None    # real data only, always
    df_val: Optional[pd.DataFrame] = None     # real data only, locked forever

    # Engineered features (populated by FeatureEngineer)
    df_model: Optional[pd.DataFrame] = None   # engineered train set, real only (EDA)
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    X_val: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    y_val: Optional[pd.Series] = None

    # Model artifacts (populated by ModelTrainer)
    models: dict = field(default_factory=dict)

    # Evaluation results (populated by Validator)
    results: dict = field(default_factory=dict)

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
