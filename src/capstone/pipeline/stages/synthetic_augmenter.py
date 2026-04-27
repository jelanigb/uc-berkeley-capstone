"""
SyntheticAugmenter — append synthetic rows to the train split only.

Generates synthetic rows from run.df_train (the real, post-engineering,
unscaled train set) via generate_synthetic_data, engineers them through the
same FeatureEngineerLogic chain, aligns them with X_train's column space,
scales them using the injected Scaler's fitted StandardScaler, then
concatenates onto run.X_train / run.y_train. X_test and X_val are never
touched.

Coupling
--------
Synth rows must match X_train exactly: same engineered columns, same
categorical encoding, scaled by the same fitted StandardScaler. This stage
is constructor-injected with:
  - scaler: the live Scaler instance (for scaler_.transform and feature_cols_)
  - logic:  a FeatureEngineerLogic instance (for the engineering chain)

The factory wires these dependencies.

Gating
------
No-op when config.use_synthetic is False.

Row-count formula
-----------------
synth = floor(real / target_real_pct) - real
"""

import math

import pandas as pd

from data_processing.synthetic_data import generate_synthetic_data
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import VersionConfig
from pipeline.stages.feature_engineer import FeatureEngineerLogic, TARGET_COL_


class SyntheticAugmenter:
    """Stage 6 — generate synth rows, align with X_train, append to train only."""

    def __init__(
        self,
        config: VersionConfig,
        scaler,
        logic: FeatureEngineerLogic = None,
        seed: int = 42,
    ):
        self.config = config
        self.scaler = scaler
        self.logic = logic or FeatureEngineerLogic()
        self.seed = seed

    def run(self, run: PipelineRun) -> PipelineRun:
        if not self.config.use_synthetic:
            print("[SyntheticAugmenter] use_synthetic=False — skipping.")
            return run

        run.assert_ready_for(Stage.AUGMENT)

        num_synth = self._compute_synth_row_count(len(run.df_train))
        if num_synth <= 0:
            print(
                f"[SyntheticAugmenter] Computed num_synth={num_synth} "
                f"(target_real_pct={self.config.target_real_pct}); skipping."
            )
            return run

        df_synth = generate_synthetic_data(
            run.df_train,
            num_rows=num_synth,
            seed=self.seed,
        )

        # Engineer synth rows through the same chain that ran on real data.
        df_synth_eng = self.logic.engineer(df_synth, label="synth")

        # Align to feature columns and scale using the already-fitted scaler.
        feature_cols = self.scaler.feature_cols_
        X_synth = df_synth_eng[feature_cols].copy()
        X_synth = self.scaler.transform(X_synth)
        y_synth = df_synth_eng[TARGET_COL_].copy()

        run.X_train = pd.concat([run.X_train, X_synth])
        run.y_train = pd.concat([run.y_train, y_synth])
        run.num_synth_rows = len(X_synth)

        print(
            f"[SyntheticAugmenter] Appended {len(X_synth)} synth rows to X_train. "
            f"X_train: {len(run.X_train)} rows total "
            f"({len(run.X_train) - len(X_synth)} real + {len(X_synth)} synth)."
        )
        return run

    def _compute_synth_row_count(self, num_real_rows: int) -> int:
        """synth = floor(real / target_real_pct) - real"""
        return math.floor(num_real_rows / self.config.target_real_pct) - num_real_rows
