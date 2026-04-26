"""
SyntheticAugmenter — append synthetic rows to the train split only.

Generates synthetic rows from the real, engineered train frame (`run.df_model`)
via `data_processing.synthetic_data.generate_synthetic_data`, then aligns the
result with `run.X_train` by routing it through the injected `FeatureEngineer`'s
`transform_external` method. The aligned synth rows are concatenated onto
`run.X_train` / `run.y_train` only — `X_test` and `X_val` are never touched.

Coupling to FeatureEngineer
---------------------------
Synth rows must end up shaped *exactly* like real X_train: same engineered
columns, same categorical encoding, scaled by the same fitted StandardScaler.
Re-implementing any part of that here would risk silent drift, so the augmenter
is constructor-injected with the live `FeatureEngineer` instance and reuses its
public `transform_external(df)` method. The factory wires this dependency.

Gating
------
The stage is a no-op when `config.use_synthetic` is False — the notebook
always calls it; the gate lives inside the stage so the notebook stays
scenario-agnostic.

Row-count formula
-----------------
Synth row count is derived from `target_real_pct` so the resulting train
mixture has that proportion of real data. `target_real_pct=0.8` → real:synth
= 4:1 (e.g. 10k real → 2.5k synth → 12.5k total, 80% real). Formula matches
the legacy `get_synth_rows_proportion` helper from the original notebook.
"""

import math

import pandas as pd

from data_processing.synthetic_data import generate_synthetic_data
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.run_config import VersionConfig
from pipeline.stages.feature_engineer import FeatureEngineer


class SyntheticAugmenter:
    """Stage 6 — generate synth rows, align with X_train, append to train only."""

    DEFAULT_TARGET_REAL_PCT = 0.8

    def __init__(
        self,
        config: VersionConfig,
        feature_engineer: FeatureEngineer,
        seed: int = 42,
        target_real_pct: float = DEFAULT_TARGET_REAL_PCT,
    ):
        if not (0 < target_real_pct < 1):
            raise ValueError(
                f"target_real_pct must be in (0, 1), got {target_real_pct!r}."
            )
        self.config = config
        self.feature_engineer = feature_engineer
        self.seed = seed
        self.target_real_pct = target_real_pct

    def run(self, run: PipelineRun) -> PipelineRun:
        if not self.config.use_synthetic:
            print("[SyntheticAugmenter] use_synthetic=False — skipping.")
            return run

        run.assert_ready_for(Stage.AUGMENT)
        if run.df_model is None:
            raise RuntimeError(
                "SyntheticAugmenter requires run.df_model. "
                "Did FeatureEngineer.run() complete?"
            )

        num_synth = self._compute_synth_row_count(len(run.df_model))
        if num_synth <= 0:
            print(
                f"[SyntheticAugmenter] Computed num_synth={num_synth} "
                f"(target_real_pct={self.target_real_pct}); skipping."
            )
            return run

        df_synth = generate_synthetic_data(
            run.df_model,
            num_rows=num_synth,
            seed=self.seed,
        )

        X_synth, y_synth = self.feature_engineer.transform_external(
            df_synth, label="synth"
        )

        run.X_train = pd.concat([run.X_train, X_synth])
        run.y_train = pd.concat([run.y_train, y_synth])

        print(
            f"[SyntheticAugmenter] Appended {len(X_synth)} synth rows to X_train. "
            f"X_train: {len(run.X_train)} rows total "
            f"({len(run.X_train) - len(X_synth)} real + {len(X_synth)} synth)."
        )
        return run

    def _compute_synth_row_count(self, num_real_rows: int) -> int:
        """Mirrors `get_synth_rows_proportion` from the legacy notebook:

            synth = floor(real / target_real_pct) - real
        """
        return math.floor(num_real_rows / self.target_real_pct) - num_real_rows
