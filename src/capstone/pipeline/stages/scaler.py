"""
Scaler — fit StandardScaler on X_train, transform X_train / X_test / X_val.

Sits between DataSplitter and SyntheticAugmenter. DataSplitter produces
unscaled X matrices; Scaler normalises them in place so every downstream
consumer (Trainer, Validator) sees unit-variance features.

X_val_unscaled is captured before scaling so Validator can apply each
loaded historical model's own saved scaler rather than the current one
(historical models were trained with a different scaler fit).

Exposes scaler_ and feature_cols_ for downstream injection into
SyntheticAugmenter and ModelTrainer, which need to transform new rows and
stamp the scaler onto saved model artifacts respectively.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import VersionConfig


class Scaler:
    """Stage 5 — fit StandardScaler on train, transform all three splits."""

    def __init__(self, config: VersionConfig):
        self.config = config
        self.scaler_: StandardScaler = None
        self.feature_cols_: list = []

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.SCALE)

        self.feature_cols_ = run.X_train.columns.tolist()
        self.scaler_ = StandardScaler()

        run.X_train = self._transform(run.X_train, fit=True)
        run.X_test = self._transform(run.X_test, fit=False)

        # Capture pre-scaling val so Validator can re-apply each model's own scaler.
        run.X_val_unscaled = run.X_val.copy()
        run.X_val = self._transform(run.X_val, fit=False)

        return run

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to an external DataFrame (e.g. synth rows)."""
        if self.scaler_ is None:
            raise RuntimeError(
                "Scaler.transform called before run() — scaler is not fitted yet."
            )
        return self._transform(X, fit=False)

    def _transform(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        arr = self.scaler_.fit_transform(X) if fit else self.scaler_.transform(X)
        return pd.DataFrame(arr, columns=self.feature_cols_, index=X.index)
