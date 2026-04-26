"""
Validator — evaluate fitted models against the locked validation set.

Architecture
------------
ValidatorLogic does the per-model evaluation: it applies each model's own
saved scaler to X_val_unscaled, predicts, and computes metrics. The outer
Validator handles orchestration and writes results to run.results.

RetroValidatorLogic is injected in place of ValidatorLogic for the
`retro_validate` scenario. The evaluation algorithm is identical — the
distinction is that run.models is keyed by "v3.0/lr_l1" etc. rather than
just "lr_l1", and the notebook displays a cross-version comparison table.

Why X_val_unscaled
------------------
FeatureEngineer fits its StandardScaler on the current training split, which
changes between runs. A loaded historical model was trained with a different
scaler (saved alongside the model artifacts in GCS). Evaluating it through
the current scaler would produce incorrect results. By keeping X_val_unscaled
(post-engineering, pre-StandardScaler) and applying each entry's own scaler,
both freshly-trained models and loaded historical models are evaluated
correctly. When entry["scaler"] is None (legacy RF/XGB artifacts saved before
unified scaling), X_val_unscaled is used directly — those models were trained
on unscaled features.

run.results shape
-----------------
    run.results = {
        "lr_l1": {
            "roc_auc":           float,
            "accuracy":          float,
            "precision_above":   float,
            "recall_above":      float,
            "f1_above":          float,
            "precision_below":   float,
            "recall_below":      float,
            "f1_below":          float,
            "confusion_matrix":  list[list[int]],
            "top_features":      list[dict],
            "val_rows":          int,
        },
        "rf": {...},
        ...
    }

For retro_validate, keys are prefixed: "v3.0/lr_l1", "v3.1/lr_l1", etc.
"""

from dataclasses import asdict

import pandas as pd

from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import VersionConfig
from utils.snapshot_model import ModelResult


class ValidatorLogic:
    """Evaluate models against the validation set — testable with just DataFrames."""

    def run(
        self,
        models: dict,
        X_val_unscaled: pd.DataFrame,
        y_val: pd.Series,
        config: VersionConfig,
    ) -> dict:
        """Evaluate every model in `models` and return {model_name: metrics_dict}."""
        results = {}
        for name, entry in models.items():
            X = self._prepare_X(entry, X_val_unscaled)
            results[name] = self._evaluate(entry["model"], X, y_val)
        self._print_summary(results)
        return results

    def _prepare_X(self, entry: dict, X_val_unscaled: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns and apply the entry's scaler (or none)."""
        feature_cols = entry["feature_cols"]
        X = X_val_unscaled[feature_cols].copy()
        scaler = entry.get("scaler")
        if scaler is not None:
            X = pd.DataFrame(
                scaler.transform(X),
                columns=feature_cols,
                index=X.index,
            )
        return X

    def _evaluate(self, model, X: pd.DataFrame, y_val: pd.Series) -> dict:
        result = ModelResult.from_sklearn(model, X, y_val, X.columns.tolist())
        return {**asdict(result), "val_rows": len(y_val)}

    @staticmethod
    def _print_summary(results: dict) -> None:
        print("\n=== Validator — validation-set results ===")
        for name, r in results.items():
            print(
                f"  {name:<14}  AUC={r['roc_auc']:.4f}  "
                f"acc={r['accuracy']:.4f}  F1↑={r['f1_above']:.4f}"
            )


class RetroValidatorLogic(ValidatorLogic):
    """Evaluate multiple saved model versions from run.models.

    Functionally identical to ValidatorLogic. The multi-version distinction
    lives in how ModelLoader keys run.models ("v3.0/lr_l1" etc.) and how the
    notebook renders the results — not in the evaluation algorithm itself.
    Injected by PipelineFactory.retro_validate in place of ValidatorLogic.
    """


class Validator:
    """Stage — evaluate run.models against the locked validation set."""

    def __init__(self, config: VersionConfig, logic: ValidatorLogic = None):
        self.config = config
        self.logic = logic or ValidatorLogic()

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.VALIDATE)

        if not run.models:
            raise RuntimeError(
                "Validator: run.models is empty. "
                "Did ModelTrainer.run() or ModelLoader.run() complete?"
            )

        run.results = self.logic.run(
            run.models,
            run.X_val_unscaled,
            run.y_val,
            self.config,
        )
        return run
