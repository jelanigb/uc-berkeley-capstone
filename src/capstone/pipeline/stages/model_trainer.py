"""
ModelTrainer — train LR (L1), RF, XGB, and a soft-voting ensemble on X_train;
evaluate each on X_test; populate run.models with artifacts for snapshotting.

Architecture
------------
ModelTrainerLogic holds the core training and evaluation logic and is fully
testable with just DataFrames and a VersionConfig. ModelTrainer (outer) handles
orchestration and stamps each entry with the scaler and feature_cols from the
injected Scaler stage.

Hyperparameter resolution order
--------------------------------
1. If config.tune_models: run search on LR, RF, XGB; update params in-memory.
2. Else: load stored params from GCS via load_hyperparams(config.hyperparam_version).
3. If GCS snapshot not found: fall back to module defaults from version_config.

Scaling
-------
The Scaler stage fits a StandardScaler on X_train and transforms all splits.
ModelTrainer is constructor-injected with the live Scaler instance so its
fitted scaler_ and feature_cols_ can be stamped onto each run.models entry
for correct inference at load time.

run.models entry shape
----------------------
After ModelTrainer.run() each entry has:
    "model"         — fitted sklearn-compatible model
    "scaler"        — StandardScaler fitted on X_train (from FeatureEngineer)
    "feature_cols"  — list[str] of feature column names (from FeatureEngineer)
    "training_data" — utils.snapshot_model.TrainingData instance
    "result"        — utils.snapshot_model.ModelResult instance
    "model_config"  — utils.snapshot_model.ModelConfig instance

This shape is consumed by ModelSnapshotter and HyperparamSnapshotter.
Validator also accepts it; see validator.py for how it unpacks either shape.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import (
    VersionConfig,
    DEFAULT_LR_PARAMS_,
    DEFAULT_RF_PARAMS_,
    DEFAULT_XGB_PARAMS_,
)
from pipeline.stages.scaler import Scaler
from utils.snapshot_hyperparameters import load_hyperparams
from utils.snapshot_model import ModelConfig, ModelResult, TrainingData
from utils.tune_hyperparameters import get_default_param_grid, tune_model


RANDOM_SEED_ = 42


class ModelTrainerLogic:
    """Core training logic — testable with just DataFrames, no GCS calls."""

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: VersionConfig,
        num_synth_rows: int = 0,
    ) -> dict:
        """Train all four models and return {model_name: partial_entry_dict}.

        Each entry contains model, training_data, result, model_config.
        The outer ModelTrainer adds scaler and feature_cols from FeatureEngineer.
        """
        feature_cols = X_train.columns.tolist()
        params = self._load_params(config)

        if config.tune_models:
            params = self._tune(X_train, y_train, params, config)

        lr = LogisticRegression(random_state=RANDOM_SEED_, **params["LogisticRegression"])
        rf = RandomForestClassifier(random_state=RANDOM_SEED_, n_jobs=-1, **params["RandomForestClassifier"])
        xgb = XGBClassifier(
            random_state=RANDOM_SEED_, n_jobs=-1, eval_metric="logloss",
            **params["XGBClassifier"],
        )

        print("Training LogisticRegression (L1)...")
        lr.fit(X_train, y_train)

        print("Training RandomForestClassifier...")
        rf.fit(X_train, y_train)

        print("Training XGBClassifier...")
        xgb.fit(X_train, y_train)

        # Ensemble uses fresh RF/XGB instances with the same best params so
        # VotingClassifier manages its own fitted copies internally.
        print("Training VotingClassifier ensemble (RF + XGB, weights=[1, 2])...")
        ensemble = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    random_state=RANDOM_SEED_, n_jobs=-1,
                    **params["RandomForestClassifier"],
                )),
                ("xgb", XGBClassifier(
                    random_state=RANDOM_SEED_, n_jobs=-1, eval_metric="logloss",
                    **params["XGBClassifier"],
                )),
            ],
            voting="soft",
            weights=[1, 2],
        )
        ensemble.fit(X_train, y_train)

        total_train = len(X_train)
        training_data = TrainingData(
            real_train_rows=total_train - num_synth_rows,
            synthetic_train_rows=num_synth_rows,
            total_train_rows=total_train,
            test_rows=len(X_test),
            synthetic_pct=round(num_synth_rows / total_train * 100, 1) if total_train > 0 else 0,
            target_balance_train=y_train.value_counts().to_dict(),
            target_balance_test=y_test.value_counts().to_dict(),
        )

        entries = {
            "lr_l1": self._entry(lr, X_test, y_test, feature_cols, training_data),
            "rf": self._entry(rf, X_test, y_test, feature_cols, training_data),
            "xgb": self._entry(xgb, X_test, y_test, feature_cols, training_data),
            "ensemble": self._entry(ensemble, X_test, y_test, feature_cols, training_data),
        }
        self._print_summary(entries)
        return entries

    def _entry(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: list,
        training_data: TrainingData,
    ) -> dict:
        return {
            "model": model,
            "training_data": training_data,
            "result": ModelResult.from_sklearn(model, X_test, y_test, feature_cols),
            "model_config": ModelConfig.from_model(model),
        }

    def _load_params(self, config: VersionConfig) -> dict:
        """Load params from GCS hyperparam snapshot; fall back to VersionConfig defaults."""
        try:
            stored = load_hyperparams(config.hyperparam_version)
            print(f"Loaded hyperparams from snapshot '{config.hyperparam_version}'.")
            return stored["params"]
        except FileNotFoundError:
            print(
                f"No hyperparams snapshot for '{config.hyperparam_version}' — "
                "using VersionConfig defaults."
            )
            return {
                "LogisticRegression": dict(DEFAULT_LR_PARAMS_),
                "RandomForestClassifier": dict(DEFAULT_RF_PARAMS_),
                "XGBClassifier": dict(DEFAULT_XGB_PARAMS_),
            }

    def _tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        current_params: dict,
        config: VersionConfig,
    ) -> dict:
        """Run hyperparameter search for LR, RF, XGB; return updated params."""
        tuned = dict(current_params)
        candidates = [
            ("LogisticRegression", LogisticRegression(random_state=RANDOM_SEED_)),
            ("RandomForestClassifier", RandomForestClassifier(random_state=RANDOM_SEED_, n_jobs=-1)),
            ("XGBClassifier", XGBClassifier(
                random_state=RANDOM_SEED_, n_jobs=-1, eval_metric="logloss"
            )),
        ]
        for model_cls, base_model in candidates:
            param_grid = (
                config.new_grids.get(model_cls)
                or get_default_param_grid(base_model)
            )
            result = tune_model(
                base_model,
                X_train,
                y_train,
                param_grid=param_grid,
                search_strategy=config.search_strategy,
                n_iter=config.search_n_iter,
                cv=config.search_cv,
                scoring=config.search_scoring,
                random_state=RANDOM_SEED_,
            )
            tuned[model_cls] = result["best_params"]

        print("\n=== Tuning complete — best params ===")
        for cls, p in tuned.items():
            print(f"  {cls}: {p}")
        return tuned

    @staticmethod
    def _print_summary(entries: dict) -> None:
        print("\n=== ModelTrainer — test-set results ===")
        for name, entry in entries.items():
            r = entry["result"]
            print(
                f"  {name:<12}  AUC={r.roc_auc:.4f}  "
                f"acc={r.accuracy:.4f}  F1↑={r.f1_above:.4f}"
            )


class ModelTrainer:
    """Stage — train all models and populate run.models.

    Constructor-injected with the live Scaler instance so its fitted
    scaler_ and feature_cols_ can be stamped onto each run.models entry.
    The factory is responsible for wiring this dependency.
    """

    def __init__(
        self,
        config: VersionConfig,
        scaler: Scaler,
        logic: ModelTrainerLogic = None,
    ):
        self.config = config
        self.scaler = scaler
        self.logic = logic or ModelTrainerLogic()

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.TRAIN)

        entries = self.logic.run(
            run.X_train, run.y_train,
            run.X_test, run.y_test,
            self.config,
            num_synth_rows=run.num_synth_rows,
        )

        # Stamp scaler + feature_cols from Scaler onto every entry so
        # ModelSnapshotter can save and inference can reproduce the transform.
        for entry in entries.values():
            entry["scaler"] = self.scaler.scaler_
            entry["feature_cols"] = self.scaler.feature_cols_

        run.models = entries
        return run
