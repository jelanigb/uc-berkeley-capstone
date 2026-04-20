"""
Model snapshot utility for versioning trained models.
Uses dataclasses for structured, extensible metadata capture.

Usage:
    from utils.snapshot_model import (
        TrainingData, ModelResult, ModelConfig,
        save_model, load_model, list_models,
    )

    training_data = TrainingData.from_splits(X_train, y_train, X_test, y_test, df_synth)
    result = ModelResult.from_sklearn(model, X_test, y_test, feature_cols)
    config = ModelConfig.for_logistic_regression(model)

    save_model(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        version_tag="v1.1_lr_l1",
        data_snapshot_tag="v1.0_model_mixed66",
        training_data=training_data,
        result=result,
        config=config,
        notes="Clean LR baseline, no 7d leakage",
    )
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from google.cloud import storage
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ID = "maduros-dolce"
BUCKET_NAME = "maduros-dolce-capstone-data"


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class TrainingData:
    """Captures the composition of training and test data."""
    real_train_rows: int
    synthetic_train_rows: int
    total_train_rows: int
    test_rows: int
    synthetic_pct: float
    target_balance_train: dict
    target_balance_test: dict

    @classmethod
    def from_splits(
        cls,
        X_train, y_train,
        X_test, y_test,
        X_synth=None,
    ):
        """Build from train/test split arrays."""
        synth_count = len(X_synth) if X_synth is not None else 0
        real_count = len(X_train) - synth_count
        total = len(X_train)

        return cls(
            real_train_rows=real_count,
            synthetic_train_rows=synth_count,
            total_train_rows=total,
            test_rows=len(X_test),
            synthetic_pct=round(synth_count / total * 100, 1) if total > 0 else 0,
            target_balance_train=y_train.value_counts().to_dict(),
            target_balance_test=y_test.value_counts().to_dict(),
        )


@dataclass
class ModelResult:
    """Captures evaluation metrics from a trained model."""
    roc_auc: float
    accuracy: float
    precision_above: float
    recall_above: float
    f1_above: float
    precision_below: float
    recall_below: float
    f1_below: float
    confusion_matrix: list
    top_features: list

    @classmethod
    def from_sklearn(
        cls,
        model,
        X_test,
        y_test,
        feature_cols: list,
        top_n: int = 20,
    ):
        """
        Build from any sklearn model with predict and predict_proba.
        Automatically detects feature importance type (coefficients vs impurity).
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(
            y_test, y_pred,
            target_names=['Below Baseline', 'Above Baseline'],
            output_dict=True,
        )

        # Extract feature importances (works for LR, RF, XGBoost, etc.)
        top_features = cls._extract_feature_importances(
            model, feature_cols, top_n
        )

        return cls(
            roc_auc=round(roc_auc_score(y_test, y_pred_proba), 4),
            accuracy=round(float((y_pred == y_test).mean()), 4),
            precision_above=round(report['Above Baseline']['precision'], 4),
            recall_above=round(report['Above Baseline']['recall'], 4),
            f1_above=round(report['Above Baseline']['f1-score'], 4),
            precision_below=round(report['Below Baseline']['precision'], 4),
            recall_below=round(report['Below Baseline']['recall'], 4),
            f1_below=round(report['Below Baseline']['f1-score'], 4),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            top_features=top_features,
        )

    @staticmethod
    def _extract_feature_importances(model, feature_cols, top_n):
        """Extract top features from any sklearn-compatible model."""
        if hasattr(model, 'coef_'):
            # Linear models (LogisticRegression, SVM, etc.)
            importances = model.coef_[0]
            importance_type = 'coefficient'
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models (RF, XGBoost, LightGBM, etc.)
            importances = model.feature_importances_
            importance_type = 'importance'
        else:
            return []

        sorted_idx = np.argsort(np.abs(importances))[::-1][:top_n]
        return [
            {
                'feature': feature_cols[i],
                importance_type: round(float(importances[i]), 6),
            }
            for i in sorted_idx
        ]


@dataclass
class ModelConfig:
    """Captures model type and hyperparameters."""
    model_type: str
    hyperparameters: dict

    @classmethod
    def for_logistic_regression(cls, model):
        """Extract config from a fitted LogisticRegression."""
        return cls(
            model_type="LogisticRegression",
            hyperparameters={
                "penalty": model.penalty,
                "solver": model.solver,
                "C": model.C,
                "max_iter": model.max_iter,
            },
        )

    @classmethod
    def for_random_forest(cls, model):
        """Extract config from a fitted RandomForestClassifier."""
        return cls(
            model_type="RandomForest",
            hyperparameters={
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_leaf": model.min_samples_leaf,
                "min_samples_split": model.min_samples_split,
                "max_features": model.max_features,
            },
        )

    @classmethod
    def for_xgboost(cls, model):
        """Extract config from a fitted XGBClassifier."""
        return cls(
            model_type="XGBoost",
            hyperparameters={
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
                "subsample": model.subsample,
                "colsample_bytree": model.colsample_bytree,
                "min_child_weight": model.min_child_weight,
            },
        )

    @classmethod
    def for_voting_ensemble(cls, model):
        """Extract config from a fitted VotingClassifier.

        Stores only estimator names, voting mode, and weights — not the nested
        estimator objects themselves, which aren't JSON-serializable.
        """
        return cls(
            model_type="VotingClassifier",
            hyperparameters={
                "voting": model.voting,
                "estimators": [name for name, _ in model.estimators],
                "weights": list(model.weights) if model.weights is not None else None,
            },
        )

    @classmethod
    def from_model(cls, model):
        """Auto-detect model type and extract config."""
        class_name = type(model).__name__

        if class_name == 'LogisticRegression':
            return cls.for_logistic_regression(model)
        elif class_name == 'RandomForestClassifier':
            return cls.for_random_forest(model)
        elif class_name == 'XGBClassifier':
            return cls.for_xgboost(model)
        elif class_name == 'VotingClassifier':
            return cls.for_voting_ensemble(model)
        else:
            # Generic fallback: capture what we can from get_params
            params = model.get_params() if hasattr(model, 'get_params') else {}
            return cls(model_type=class_name, hyperparameters=params)


# =========================================================================
# Save / Load / List
# =========================================================================

def save_model(
    model,
    scaler,
    feature_cols: list,
    version_tag: str,
    data_snapshot_tag: str,
    training_data: TrainingData,
    result: ModelResult,
    config: ModelConfig,
    notes: str = "",
    overwrite: bool = False,
):
    """
    Save model artifacts locally and upload to GCS.

    Saves: model.pkl, scaler.pkl, feature_cols.json, metadata.json

    overwrite=True skips the existence check. Use when re-running after an
    interrupted snapshot where some models landed but others did not.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    existing = list(bucket.list_blobs(prefix=f"models/{version_tag}/"))
    if existing and not overwrite:
        raise ValueError(
            f"Model snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag, pass overwrite=True, or delete the existing snapshot first."
        )

    model_dir = f"models/{version_tag}"
    os.makedirs(model_dir, exist_ok=True)

    # Save artifacts
    joblib.dump(model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    with open(f"{model_dir}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Build metadata from dataclasses
    metadata = {
        "version": version_tag,
        "training_date": datetime.utcnow().isoformat(),
        "data_snapshot": data_snapshot_tag,
        "config": asdict(config),
        "training_data": asdict(training_data),
        "result": asdict(result),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "notes": notes,
    }

    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model artifacts saved to {model_dir}/")

    # Upload to GCS

    for filename in ['model.pkl', 'scaler.pkl', 'feature_cols.json', 'metadata.json']:
        local_path = f"{model_dir}/{filename}"
        blob = bucket.blob(f"models/{version_tag}/{filename}")
        blob.upload_from_filename(local_path)
        print(f"  Uploaded gs://{BUCKET_NAME}/models/{version_tag}/{filename}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Model {version_tag} ({config.model_type})")
    print(f"  Data: {data_snapshot_tag} "
          f"({training_data.real_train_rows} real + "
          f"{training_data.synthetic_train_rows} synthetic)")
    print(f"  ROC-AUC: {result.roc_auc}")
    print(f"  Accuracy: {result.accuracy}")
    print(f"  F1 (above): {result.f1_above}")
    print(f"{'=' * 60}")

    return metadata


def load_model(version_tag: str):
    """
    Load model artifacts from GCS.
    Returns (model, scaler, feature_cols, metadata).
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    model_dir = f"/tmp/models/{version_tag}"
    os.makedirs(model_dir, exist_ok=True)

    for filename in ['model.pkl', 'scaler.pkl', 'feature_cols.json', 'metadata.json']:
        blob = bucket.blob(f"models/{version_tag}/{filename}")
        blob.download_to_filename(f"{model_dir}/{filename}")

    model = joblib.load(f"{model_dir}/model.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")

    with open(f"{model_dir}/feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(f"{model_dir}/metadata.json") as f:
        metadata = json.load(f)

    print(f"Loaded model '{version_tag}' ({metadata['config']['model_type']})")
    print(f"  ROC-AUC: {metadata['result']['roc_auc']}")

    return model, scaler, feature_cols, metadata


def list_models():
    """List all saved models in GCS."""
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="models/"))

    meta_blobs = [b for b in blobs if b.name.endswith("metadata.json")]

    print(f"Found {len(meta_blobs)} models:\n")
    for mb in sorted(meta_blobs, key=lambda b: b.name):
        meta = json.loads(mb.download_as_text())
        config = meta.get('config', {})
        result = meta.get('result', {})
        td = meta.get('training_data', {})

        print(f"  {meta['version']}  |  {config.get('model_type', '?')}  |  "
              f"AUC: {result.get('roc_auc', '?')}  |  "
              f"{td.get('real_train_rows', '?')} real + "
              f"{td.get('synthetic_train_rows', '?')} synth")
        print(f"    Data: {meta.get('data_snapshot', '?')}  |  "
              f"{meta.get('training_date', '')[:10]}")
        if meta.get('notes'):
            print(f"    Notes: {meta['notes']}")
        print()


def compare_models() -> pd.DataFrame:
    """
    Load all saved model metadata from GCS and return a comparison DataFrame.

    Columns: version, model_type, data_snapshot, training_date,
             real_rows, synth_rows, accuracy, roc_auc,
             precision_above, recall_above, f1_above,
             precision_below, recall_below, f1_below

    Usage:
        df_comparison = compare_models()
        df_comparison.style.highlight_max(color='#d4edda')
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="models/"))
    meta_blobs = [b for b in blobs if b.name.endswith("metadata.json")]

    rows = []
    for mb in sorted(meta_blobs, key=lambda b: b.name):
        meta = json.loads(mb.download_as_text())
        r  = meta.get('result', {})
        td = meta.get('training_data', {})
        rows.append({
            'version':        meta.get('version', '?'),
            'model_type':     meta.get('config', {}).get('model_type', '?'),
            'data_snapshot':  meta.get('data_snapshot', '?'),
            'training_date':  meta.get('training_date', '')[:10],
            'real_rows':      td.get('real_train_rows', None),
            'synth_rows':     td.get('synthetic_train_rows', None),
            'accuracy':       r.get('accuracy', None),
            'roc_auc':        r.get('roc_auc', None),
            'precision_above': r.get('precision_above', None),
            'recall_above':    r.get('recall_above', None),
            'f1_above':        r.get('f1_above', None),
            'precision_below': r.get('precision_below', None),
            'recall_below':    r.get('recall_below', None),
            'f1_below':        r.get('f1_below', None),
        })

    df = pd.DataFrame(rows).set_index('version')
    print(f"Loaded {len(df)} model snapshots")
    return df