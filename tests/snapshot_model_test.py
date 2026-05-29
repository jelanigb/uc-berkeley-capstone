"""Unit tests for the global (macro) metrics added to ModelResult and the
compare_models helper. GCS-touching functions (save_model, compare_models,
load_*) are not exercised here — only the pure metric logic is.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.snapshot_model import ModelResult, _macro_metric_


def _fitted_model_and_data(seed: int = 0):
    """Small, separable-ish binary problem so metrics are well-defined."""
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.standard_normal((n, 4))
    # Signal in column 0 so the model learns something non-trivial.
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, X, y


def test_model_result_has_total_fields():
    model, X, y = _fitted_model_and_data()
    result = ModelResult.from_sklearn(model, X, y, [f"f{i}" for i in range(4)])
    for field in ("precision_macro", "recall_macro", "f1_macro"):
        assert hasattr(result, field)
        assert 0.0 <= getattr(result, field) <= 1.0


def test_total_metrics_match_sklearn_macro():
    model, X, y = _fitted_model_and_data()
    result = ModelResult.from_sklearn(model, X, y, [f"f{i}" for i in range(4)])
    y_pred = model.predict(X)
    assert result.f1_macro == round(f1_score(y, y_pred, average="macro"), 4)
    assert result.precision_macro == round(precision_score(y, y_pred, average="macro"), 4)
    assert result.recall_macro == round(recall_score(y, y_pred, average="macro"), 4)


def test_f1_macro_equals_mean_of_per_class():
    model, X, y = _fitted_model_and_data()
    result = ModelResult.from_sklearn(model, X, y, [f"f{i}" for i in range(4)])
    # Macro F1 is the unweighted mean of the per-class F1 scores.
    expected = round((result.f1_above + result.f1_below) / 2, 4)
    assert abs(result.f1_macro - expected) <= 0.0001


# --- _macro_metric_ derivation for historical snapshots -----------------------

def test_macro_metric_prefers_stored_total():
    result = {"f1_macro": 0.9, "f1_above": 0.5, "f1_below": 0.5}
    assert _macro_metric_(result, "f1") == 0.9


def test_macro_metric_derives_from_per_class_when_total_absent():
    result = {"precision_above": 0.80, "precision_below": 0.60}
    assert _macro_metric_(result, "precision") == 0.70


def test_macro_metric_returns_none_when_unavailable():
    assert _macro_metric_({}, "recall") is None
