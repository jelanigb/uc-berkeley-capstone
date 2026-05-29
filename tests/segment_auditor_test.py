"""Unit tests for SegmentAuditor.

Uses a synthetic fixture with 200 rows, two verticals ('A', 'B'), two tiers ('S', 'L'),
and a DummyClassifier so no real model training is required.

Coverage:
  - Output DataFrame has expected columns
  - Row count equals n_models × n_dimensions × n_unique_segment_values
  - Segments with < 30 samples are excluded from results (with a warning)
  - A segment where all labels are the same class produces roc_auc=NaN without crashing
  - Missing 'vertical' / 'tier' column raises KeyError
  - Non-RangeIndex df_val raises AssertionError
  - Model without predict_proba raises AttributeError
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from evaluation.segment_auditor import SegmentAuditor

# ── Shared fixture helpers ─────────────────────────────────────────────────────


def _make_df_val(n: int, verticals, tiers) -> pd.DataFrame:
    return pd.DataFrame({"vertical": verticals, "tier": tiers})


def _make_fixture(n: int = 200, seed: int = 42):
    """200 rows, 2 verticals (A/B), 2 tiers (S/L), stratified 50/50 split."""
    rng = np.random.default_rng(seed)
    half = n // 2
    verticals = (["A"] * half + ["B"] * half)
    tiers = (["S"] * (n // 4) + ["L"] * (n // 4)) * 2
    df_val = _make_df_val(n, verticals, tiers)
    y_val = pd.Series(rng.integers(0, 2, size=n), name="above_baseline")
    X_val = rng.standard_normal((n, 5))

    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit(X_val, y_val)
    models = {"dummy": {"model": clf, "scaler": None}}
    return models, X_val, y_val, df_val


# ── Expected columns ───────────────────────────────────────────────────────────


def test_audit_has_expected_columns():
    models, X_val, y_val, df_val = _make_fixture()
    results = SegmentAuditor(models, X_val, y_val, df_val).audit()
    expected = {
        "model", "segment_type", "segment_value", "n_samples",
        "pct_positive", "roc_auc", "accuracy", "precision_above",
        "recall_above", "f1_above", "recall_below",
    }
    assert expected.issubset(set(results.columns))


# ── Row count ─────────────────────────────────────────────────────────────────


def test_audit_row_count_single_model():
    """1 model × 2 dimensions × 2 unique values each = 4 rows."""
    models, X_val, y_val, df_val = _make_fixture()
    results = SegmentAuditor(models, X_val, y_val, df_val).audit()
    assert len(results) == 4


def test_audit_row_count_two_models():
    """2 models × 2 dimensions × 2 unique values = 8 rows."""
    rng = np.random.default_rng(0)
    n = 200
    half = n // 2
    verticals = ["A"] * half + ["B"] * half
    tiers = (["S"] * (n // 4) + ["L"] * (n // 4)) * 2
    df_val = _make_df_val(n, verticals, tiers)
    y_val = pd.Series(rng.integers(0, 2, size=n))
    X_val = rng.standard_normal((n, 5))

    clf_a = DummyClassifier(strategy="most_frequent").fit(X_val, y_val)
    clf_b = DummyClassifier(strategy="stratified", random_state=1).fit(X_val, y_val)
    models = {
        "model_a": {"model": clf_a},
        "model_b": {"model": clf_b},
    }
    results = SegmentAuditor(models, X_val, y_val, df_val).audit()
    assert len(results) == 8


# ── Small segment exclusion ───────────────────────────────────────────────────


def test_small_segment_excluded_and_warns():
    """Segments with < 30 samples are skipped with a UserWarning."""
    rng = np.random.default_rng(0)
    n = 100
    # vertical=B has only 5 rows — below the 30-sample minimum
    verticals = ["A"] * 95 + ["B"] * 5
    tiers = ["S"] * n
    df_val = _make_df_val(n, verticals, tiers)
    y_val = pd.Series(rng.integers(0, 2, size=n))
    X_val = rng.standard_normal((n, 5))

    clf = DummyClassifier(strategy="stratified", random_state=0).fit(X_val, y_val)
    models = {"dummy": {"model": clf}}

    with pytest.warns(UserWarning, match="Skipping"):
        results = SegmentAuditor(models, X_val, y_val, df_val).audit()

    vert_results = results[results["segment_type"] == "vertical"]
    assert "B" not in vert_results["segment_value"].values
    assert "A" in vert_results["segment_value"].values


# ── All-same-label segment → roc_auc = NaN ───────────────────────────────────


def test_all_same_label_produces_nan_roc_auc():
    """A segment where every label is the same class produces roc_auc=NaN, not an exception."""
    rng = np.random.default_rng(1)
    n = 120
    # vertical=A: all positive; vertical=B: mixed
    verticals = ["A"] * 60 + ["B"] * 60
    tiers = ["S"] * n
    df_val = _make_df_val(n, verticals, tiers)
    y_val = pd.Series(np.array([1] * 60 + list(rng.integers(0, 2, size=60))))
    X_val = rng.standard_normal((n, 5))

    clf = DummyClassifier(strategy="stratified", random_state=1).fit(X_val, y_val)
    models = {"dummy": {"model": clf}}

    results = SegmentAuditor(models, X_val, y_val, df_val).audit()

    a_row = results[
        (results["segment_type"] == "vertical") & (results["segment_value"] == "A")
    ]
    assert len(a_row) == 1
    assert np.isnan(a_row["roc_auc"].iloc[0])


# ── Edge case: missing column ─────────────────────────────────────────────────


def test_missing_vertical_column_raises():
    models, X_val, y_val, _ = _make_fixture()
    df_bad = pd.DataFrame({"tier": ["S"] * len(y_val)})
    with pytest.raises(KeyError, match="vertical"):
        SegmentAuditor(models, X_val, y_val, df_bad).audit()


def test_missing_tier_column_raises():
    models, X_val, y_val, _ = _make_fixture()
    df_bad = pd.DataFrame({"vertical": ["A"] * len(y_val)})
    with pytest.raises(KeyError, match="tier"):
        SegmentAuditor(models, X_val, y_val, df_bad).audit()


# ── Edge case: non-default index ──────────────────────────────────────────────


def test_non_range_index_df_val_still_works():
    """df_val with a non-default index (as DataSplitter produces) is handled correctly.

    label extraction calls reset_index(drop=True) on the Series, so positional
    alignment with X_val is preserved regardless of df_val's index.
    """
    models, X_val, y_val, df_val = _make_fixture()
    df_shifted = df_val.copy()
    df_shifted.index = df_shifted.index + 10  # simulate DataSplitter's non-default index
    results = SegmentAuditor(models, X_val, y_val, df_shifted).audit()
    assert len(results) == 4  # same output as RangeIndex case


# ── Edge case: model without predict_proba ────────────────────────────────────


def test_model_without_predict_proba_raises():
    class NoProbaModel:
        def predict(self, X):
            return np.zeros(len(X))

    _, X_val, y_val, df_val = _make_fixture()
    models = {"bad": {"model": NoProbaModel()}}
    with pytest.raises(AttributeError, match="predict_proba"):
        SegmentAuditor(models, X_val, y_val, df_val).audit()
