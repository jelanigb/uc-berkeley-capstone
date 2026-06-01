"""Unit tests for modeling.tier_routing.

Routing correctness is the load-bearing behavior: each row must be scored by its
own tier's sub-model and reassembled in the input's row order, even when the
caller passes a non-contiguous index subset (as SegmentAuditor does when masking
a segment). A per-tier constant-probability stub makes the routed-to model
unambiguous, so the assembled output reveals any mis-routing or mis-ordering.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from modeling.tier_routing import (
    TierRoutedClassifier,
    build_tier_lookup,
    train_tier_models,
)


class _ConstProba:
    """Stub classifier whose positive-class probability is a fixed constant,
    so the sub-model a row was routed to is identifiable from the output."""

    classes_ = np.array([0, 1])

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _lookup_and_models():
    # Deliberately shuffled, non-contiguous index to catch order/index bugs.
    idx = [10, 11, 12, 13, 14, 15]
    tiers = ["S", "M", "L", "S", "L", "M"]
    lookup = pd.Series(tiers, index=idx, name="tier")
    tier_models = {"S": _ConstProba(0.1), "M": _ConstProba(0.5), "L": _ConstProba(0.9)}
    X = pd.DataFrame(np.zeros((6, 3)), index=idx)
    return lookup, tier_models, X, tiers


def test_predict_proba_routes_each_row_to_its_tier():
    lookup, tier_models, X, tiers = _lookup_and_models()
    clf = TierRoutedClassifier(tier_models, lookup)

    proba = clf.predict_proba(X)
    expected_p = [{"S": 0.1, "M": 0.5, "L": 0.9}[t] for t in tiers]

    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba[:, 1], expected_p)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_routing_survives_masked_subset():
    """SegmentAuditor scores arbitrary row subsets; routing must still hold."""
    lookup, tier_models, X, tiers = _lookup_and_models()
    clf = TierRoutedClassifier(tier_models, lookup)

    mask = np.array([t == "S" for t in tiers])  # rows at index 10 and 13
    proba = clf.predict_proba(X[mask])

    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba[:, 1], [0.1, 0.1])


def test_predict_applies_threshold():
    lookup, tier_models, X, _ = _lookup_and_models()
    clf = TierRoutedClassifier(tier_models, lookup, threshold=0.5)
    # S=0.1 -> 0, M=0.5 -> 1 (>=), L=0.9 -> 1
    expected = [0, 1, 1, 0, 1, 1]
    np.testing.assert_array_equal(clf.predict(X), expected)


def test_missing_tier_in_lookup_raises():
    lookup, tier_models, X, _ = _lookup_and_models()
    clf = TierRoutedClassifier(tier_models, lookup)
    X_unknown = X.rename(index={10: 999})  # 999 not in lookup
    with pytest.raises(KeyError, match="no tier in tier_lookup"):
        clf.predict_proba(X_unknown)


def test_unrouted_tier_raises():
    lookup, _, X, _ = _lookup_and_models()
    partial = {"S": _ConstProba(0.1), "M": _ConstProba(0.5)}  # no L model
    clf = TierRoutedClassifier(partial, lookup)
    with pytest.raises(KeyError, match="No sub-model for tier"):
        clf.predict_proba(X)


def test_ndarray_input_raises():
    lookup, tier_models, X, _ = _lookup_and_models()
    clf = TierRoutedClassifier(tier_models, lookup)
    with pytest.raises(TypeError, match="DataFrame"):
        clf.predict_proba(X.to_numpy())


def test_build_tier_lookup_rejects_overlapping_indices():
    a = pd.DataFrame({"tier": ["S", "M"]}, index=[0, 1])
    b = pd.DataFrame({"tier": ["L"]}, index=[1])  # index 1 overlaps a
    with pytest.raises(ValueError, match="overlapping indices"):
        build_tier_lookup(a, b)


def test_build_tier_lookup_spans_all_frames():
    a = pd.DataFrame({"tier": ["S", "M"]}, index=[0, 1])
    b = pd.DataFrame({"tier": ["L"]}, index=[2])
    lookup = build_tier_lookup(a, b)
    assert lookup.to_dict() == {0: "S", 1: "M", 2: "L"}


def test_train_tier_models_fits_one_model_per_tier():
    rng = np.random.default_rng(0)
    n = 90
    df_train = pd.DataFrame({"tier": (["S"] * 30 + ["M"] * 30 + ["L"] * 30)})
    X_train = pd.DataFrame(rng.standard_normal((n, 4)))
    y_train = pd.Series(rng.integers(0, 2, size=n))
    globals_ = {"lr": {"model": LogisticRegression(C=0.3, max_iter=200)}}

    out = train_tier_models(globals_, X_train, y_train, df_train, families=["lr"])

    assert set(out["lr"]) == {"S", "M", "L"}
    # Each sub-model is a fresh clone carrying the global's params, refit per tier.
    for tier, est in out["lr"].items():
        assert est.get_params()["C"] == 0.3
        assert hasattr(est, "coef_")  # fitted
