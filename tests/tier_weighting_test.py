"""Unit tests for modeling.tier_weighting.

Two behaviors carry the risk: the {tier: multiplier} -> per-row weight mapping
(defaults, validation, row alignment), and that train_weighted_models actually
threads sample_weight through a fresh clone of the global. A stub estimator
captures the sample_weight it receives so the wiring is checked without training
a real model.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from modeling.tier_weighting import tier_sample_weights, train_weighted_models


def _df(tiers):
    return pd.DataFrame({"tier": tiers})


def test_weights_map_per_tier_with_default_one():
    df = _df(["S", "M", "L", "S"])
    w = tier_sample_weights(df, {"S": 3.0})
    np.testing.assert_array_equal(w, [3.0, 1.0, 1.0, 3.0])


def test_weights_respect_row_order():
    df = _df(["M", "S", "L"])
    w = tier_sample_weights(df, {"S": 2.0, "L": 5.0})
    np.testing.assert_array_equal(w, [1.0, 2.0, 5.0])


def test_empty_weights_raise():
    with pytest.raises(ValueError, match="must not be empty"):
        tier_sample_weights(_df(["S"]), {})


def test_nonpositive_weight_raises():
    with pytest.raises(ValueError, match="positive"):
        tier_sample_weights(_df(["S", "M"]), {"S": 0})


class _CapturingEstimator(LogisticRegression):
    """LogisticRegression that records the sample_weight it was fit with."""

    def fit(self, X, y, sample_weight=None):
        self.seen_weight_ = sample_weight
        return super().fit(X, y, sample_weight=sample_weight)


def test_train_weighted_threads_sample_weight_into_clone():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.standard_normal((n, 3)))
    y = pd.Series(rng.integers(0, 2, size=n))
    w = np.where(np.arange(n) % 3 == 0, 4.0, 1.0)
    globals_ = {"lr": {"model": _CapturingEstimator(C=0.7, max_iter=200)}}

    out = train_weighted_models(globals_, X, y, w, families=["lr"])

    fitted = out["lr"]
    # It's a fresh clone (not the same object) carrying the global's params...
    assert fitted is not globals_["lr"]["model"]
    assert fitted.get_params()["C"] == 0.7
    # ...and the exact weights were passed through to fit.
    np.testing.assert_array_equal(fitted.seen_weight_, w)
    assert hasattr(fitted, "coef_")  # actually fitted


def test_mismatched_weight_length_raises():
    X = pd.DataFrame(np.zeros((5, 2)))
    y = pd.Series([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="!= X_train rows"):
        train_weighted_models(
            {"lr": LogisticRegression()}, X, y, np.ones(4), families=["lr"]
        )
