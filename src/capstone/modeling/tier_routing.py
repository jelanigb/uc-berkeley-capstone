"""
TierRoutedClassifier + train_tier_models — tier-specialized modeling.

Background
----------
The v6.2 global models show a uniform ~0.037 ROC-AUC drop on tier=S (small
channels) relative to their global score, while tier=M / tier=L stay strong.
This module supports training one sub-model per tier and routing each row to its
tier's sub-model at prediction time, so a tier=S specialist can be evaluated
without disturbing M / L.

Evaluation parity
-----------------
Routing happens at the *prediction* level: every row is scored by its tier's
sub-model, the per-row probabilities are reassembled into one full-length
vector, and metrics are computed once over that pooled vector — identical to how
the global models are scored. This is NOT an average of per-tier metrics; ROC-AUC
in particular is non-decomposable (pooled AUC != mean of per-tier AUCs, because
the pooled score also ranks S-rows against M / L-rows).

Row routing keys off the DataFrame index: TierRoutedClassifier looks up each
row's tier from a `tier_lookup` Series indexed the same way as the feature
matrices. Because Scaler preserves the index and SegmentAuditor masks rows
without dropping it, the wrapper drops into run.models, Validator, SegmentAuditor,
and the metrics tables with no changes to those.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone


def build_tier_lookup(*frames: pd.DataFrame, tier_col: str = "tier") -> pd.Series:
    """Combine the `tier` column from one or more split frames into a single
    index -> tier Series spanning every row those frames contain.

    The frames (df_train / df_test / df_val / df_gen) carry disjoint subsets of
    the engineered index, so concatenation yields a unique index covering all
    rows the wrapper might be asked to score. A duplicate-index check guards
    against accidentally passing overlapping frames.
    """
    if not frames:
        raise ValueError("build_tier_lookup needs at least one frame.")
    lookup = pd.concat([f[tier_col] for f in frames])
    if lookup.index.has_duplicates:
        dupes = lookup.index[lookup.index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"build_tier_lookup got overlapping indices across frames (e.g. {dupes}). "
            "Pass non-overlapping splits (df_train / df_test / df_val / df_gen)."
        )
    return lookup


def train_tier_models(
    global_models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_train: pd.DataFrame,
    families,
    tier_col: str = "tier",
) -> dict:
    """Fit one sub-model per tier for each requested family.

    Each sub-model is a fresh ``clone()`` of the corresponding trained global
    model — so it inherits the exact v6.2 hyperparameters — refit on only that
    tier's training rows. Returns ``{family: {tier: fitted_model}}``.

    Parameters
    ----------
    global_models : dict
        A run.models-style dict (``{name: {"model": est, ...}}``) or
        ``{name: est}``. Each requested family is looked up here for its base
        estimator and params.
    X_train, y_train, df_train : aligned train split
        ``X_train`` (scaled features) must be row-aligned with ``df_train``
        (same index and order); ``df_train`` supplies the ``tier`` labels.
    families : iterable of str
        Keys into ``global_models`` to specialize, e.g. ``["xgb", "lgb",
        "ensemble_stacking"]``.
    """
    tiers = df_train[tier_col].to_numpy()
    out = {}
    for fam in families:
        base = _unwrap_(global_models[fam])
        per_tier = {}
        for tier in pd.unique(tiers):
            mask = tiers == tier
            est = clone(base)
            est.fit(X_train[mask], y_train[mask])
            per_tier[tier] = est
            print(f"  {fam}: trained tier={tier} sub-model on {int(mask.sum()):,} rows")
        out[fam] = per_tier
    return out


def _unwrap_(entry):
    """Accept either a run.models entry dict or a bare estimator."""
    return entry["model"] if isinstance(entry, dict) else entry


class TierRoutedClassifier:
    """Routes each row to its tier's sub-model; behaves like one classifier.

    Implements just enough of the sklearn classifier surface
    (``predict_proba`` / ``predict`` / ``classes_``) to slot into run.models,
    Validator, SegmentAuditor, and ``ModelResult.from_sklearn`` unchanged.

    Parameters
    ----------
    tier_models : dict[str, estimator]
        Fitted sub-model per tier, e.g. ``{"S": est_S, "M": est_M, "L": est_L}``.
        For an S-specialist hybrid, pass the global model for M and L:
        ``{"S": specialist_S, "M": global_est, "L": global_est}``.
    tier_lookup : pd.Series
        index -> tier label, aligned with the index of any X passed to
        ``predict`` / ``predict_proba``. Build with :func:`build_tier_lookup`.
    classes : array-like, default (0, 1)
        Class labels ordered to match the ``predict_proba`` columns. The
        sub-models must share this ordering (they do when trained on the same
        binary target).
    threshold : float, default 0.5
        Operating threshold used by ``predict``.
    """

    def __init__(self, tier_models: dict, tier_lookup: pd.Series, classes=(0, 1), threshold: float = 0.5):
        if not tier_models:
            raise ValueError("tier_models must not be empty.")
        self.tier_models = dict(tier_models)
        self.tier_lookup = tier_lookup
        self.classes_ = np.asarray(classes)
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "TierRoutedClassifier needs a DataFrame so rows can be routed by "
                f"index; got {type(X).__name__}. Pass the scaled feature DataFrame "
                "(run.X_val / run.X_test), not a bare ndarray."
            )
        tiers = self.tier_lookup.reindex(X.index)
        if tiers.isna().any():
            missing = X.index[tiers.isna().to_numpy()][:5].tolist()
            raise KeyError(
                f"{int(tiers.isna().sum())} row(s) have no tier in tier_lookup "
                f"(e.g. index {missing}). Rebuild the lookup to cover this X."
            )

        tiers = tiers.to_numpy()
        proba = np.empty((len(X), len(self.classes_)), dtype=float)
        routed = np.zeros(len(X), dtype=bool)
        for tier, model in self.tier_models.items():
            mask = tiers == tier
            if not mask.any():
                continue
            proba[mask] = model.predict_proba(X.iloc[np.flatnonzero(mask)])
            routed |= mask
        if not routed.all():
            unrouted = sorted(set(tiers[~routed]))
            raise KeyError(
                f"No sub-model for tier(s) {unrouted}; tier_models covers "
                f"{sorted(self.tier_models)}."
            )
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        positive = self.predict_proba(X)[:, 1] >= self.threshold
        return self.classes_[positive.astype(int)]
