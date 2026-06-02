"""
tier_weighting — up-weight an under-served tier inside the *global* model.

Background
----------
The per-tier split experiment (docs/tier_split_models.md) rejected training a
separate model per tier: the tier=S specialist lost both training volume and the
cross-tier signal transfer that the global model exploits, so it scored worse on
S *and* globally.

Sample weighting attacks the same tier=S blind spot without those costs. We clone
each trained v6.2 global, refit it on **all** training rows (full volume,
cross-tier transfer intact), but pass a `sample_weight` that up-weights tier=S so
the loss cares more about small channels. The result is a single global model —
no routing — so it evaluates directly through ModelResult.from_sklearn and
SegmentAuditor, exactly like the v6.2 globals.

The right weight is an empirical trade-off (lift tier=S without sinking the global
score), so the notebook sweeps a few multipliers and reads the curve.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone


def tier_sample_weights(
    df_train: pd.DataFrame,
    tier_weights: dict,
    tier_col: str = "tier",
) -> np.ndarray:
    """Per-row sample weights from a {tier: multiplier} map.

    Tiers absent from `tier_weights` default to 1.0. Returns a float array
    aligned with `df_train`'s row order, ready to pass as `sample_weight`.

    Example
    -------
    ``tier_sample_weights(df_train, {"S": 3.0})`` weights every tier=S row 3x and
    leaves M / L at 1x.
    """
    if not tier_weights:
        raise ValueError("tier_weights must not be empty.")
    bad = [w for w in tier_weights.values() if w <= 0]
    if bad:
        raise ValueError(f"tier weights must be positive, got {bad}.")
    tiers = df_train[tier_col].to_numpy()
    return np.array([float(tier_weights.get(t, 1.0)) for t in tiers])


def train_weighted_models(
    global_models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
    families,
) -> dict:
    """Refit a clone of each requested global model with `sample_weight`.

    Each clone inherits the global's v6.2 hyperparameters (via ``clone()``) and
    trains on the full ``X_train`` — only the per-row weighting differs. Returns
    ``{family: fitted_model}``.

    Parameters
    ----------
    global_models : dict
        run.models-style dict (``{name: {"model": est, ...}}``) or
        ``{name: est}``. Each requested family is looked up here.
    sample_weight : np.ndarray
        Per-row weights aligned with ``X_train`` (see
        :func:`tier_sample_weights`).
    families : iterable of str
        Keys into ``global_models`` to refit, e.g. ``["xgb", "lgb",
        "ensemble_stacking"]``.
    """
    if len(sample_weight) != len(X_train):
        raise ValueError(
            f"sample_weight length {len(sample_weight)} != X_train rows {len(X_train)}."
        )
    out = {}
    for fam in families:
        base = global_models[fam]
        base = base["model"] if isinstance(base, dict) else base
        est = clone(base)
        est.fit(X_train, y_train, sample_weight=sample_weight)
        out[fam] = est
    return out
