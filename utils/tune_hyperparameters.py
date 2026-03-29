"""
Hyperparameter tuning utilities for sklearn-compatible classifiers.

Supports three search strategies selectable via the `search_strategy` param:
  "random"   — RandomizedSearchCV  (default; fast, near-optimal)
  "halving"  — HalvingGridSearchCV (progressive resource allocation; good middle ground)
  "grid"     — GridSearchCV        (exhaustive; slow on large grids)

Usage:
    from utils.tune_hyperparameters import tune_model, get_default_param_grid

    param_grid = get_default_param_grid(model_xgb)
    result = tune_model(model_xgb, X_train, y_train, param_grid)

    print(result['best_params'])
    print(result['best_score'])

    # Re-initialize model with best params and retrain
    model_xgb.set_params(**result['best_params'])
    model_xgb.fit(X_train, y_train)
"""

from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 — required to enable HalvingGridSearchCV
from sklearn.model_selection import HalvingGridSearchCV


# =============================================================================
# Default parameter grids per model type
# =============================================================================

_PARAM_GRIDS = {
    "LogisticRegression": {
        "C":        [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
        "penalty":  ["l1"],        # keep L1 — established in project
        "solver":   ["saga"],      # only solver that supports L1 + large datasets
        "max_iter": [2000, 5000],
    },
    "RandomForestClassifier": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [None, 5, 10, 20, 30],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split":[2, 5, 10],
        "max_features":     ["sqrt", "log2", None],
    },
    "XGBClassifier": {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [3, 4, 5, 6, 8],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5, 10],
        "gamma":             [0, 0.1, 0.3, 0.5],
    },
}

_STRATEGY_MAP = {
    "random":  RandomizedSearchCV,
    "halving": HalvingGridSearchCV,
    "grid":    GridSearchCV,
}


def get_default_param_grid(model) -> dict:
    """
    Return a sensible default parameter grid for the given model instance.
    Raises ValueError for unsupported model types.
    """
    class_name = type(model).__name__
    if class_name not in _PARAM_GRIDS:
        supported = list(_PARAM_GRIDS.keys())
        raise ValueError(
            f"No default param grid for '{class_name}'. "
            f"Supported: {supported}. Pass a custom param_grid instead."
        )
    return _PARAM_GRIDS[class_name].copy()


def tune_model(
    model,
    X_train,
    y_train,
    param_grid: dict,
    search_strategy: str = "random",
    cv: int = 5,
    n_iter: int = 50,        # RandomizedSearchCV only
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> dict:
    """
    Run a hyperparameter search and return results.

    Parameters
    ----------
    model            : unfitted or pre-configured sklearn-compatible classifier
    X_train, y_train : training data
    param_grid       : dict of param name → list of values to search
    search_strategy  : "random" (default), "halving", or "grid"
    cv               : number of cross-validation folds
    n_iter           : number of random samples (RandomizedSearchCV only)
    scoring          : sklearn scoring string, default "roc_auc"
    random_state     : for reproducibility (RandomizedSearchCV + HalvingGridSearchCV)
    n_jobs           : parallel jobs (-1 = all cores)
    verbose          : verbosity level passed to the searcher

    Returns
    -------
    dict with keys:
        best_params  — dict of best hyperparameters
        best_score   — best CV score
        searcher     — fitted searcher object (access .cv_results_ for full results)
        model_type   — class name of the model
        strategy     — search strategy used
    """
    if search_strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown search_strategy '{search_strategy}'. "
            f"Choose from: {list(_STRATEGY_MAP.keys())}"
        )

    SearchClass = _STRATEGY_MAP[search_strategy]

    # Build kwargs — n_iter and random_state only apply to certain strategies
    kwargs = dict(
        estimator=model,
        param_grid=param_grid if search_strategy != "random" else None,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if search_strategy == "random":
        kwargs["param_distributions"] = param_grid
        kwargs.pop("param_grid")
        kwargs["n_iter"] = n_iter
        kwargs["random_state"] = random_state
    elif search_strategy == "halving":
        kwargs["random_state"] = random_state

    searcher = SearchClass(**kwargs)

    model_name = type(model).__name__
    print(f"Tuning {model_name} with strategy='{search_strategy}' "
          f"(cv={cv}, scoring='{scoring}')")
    if search_strategy == "random":
        print(f"  Sampling {n_iter} combinations from grid of "
              f"~{_grid_size(param_grid):,} total")

    searcher.fit(X_train, y_train)

    print(f"\nBest {scoring}: {searcher.best_score_:.4f}")
    print(f"Best params:   {searcher.best_params_}")

    return {
        "best_params": searcher.best_params_,
        "best_score":  searcher.best_score_,
        "searcher":    searcher,
        "model_type":  model_name,
        "strategy":    search_strategy,
    }


def _grid_size(param_grid: dict) -> int:
    """Count total combinations in a param grid."""
    total = 1
    for vals in param_grid.values():
        total *= len(vals)
    return total
