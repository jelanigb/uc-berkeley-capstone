# Implementation Plan: Per-Segment Performance Audit (`SegmentAuditor`)

## Goal

Evaluate the global XGBoost (and other trained models) on per-segment subsets of the
validation set, broken out independently by **vertical** and **tier**. This reveals whether
the global model has blind spots across content domains or channel sizes, and informs
whether segment-specific models are worth training.

---

## 1. File Location

Create a new file:

```
src/capstone/evaluation/segment_auditor.py
```

This is a Logic class — a peer to `DataSplitter` and `SyntheticAugmenter`, not a nested
inner class. It should have no dependency on the pipeline runner itself; it receives
what it needs via constructor injection.

---

## 2. Preliminary Assumption to Verify

Before writing any logic, add an assertion at the top of `audit()` that verifies the
index alignment between `df_val` and `X_val`:

```python
assert df_val.index.equals(pd.Index(range(len(df_val)))), \
    "df_val index is not a default RangeIndex — alignment with X_val may be incorrect"
assert len(df_val) == X_val.shape[0], \
    f"Row count mismatch: df_val={len(df_val)}, X_val={X_val.shape[0]}"
```

If these assertions fail, Claude Code should stop and report the actual index types of
both objects before proceeding. Do not work around a mismatch silently.

---

## 3. Class Design

```python
class SegmentAuditor:
    """
    Evaluates trained models on per-segment subsets of the validation set.

    Segments are broken out by vertical and tier independently (not crossed).
    Requires df_val (pre-feature-engineering split) to extract string segment labels,
    aligned by row position with X_val / y_val.
    """

    def __init__(
        self,
        models: dict,            # run.models — keyed by model name string
        X_val: np.ndarray,       # scaled validation feature matrix
        y_val: pd.Series,        # validation labels (above_baseline)
        df_val: pd.DataFrame,    # pre-FE validation split — must have 'vertical', 'tier'
        threshold: float = 0.5,  # classification threshold; pass optimized value if known
    ):
        ...

    def audit(self) -> pd.DataFrame:
        """
        Run per-segment evaluation for all models.
        Returns a tidy DataFrame with one row per (model, segment_type, segment_value).
        """
        ...

    def print_summary(self, results: pd.DataFrame) -> None:
        """Print a formatted summary table grouped by segment_type."""
        ...
```

---

## 4. Method: `audit()`

### 4a. Extract segment label Series from df_val

```python
# Row-position aligned with X_val / y_val
vertical_labels = df_val['vertical'].reset_index(drop=True)
tier_labels     = df_val['tier'].reset_index(drop=True)
```

### 4b. Define segment dimensions

```python
SEGMENT_DIMENSIONS = {
    'vertical': vertical_labels,
    'tier':     tier_labels,
}
```

### 4c. Per-model, per-segment loop

For each `model_name, model_obj` in `self.models.items()`:
  For each `dimension_name, label_series` in `SEGMENT_DIMENSIONS.items()`:
    For each unique value in `label_series.unique()`:
      - Build a boolean mask: `mask = (label_series == segment_value).values`
      - Slice: `X_seg = self.X_val[mask]`, `y_seg = self.y_val.values[mask]`
      - Skip if `len(y_seg) < 30` — too small for reliable metrics; log a warning
      - Get probabilities: `y_prob = model_obj.predict_proba(X_seg)[:, 1]`
      - Threshold: `y_pred = (y_prob >= self.threshold).astype(int)`
      - Compute metrics (see §4d)
      - Append one record to results list

### 4d. Metrics to compute per segment

```python
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)

record = {
    'model':           model_name,
    'segment_type':    dimension_name,   # 'vertical' or 'tier'
    'segment_value':   segment_value,    # e.g. 'Education', 'S'
    'n_samples':       len(y_seg),
    'pct_positive':    y_seg.mean(),
    'roc_auc':         roc_auc_score(y_seg, y_prob),
    'accuracy':        accuracy_score(y_seg, y_pred),
    'precision_above': precision_score(y_seg, y_pred, zero_division=0),
    'recall_above':    recall_score(y_seg, y_pred, zero_division=0),
    'f1_above':        f1_score(y_seg, y_pred, zero_division=0),
    'recall_below':    recall_score(1 - y_seg, 1 - y_pred, zero_division=0),
}
```

### 4e. Return value

Convert the results list to a DataFrame. Sort by `['segment_type', 'model',
'segment_value']`. Return it.

---

## 5. Method: `print_summary()`

Print two blocks — one for `segment_type == 'vertical'`, one for `segment_type == 'tier'`.

Within each block, show a table with models as columns and segment values as rows,
displaying `roc_auc` and `accuracy` side by side. Also print the global (all-segment)
metrics for each model as a reference row labeled `"ALL"`.

Use `pd.pivot_table` or manual formatting — whichever is cleaner. The output should
be readable without a notebook renderer (plain text table via `tabulate` or manual
`str.format` alignment).

---

## 6. Integration Point

In the main pipeline notebook (or `pipeline_runner.py` if applicable), add the audit
call **after** models are trained and the optimal threshold is determined, and **before**
the holdout evaluation:

```python
from capstone.evaluation.segment_auditor import SegmentAuditor

auditor = SegmentAuditor(
    models=run.models,        # dict of {model_name: fitted model}
    X_val=run.X_val,
    y_val=run.y_val,
    df_val=run.df_val,        # pre-FE split with string vertical/tier columns
    threshold=0.58,           # optimized threshold from threshold sweep
)

segment_results = auditor.audit()
auditor.print_summary(segment_results)

# Optionally persist
segment_results.to_csv('outputs/segment_audit_v5.3.csv', index=False)
```

---

## 7. Edge Cases to Handle

| Case | Handling |
|---|---|
| Segment with < 30 samples | Skip, print warning with segment name and count |
| `y_seg` is all one class (no positive or no negative) | `roc_auc_score` will raise — catch `ValueError`, set `roc_auc = np.nan` |
| Model without `predict_proba` | Raise `AttributeError` with a descriptive message |
| `df_val` missing `vertical` or `tier` column | Raise `KeyError` with column name in message |

---

## 8. Unit Tests

Create `tests/test_segment_auditor.py`. Use a synthetic fixture with 200 rows,
two verticals (`'A'`, `'B'`), two tiers (`'S'`, `'L'`), a dummy binary classifier
(e.g. `DummyClassifier`), and verify:

- Output DataFrame has expected columns
- Row count equals `n_models × n_dimensions × n_unique_segment_values`
- Segments with < 30 samples are excluded from results
- A segment where all labels are the same produces `roc_auc = np.nan` without crashing

---

## 9. What to Report From the Output

The primary question this audit answers: **does the global model perform uniformly
across segments, or does it have blind spots?**

Flag any segment where `roc_auc` drops more than **0.03 below** the global model AUC
(XGBoost global = 0.909). Those are candidates for segment-specific model investment.
