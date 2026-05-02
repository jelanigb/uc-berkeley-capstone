# EDA Module design

**Author:** Jelani Gould-Bailey
**LLM Co-Author:** Gemini Thinking, Claude Sonnet 4.6
**Creation Date:** April 29, 2026
**Last Updated:** May 2, 2026

---

## 1. Updated `PipelineRun` Design

To support persistent EDA settings without cluttering the notebook, we are adding an `eda_state` field. This allows the user to switch context (e.g., from `df_videos` to `df_engineered`) and have all subsequent plots automatically use that context.

### Modified `pipeline/pipeline_run.py`

```python
@dataclass
class PipelineRun:
    config: VersionConfig
    
    # ... [Existing data fields: df_videos, df_clean, df_engineered, etc.] ...

    # --- EDA State (New) ---
    # The dataframe currently targeted by eda functions (e.g., df_clean, X_train)
    active_eda_df: Optional[pd.DataFrame] = None

    # Loaded by eda.load_model_comparison(run); used by all model comparison plots
    model_comparison_df: Optional[pd.DataFrame] = None
    
    # Persistent plotting config (e.g., {'fig_size': (12, 6), 'style': 'whitegrid'})
    eda_config: dict = field(default_factory=lambda: {
        "fig_size": (10, 6),
        "palette": "viridis",
    })

    # ... [Existing methods: assert_ready_for, summary] ...
```

For lightweight analysis notebooks that don't need the full `PipelineRun` (e.g., model results analysis), a `SimpleNamespace` with the same shape is sufficient:

```python
from types import SimpleNamespace
run = SimpleNamespace(
    eda_config={"fig_size": (14, 7), "palette": "Set2"},
    model_comparison_df=None,
)
```

---

## 2. The `eda.py` Module Design

The module is designed as a suite of functional "tools." It abstracts the complex joining logic required to get human-readable labels (Vertical/Tier) back onto engineered data. All plotting functions accept an optional `save_figure_name` argument; when provided, the plot is saved under `images/eda/`.

### `pipeline/stages/eda.py`

#### **Configuration & State Management**

* **`set_active_df(run, df)`**: Sets `run.active_eda_df` directly, allowing for overrides at any point in the EDA execution flow (e.g., switching from a pre-feature-engineering DF to a post-engineering one).
* **`set_fig_size(run, width, height)`**: Updates the matplotlib figure size for all subsequent plots.
* **`set_palette(run, palette)`**: Updates the seaborn palette for all subsequent plots.
* **`get_plt(run)`**: Returns the active `plt` object for any custom Matplotlib interactions not covered by the module.
* **`_get_readable_df(run)`**: *Internal helper.* If the active DF is engineered data, performs a left-join back to `df_clean` on `video_id` to restore categorical "Vertical" and "Tier" labels for legends. Guards against X_train/X_test arrays that lack `video_id`.

#### **Feature EDA Plotting Functions**

* **`plot_label_rates(run)`**: Grouped bar chart of `above_baseline` success rate by Vertical and Tier.
* **`plot_engagement_distribution(run)`**: Paginated histograms (no KDE) for all continuous features, `_FEATURES_PER_PAGE` per page.
* **`plot_kde_distributions(run, features=None)`**: Histograms with pre-computed scipy KDE curves for priority engagement features (or a custom list). Paginated.
* **`plot_feature_correlations(run, target='above_baseline')`**: Full feature correlation heatmap. Annotated when ≤ 25 features; auto-scales figure size. Doubles as a leakage check.
* **`plot_target_correlations(run, target='above_baseline')`**: Focused horizontal bar chart of per-feature Pearson correlations with the target, sorted by absolute magnitude.
* **`plot_vertical_segmentation(run, feature='view_count_24h')`**: Side-by-side boxplot and per-vertical KDE overlay for a single feature.

---

## 3. Model Comparison Functions

These functions compare trained model snapshots across versions and model types. They read from `run.model_comparison_df` (populated by `load_model_comparison`), and accept `compare_versions` and `model_type` filters consistently across all plot functions.

### Supporting utilities in `utils/snapshot_model.py`

#### **`ModelType` enum**

```python
class ModelType(str, Enum):
    XGB         = "XGBoost"
    LR          = "LogisticRegression"
    RF          = "RandomForest"
    ENSEMBLE_VC = "VotingClassifier"
```

Used to filter plots to one or more model types. The display name `"Ensemble (VotingClassifier)"` is applied in the EDA layer (see `_MODEL_DISPLAY_NAMES` below) so the raw GCS metadata is unchanged.

#### **`filter_comparison(df, compare_versions, model_type)`**

| `compare_versions` value | Behaviour |
| --- | --- |
| `"all"` (default) | No version filtering |
| `"last2versions"` | Two most recent distinct version prefixes |
| `"last3versions"` | Three most recent distinct version prefixes |
| `["v4.0", "v5.1"]` | Rows whose version tag contains any listed substring |

`model_type` accepts `None` (all types), a single `ModelType`, or a list of `ModelType` values.

#### **Other utilities**

* **`get_version_prefixes(df)`**: Returns chronologically sorted unique `vX.Y` prefixes extracted from version tags.
* **`load_top_features_over_time(model_type, compare_versions)`**: Downloads `metadata.json` from GCS for each matching snapshot and returns a tidy `[version, version_prefix, feature, importance]` DataFrame suitable for pivot → heatmap.
* **`save_comparison_to_gcs(df)`**: Writes the comparison DataFrame to `gs://{BUCKET}/evals/model_comparison.jsonl`. Not versioned — overwrites on each call.

### Model comparison functions in `pipeline/stages/eda.py`

#### **State Management**

* **`load_model_comparison(run)`**: Calls `compare_models()` and stores the result as `run.model_comparison_df`. Must be called before any comparison plot.

#### **Internal helpers**

* **`_MODEL_DISPLAY_NAMES`**: Dict mapping raw `model_type` strings to plot-friendly labels (e.g., `"VotingClassifier"` → `"Ensemble (VotingClassifier)"`). Add entries here when new model types are introduced.
* **`_prep_comparison_df(df)`**: Resets the version index, adds a sorted categorical `version_prefix` column, and adds a `model_type_label` column using `_MODEL_DISPLAY_NAMES`.
* **`_add_eval_divider(ax, sorted_prefixes)`**: Draws a dashed vertical line between the last pre-v4.0 version and the first v4.0+ version. Labels the two regions `← test set | hold-out →`. No-ops when all displayed versions are on the same side of the boundary.

#### **Comparison Plotting Functions**

All accept `(run, compare_versions="all", model_type=None, save_figure_name=None)` unless noted.

* **`plot_roc_auc(run, ...)`**: Line chart of ROC-AUC across versions, one line per model type.
* **`plot_accuracy(run, ...)`**: Line chart of accuracy across versions, one line per model type.
* **`plot_precision_recall(run, ...)`**: 2×2 subplot grid — precision and recall for both the *Above Baseline* and *Below Baseline* classes.
* **`plot_f1(run, ...)`**: Side-by-side F1 for Above Baseline and Below Baseline.
* **`plot_data_composition_vs_accuracy(run, compare_versions, save_figure_name)`**: Correlation heatmap of `real_rows`, `synth_rows`, `total_rows` vs. `accuracy`, `roc_auc`, `f1_above`, `f1_below`. Directional signal — sample size is small (one row per snapshot).
* **`plot_top_features_over_time(run, model_type, compare_versions, top_n, save_figure_name)`**: Heatmap of the top `top_n` features (by mean absolute importance) for a **single** model type across versions. Fetches `top_features` from GCS via `load_top_features_over_time`. Works for both LR coefficients and tree-based importances.

> **Eval methodology note:** Models with version prefix `< v4.0` were evaluated on a test split drawn from the same data snapshot used for training. Models `≥ v4.0` are evaluated on a shared validation hold-out set created at the v4.0 data snapshot. All comparison plots mark this boundary with a dashed divider. Cross-boundary metric comparisons are directional, not apples-to-apples.

---

## 4. Example Implementations

### Feature EDA Workflow

```python
# Context: Pre-engineered wide data
eda.set_active_df(run, run.df_clean)
eda.plot_engagement_distribution(run)

# Context: Post-engineered (automatically joins labels for the legend)
eda.set_active_df(run, run.df_engineered)
eda.set_fig_size(run, 14, 7)
eda.plot_label_rates(run)
eda.plot_feature_correlations(run)  # includes leakage check
```

### Model Comparison Workflow

```python
from types import SimpleNamespace
from utils.snapshot_model import ModelType
from pipeline.stages import eda

run = SimpleNamespace(
    eda_config={"fig_size": (14, 7), "palette": "Set2"},
    model_comparison_df=None,
)

# Load once; all plot functions read from run.model_comparison_df
eda.load_model_comparison(run)

# All versions, all model types
eda.plot_roc_auc(run)
eda.plot_accuracy(run)
eda.plot_precision_recall(run)
eda.plot_f1(run)

# Last two version prefixes only
eda.plot_roc_auc(run, compare_versions="last2versions")

# XGB only, specific versions
eda.plot_roc_auc(run, compare_versions=["v4.0", "v5.0", "v5.1"], model_type=ModelType.XGB)

# Data composition vs. accuracy correlation
eda.plot_data_composition_vs_accuracy(run)

# Feature importance trajectory for XGB
eda.plot_top_features_over_time(run, ModelType.XGB, top_n=10)

# Save comparison snapshot to GCS
from utils.snapshot_model import save_comparison_to_gcs
save_comparison_to_gcs(run.model_comparison_df)
```

---
