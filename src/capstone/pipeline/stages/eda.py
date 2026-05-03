import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc

# --- Configuration & State Management ---


def set_active_df(run, active_df: pd.DataFrame):
    """
    Sets the active_eda_df on the PipelineRun object.
    Allows switching between raw, clean, and engineered data contexts.
    """
    run.active_eda_df = active_df


def set_fig_size(run, width: int, height: int):
    """Updates the persistent plotting configuration in the PipelineRun."""
    run.eda_config["fig_size"] = (width, height)


def set_palette(run, palette: str):
    """Updates the seaborn palette used by all subsequent plots."""
    run.eda_config["palette"] = palette


def get_plt(run):
    """Returns the matplotlib.pyplot object for custom manual adjustments."""
    return plt


def _get_readable_df(run):
    """
    Internal Helper: Joins categorical labels (Vertical/Tier) back to
    engineered data if they are missing, ensuring clean legends.
    """
    if run.active_eda_df is None:
        raise RuntimeError(
            "No active EDA DataFrame set. Call eda.set_active_df(run, df) first."
        )

    df = run.active_eda_df.copy()

    # If working with engineered data, categorical labels are often encoded or dropped.
    # Join them back from df_clean for visualization purposes.
    # Guard: only attempt the join when video_id is present (X_train / X_test lack it).
    if (
        "vertical" not in df.columns
        and "video_id" in df.columns
        and run.df_clean is not None
    ):
        labels = run.df_clean[["video_id", "vertical", "tier"]]
        df = df.merge(labels, on="video_id", how="left")
    return df


def _is_valid_figure_name(name: str) -> bool:
    if not name:
        return False
    #if re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
    #    return False
    valid_exts = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}
    ext = os.path.splitext(name)[1].lower()
    return ext in valid_exts


def _save_fig(plt, figure_name: str, page: Optional[int] = None):
    if page is not None:
        base, ext = os.path.splitext(figure_name)
        full_path = f"{base}_{page:02d}{ext}"
    else:
        full_path = figure_name
    if not _is_valid_figure_name(full_path):
        raise ValueError(f"Expected a valid figure name, got {full_path!r}")
    plt.savefig(full_path)


_FEATURES_PER_PAGE = 10
_SUBPLOT_HEIGHT_IN = 3.5


# --- Plotting Functions ---


def plot_label_rates(run, save_figure_name: Optional[str] = None):
    """
    Visualizes the success rate of the 'above_baseline' target segmented
    by Vertical and Tier to identify performance variations.
    """
    df = _get_readable_df(run)
    if "above_baseline" not in df.columns:
        print("Warning: 'above_baseline' target not found in active DataFrame.")
        return

    plt.figure(figsize=run.eda_config["fig_size"])
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=df,
        x="vertical",
        y="above_baseline",
        hue="tier",
        palette=run.eda_config.get("palette", "viridis"),
    )

    ax.set_title("Engagement Success Rate (Above Baseline) by Vertical & Tier")
    ax.set_ylabel("Success Rate (Mean)")
    ax.set_xlabel("Content Vertical")
    plt.legend(title="Channel Tier", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_engagement_distribution(run, save_figure_name: Optional[str] = None):
    """
    Plain histograms (no KDE) for all continuous features, one per row.

    Features are chunked into pages of up to _FEATURES_PER_PAGE so each page
    is a manageable figure rather than one giant grid. Each page is shown and,
    if save_figure_name is provided, saved as root_01.png, root_02.png, etc.

    KDE is intentionally omitted — see plot_kde_distributions() for smoothed
    density curves on the priority features.
    """
    df = _get_readable_df(run)
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["video_id", "above_baseline", "is_short", "is_long"]
    ]

    fig_w = run.eda_config["fig_size"][0]
    pages = [cols[i:i + _FEATURES_PER_PAGE] for i in range(0, len(cols), _FEATURES_PER_PAGE)]
    sns.set_theme(style="ticks")

    for page_idx, page_cols in enumerate(pages, start=1):
        fig, axes = plt.subplots(len(page_cols), 1, figsize=(fig_w, len(page_cols) * _SUBPLOT_HEIGHT_IN))
        if len(page_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, page_cols):
            sns.histplot(df[col], kde=False, bins=40, color="teal", ax=ax)
            ax.set_title(f"Dist: {col}")
            ax.set_xlabel("")
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        if save_figure_name is not None:
            _save_fig(plt, save_figure_name, page=page_idx)
        plt.show()


# Priority features for KDE: the core engagement signals whose smoothed density
# curve adds the most analytical value (growth velocity + raw count anchors).
_KDE_DEFAULT_FEATURES = [
    "view_count_upload",
    "view_count_24h",
    "view_count_7d",
    "like_count_24h",
    "comment_count_24h",
]


def plot_kde_distributions(
    run,
    features: Optional[list] = None,
    save_figure_name: Optional[str] = None,
):
    """
    Histograms with pre-computed scipy KDE curves, one feature per row.

    Features are chunked into pages of up to _FEATURES_PER_PAGE, each saved
    as root_01.png, root_02.png, etc. KDE is computed once on a 200-point grid
    via scipy.stats.gaussian_kde — fast regardless of dataset size. Pass
    `features` to override the default priority list.
    """
    df = _get_readable_df(run)

    candidate_ = features if features is not None else _KDE_DEFAULT_FEATURES
    cols = [c for c in candidate_ if c in df.columns]

    if not cols:
        print("No matching features found in active DataFrame.")
        return

    fig_w = run.eda_config["fig_size"][0]
    pages = [cols[i:i + _FEATURES_PER_PAGE] for i in range(0, len(cols), _FEATURES_PER_PAGE)]
    sns.set_theme(style="ticks")

    for page_idx, page_cols in enumerate(pages, start=1):
        fig, axes = plt.subplots(len(page_cols), 1, figsize=(fig_w, len(page_cols) * _SUBPLOT_HEIGHT_IN))
        if len(page_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, page_cols):
            data = df[col].dropna().values
            ax.hist(data, bins=40, density=True, alpha=0.55, color="teal")
            kde_fn = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde_fn(x_range), color="#e55c00", linewidth=1.8)
            ax.set_title(f"KDE: {col}")
            ax.set_xlabel("")
            ax.set_ylabel("Density")
        plt.tight_layout()
        if save_figure_name is not None:
            _save_fig(plt, save_figure_name, page=page_idx)
        plt.show()


def plot_feature_correlations(
    run, target="above_baseline", save_figure_name: Optional[str] = None
):
    """
    Heatmap of feature correlations. Includes a target leakage check by
    highlighting relationships with the engagement label.

    Figure size is auto-computed as 0.5 in per feature so every cell is
    readable regardless of how many features are present. Numeric annotations
    are added when the matrix is small enough (≤ 25 features) to fit them;
    font size scales down with matrix size.
    """
    df = _get_readable_df(run).select_dtypes(include=[np.number])
    df = df.loc[:, df.var() > 0.01]

    corr = df.corr()
    n = len(corr)

    # Give each cell ~0.5 inches; respect the configured fig_size as a minimum.
    cell_size = 0.5
    fig_w = max(run.eda_config["fig_size"][0], n * cell_size)
    fig_h = max(run.eda_config["fig_size"][1], n * cell_size)

    annot = n <= 25
    annot_kws = {"size": max(6, min(9, 180 // n))} if annot else None

    plt.figure(figsize=(fig_w, fig_h))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        annot_kws=annot_kws,
        fmt=".2f" if annot else "",
        cmap="RdBu_r",
        center=0,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(f"Feature Correlation Matrix (Target: {target})", pad=12)
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_target_correlations(
    run, target="above_baseline", save_figure_name: Optional[str] = None
):
    """
    Horizontal bar chart of per-feature Pearson correlations with the target.
    The focused leakage check: positive correlations in blue, negative in red,
    sorted by absolute magnitude. Low-variance features are filtered out first.
    """
    df = _get_readable_df(run).select_dtypes(include=[np.number])
    df = df.loc[:, df.var() > 0.01]

    if target not in df.columns:
        print(f"Warning: target '{target}' not found in active DataFrame.")
        return

    correlations = (
        df.corr()[target]
        .drop(labels=[target])
        .sort_values(key=abs, ascending=True)
    )

    colors = ["#e74c3c" if v < 0 else "#2980b9" for v in correlations]

    fig_w, fig_h = run.eda_config["fig_size"]
    plt.figure(figsize=(fig_w, max(fig_h, len(correlations) * 0.35)))
    plt.barh(correlations.index, correlations.values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title(f"Feature Correlations with '{target}' (leakage check)")
    plt.xlabel("Pearson r")
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_vertical_segmentation(
    run, feature="view_count_24h", save_figure_name: Optional[str] = None
):
    """
    Compares the distribution of a single metric across content verticals.

    Left subplot: boxplot (median, IQR, and outlier positions).
    Right subplot: density overlay using scipy pre-computed KDE — one
    gaussian_kde call per vertical evaluated at 200 points, making it fast
    even for large datasets.
    """
    df = _get_readable_df(run)

    if feature not in df.columns:
        print(f"Feature '{feature}' not found in active DataFrame.")
        return

    verticals_ = sorted(df["vertical"].dropna().unique())
    palette_ = sns.color_palette("Set2", len(verticals_))

    plt.figure(figsize=run.eda_config["fig_size"])

    # Subplot 1: boxplot — positions and spread
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="vertical", y=feature, palette="Set2", order=verticals_)
    plt.title(f"{feature} by Vertical")
    plt.xticks(rotation=45)

    # Subplot 2: pre-computed KDE per vertical
    ax2 = plt.subplot(1, 2, 2)
    for idx, v in enumerate(verticals_):
        data = df[df["vertical"] == v][feature].dropna().values
        if len(data) < 2:
            continue
        kde_fn = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        density = kde_fn(x_range)
        ax2.fill_between(x_range, density, alpha=0.25, color=palette_[idx])
        ax2.plot(x_range, density, color=palette_[idx], linewidth=1.8, label=v)

    ax2.set_title(f"{feature} Density per Vertical")
    ax2.set_xlabel(feature)
    ax2.set_ylabel("Density")
    ax2.legend(title="Vertical")

    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


# =========================================================================
# Model Comparison (across versions and model types)
# =========================================================================


def load_model_comparison(run) -> pd.DataFrame:
    """Load model comparison DataFrame from GCS and attach to run.

    Stores the result as run.model_comparison_df. All comparison plot
    functions read from this attribute. Returns the DataFrame.
    """
    from utils.snapshot_model import compare_models
    run.model_comparison_df = compare_models()
    return run.model_comparison_df


def _get_filtered_comparison(run, compare_versions, model_type):
    if not hasattr(run, "model_comparison_df") or run.model_comparison_df is None:
        raise RuntimeError(
            "No comparison data loaded. Call eda.load_model_comparison(run) first."
        )
    from utils.snapshot_model import filter_comparison
    return filter_comparison(run.model_comparison_df, compare_versions, model_type)


_MODEL_DISPLAY_NAMES = {
    "VotingClassifier": "Ensemble (VotingClassifier)",
}


def _prep_comparison_df(df):
    """Add sorted categorical version_prefix column and reset index.

    Input df must have version tags as its index (as returned by compare_models()).
    Returns (df_plot, sorted_prefixes).
    """
    from utils.snapshot_model import version_prefix_, get_version_prefixes
    sorted_prefixes = get_version_prefixes(df)
    df = df.reset_index()
    df["version_prefix"] = df["version"].map(version_prefix_)
    df["version_prefix"] = pd.Categorical(
        df["version_prefix"], categories=sorted_prefixes, ordered=True
    )
    df["model_type_label"] = df["model_type"].map(lambda x: _MODEL_DISPLAY_NAMES.get(x, x))
    return df, sorted_prefixes


def _add_eval_divider(ax, sorted_prefixes: list):
    """Draw a dashed vertical line where eval methodology changes at v4.0."""
    for i, vp in enumerate(sorted_prefixes):
        major = int(vp.lstrip("v").split(".")[0])
        if major >= 4 and i > 0:
            ax.axvline(x=i - 0.5, linestyle="--", color="gray", alpha=0.55, linewidth=1.2)
            ax.text(
                i - 0.4, 0.98,
                "← test set | hold-out →",
                fontsize=7, color="gray", va="top",
                transform=ax.get_xaxis_transform(),
            )
            break


def plot_roc_auc(
    run,
    compare_versions="all",
    model_type=None,
    save_figure_name: Optional[str] = None,
):
    """Line chart of ROC-AUC across model snapshot versions, one line per model type."""
    df = _get_filtered_comparison(run, compare_versions, model_type)
    df_plot, sorted_prefixes = _prep_comparison_df(df)

    fig, ax = plt.subplots(figsize=run.eda_config["fig_size"])
    sns.lineplot(
        data=df_plot, x="version_prefix", y="roc_auc",
        hue="model_type_label", style="model_type_label", markers=True, dashes=False,
        palette=run.eda_config.get("palette", "Set2"), ax=ax,
    )
    _add_eval_divider(ax, sorted_prefixes)
    ax.set_title("ROC-AUC by Model Version")
    ax.set_xlabel("Version")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(bottom=max(0, df_plot["roc_auc"].min() - 0.05))
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_roc_curves(
    run,
    save_figure_name: Optional[str] = None,
):
    """
    Plots actual ROC curves (TPR vs. FPR) for every model in run.models,
    evaluated against the locked validation set (run.X_val_unscaled + run.y_val).

    Each model's stored scaler is applied before prediction so results match
    the validated metrics. The AUC is annotated in the legend for each curve.
    A dashed diagonal reference line (random classifier) is included.

    Requires: run.models, run.X_val_unscaled, run.y_val all populated.
    """
    if not getattr(run, "models", None):
        print("No models found in run.models. Train or load models first.")
        return
    if getattr(run, "X_val_unscaled", None) is None or getattr(run, "y_val", None) is None:
        print("Validation set not found. Run DataSplitter + Scaler first.")
        return

    palette_ = sns.color_palette("Set2", len(run.models))
    fig, ax = plt.subplots(figsize=run.eda_config["fig_size"])

    for idx, (model_name, entry) in enumerate(run.models.items()):
        model = entry["model"]
        scaler = entry.get("scaler")
        feature_cols = entry.get("feature_cols", run.X_val_unscaled.columns.tolist())

        X_val = run.X_val_unscaled[feature_cols]
        if scaler is not None:
            X_val = scaler.transform(X_val)

        y_score = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_val)
        )

        fpr, tpr, _ = roc_curve(run.y_val, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            color=palette_[idx],
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Random (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — v5.1 Models (Locked Validation Set)")
    ax.legend(loc="lower right")
    sns.despine()
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()



def plot_accuracy(
    run,
    compare_versions="all",
    model_type=None,
    save_figure_name: Optional[str] = None,
):
    """Line chart of accuracy across model snapshot versions, one line per model type."""
    df = _get_filtered_comparison(run, compare_versions, model_type)
    df_plot, sorted_prefixes = _prep_comparison_df(df)

    fig, ax = plt.subplots(figsize=run.eda_config["fig_size"])
    sns.lineplot(
        data=df_plot, x="version_prefix", y="accuracy",
        hue="model_type_label", style="model_type_label", markers=True, dashes=False,
        palette=run.eda_config.get("palette", "Set2"), ax=ax,
    )
    _add_eval_divider(ax, sorted_prefixes)
    ax.set_title("Accuracy by Model Version")
    ax.set_xlabel("Version")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(bottom=max(0, df_plot["accuracy"].min() - 0.05))
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_precision_recall(
    run,
    compare_versions="all",
    model_type=None,
    save_figure_name: Optional[str] = None,
):
    """2x2 grid: precision and recall for both classes across model versions."""
    df = _get_filtered_comparison(run, compare_versions, model_type)
    df_plot, sorted_prefixes = _prep_comparison_df(df)

    palette_ = run.eda_config.get("palette", "Set2")
    fig_w, fig_h = run.eda_config["fig_size"]
    fig, axes = plt.subplots(2, 2, figsize=(fig_w * 1.3, fig_h * 1.3))
    fig.suptitle("Precision & Recall by Model Version", y=1.01)

    specs_ = [
        (axes[0, 0], "precision_above", "Precision — Above Baseline", False),
        (axes[0, 1], "recall_above",    "Recall — Above Baseline",    True),
        (axes[1, 0], "precision_below", "Precision — Below Baseline", False),
        (axes[1, 1], "recall_below",    "Recall — Below Baseline",    False),
    ]
    for ax, col, title, show_legend in specs_:
        sns.lineplot(
            data=df_plot, x="version_prefix", y=col,
            hue="model_type_label", style="model_type_label", markers=True, dashes=False,
            palette=palette_, ax=ax, legend=show_legend,
        )
        if show_legend:
            ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        _add_eval_divider(ax, sorted_prefixes)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(col.split("_")[0].capitalize())
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_f1(
    run,
    compare_versions="all",
    model_type=None,
    save_figure_name: Optional[str] = None,
):
    """Side-by-side F1 for Above Baseline and Below Baseline across model versions."""
    df = _get_filtered_comparison(run, compare_versions, model_type)
    df_plot, sorted_prefixes = _prep_comparison_df(df)

    palette_ = run.eda_config.get("palette", "Set2")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=run.eda_config["fig_size"])

    for ax, col, title, show_legend in [
        (ax1, "f1_above", "F1 — Above Baseline", False),
        (ax2, "f1_below", "F1 — Below Baseline", True),
    ]:
        sns.lineplot(
            data=df_plot, x="version_prefix", y=col,
            hue="model_type_label", style="model_type_label", markers=True, dashes=False,
            palette=palette_, ax=ax, legend=show_legend,
        )
        if show_legend:
            ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        _add_eval_divider(ax, sorted_prefixes)
        ax.set_title(title)
        ax.set_xlabel("Version")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_data_composition_vs_accuracy(
    run,
    compare_versions="all",
    save_figure_name: Optional[str] = None,
):
    """Correlation heatmap of training data composition vs. accuracy metrics.

    Columns: real_rows, synth_rows, total_rows, accuracy, roc_auc, f1_above, f1_below.
    Treat results as directional signals — correlation is across model snapshots,
    not independent observations.
    """
    df = _get_filtered_comparison(run, compare_versions, None)
    df_corr_ = df.copy()
    df_corr_["total_rows"] = df_corr_["real_rows"].fillna(0) + df_corr_["synth_rows"].fillna(0)

    cols_ = ["real_rows", "synth_rows", "total_rows", "accuracy", "roc_auc", "f1_above", "f1_below"]
    corr_ = df_corr_[cols_].corr()

    fig_w, fig_h = run.eda_config["fig_size"]
    plt.figure(figsize=(max(fig_w, 8), max(fig_h, 7)))
    sns.heatmap(
        corr_, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.4, cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Training Data Composition vs. Accuracy Metrics (n={len(df_corr_)} snapshots)")
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_top_features_over_time(
    run,
    model_type,
    compare_versions="all",
    top_n: int = 10,
    save_figure_name: Optional[str] = None,
):
    """Heatmap of top feature importances for a single model type across snapshot versions.

    model_type must be a ModelType enum value. top_n features are selected by mean
    absolute importance across versions, so coefficients (LR) and tree importances
    are handled consistently.
    """
    from utils.snapshot_model import load_top_features_over_time, get_version_prefixes

    df_features_ = load_top_features_over_time(model_type, compare_versions)
    if df_features_.empty:
        return

    top_feats_ = (
        df_features_.groupby("feature")["importance"]
        .apply(lambda x: x.abs().mean())
        .nlargest(top_n)
        .index
    )
    df_top_ = df_features_[df_features_["feature"].isin(top_feats_)]

    pivot_ = df_top_.pivot_table(
        index="feature", columns="version_prefix", values="importance", aggfunc="mean"
    )
    all_prefixes_ = get_version_prefixes(run.model_comparison_df)
    sorted_cols_ = [c for c in all_prefixes_ if c in pivot_.columns]
    pivot_ = pivot_[sorted_cols_]

    fig_w, fig_h = run.eda_config["fig_size"]
    plt.figure(figsize=(max(fig_w, len(sorted_cols_) * 1.4), max(fig_h, top_n * 0.55)))
    sns.heatmap(
        pivot_, annot=True, fmt=".4f", cmap="RdBu_r", center=0,
        linewidths=0.3, cbar_kws={"shrink": 0.7},
    )
    display_name_ = _MODEL_DISPLAY_NAMES.get(model_type.value, model_type.value)
    plt.title(f"Top {top_n} Features Over Time — {display_name_}")
    plt.xlabel("Version")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_feature_importance(
    run,
    model_name: str = "xgb",
    top_n: int = 20,
    save_figure_name: Optional[str] = None,
):
    """
    Horizontal bar chart of the top N feature importances (or coefficients)
    for a single model in run.models. Positive importances in blue, negative
    coefficients (LR) in red.

    Args:
        model_name: Key in run.models (e.g. 'xgb', 'rf', 'lr_l1', 'ensemble').
        top_n:      Number of features to show, ranked by absolute value.
    """
    if model_name not in run.models:
        print(f"Model '{model_name}' not found in run.models. "
              f"Available: {list(run.models.keys())}")
        return

    entry = run.models[model_name]
    model = entry["model"]
    feature_cols = entry.get("feature_cols", [])

    # Extract importances / coefficients depending on model type
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Feature Importance (impurity reduction)"
        is_signed = False
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
        xlabel = "Coefficient"
        is_signed = True
    else:
        # VotingClassifier — aggregate from sub-estimators if possible
        importances_list = []
        for _, estimator in model.estimators:
            if hasattr(estimator, "feature_importances_"):
                importances_list.append(estimator.feature_importances_)
        if not importances_list:
            print(f"Model '{model_name}' does not expose feature importances.")
            return
        importances = np.mean(importances_list, axis=0)
        xlabel = "Mean Feature Importance (sub-estimators)"
        is_signed = False

    if len(feature_cols) != len(importances):
        print(f"Feature column count mismatch: {len(feature_cols)} cols vs "
              f"{len(importances)} importances.")
        return

    df_imp = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .reindex(pd.Series(importances).abs().nlargest(top_n).index)
        .sort_values("importance", ascending=True)
    )

    colors = (
        ["#e74c3c" if v < 0 else "#2980b9" for v in df_imp["importance"]]
        if is_signed
        else ["#2980b9"] * len(df_imp)
    )

    fig_w, fig_h = run.eda_config["fig_size"]
    plt.figure(figsize=(fig_w, max(fig_h, top_n * 0.4)))
    plt.barh(df_imp["feature"], df_imp["importance"], color=colors)
    if is_signed:
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel(xlabel)
    plt.title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    if save_figure_name is not None:
        _save_fig(plt, save_figure_name)
    plt.show()


def plot_is_short_engagement(
    run,
    feature: str = "like_rate_24h",
    save_figure_name: Optional[str] = None,
):
    """
    Compares the distribution of an engagement metric between YouTube Shorts
    (is_short=1) and standard videos (is_short=0), split by vertical.

    Left subplot: violin plot of `feature` by content type (Short vs. Standard),
    coloured by vertical.
    Right subplot: median `feature` per vertical × content type as a grouped
    bar chart, with error bars showing 95% CI via bootstrapping.

    Args:
        feature: Engagement metric to compare. Defaults to 'like_rate_24h'.
                 Other useful options: 'view_velocity_ratio', 'like_rate_upload'.
    """
    df = _get_readable_df(run)

    required = [feature, "is_short", "vertical"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}. Set a post-engineering DataFrame first.")
        return

    df_plot = df[required].copy()
    df_plot["Content Type"] = df_plot["is_short"].map({1: "Short (≤60s)", 0: "Standard"})

    palette_ = run.eda_config.get("palette", "Set2")
    fig_w, fig_h = run.eda_config["fig_size"]

    # Figure 1: side-by-side violins — Standard (left) and Short (right)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w * 1.3, fig_h), sharey=False)
    for ax, label, title_suffix in [
        (ax1, "Standard",    "Standard Videos"),
        (ax2, "Short (≤60s)", "YouTube Shorts (≤60s)"),
    ]:
        subset_ = df_plot[df_plot["Content Type"] == label]
        sns.violinplot(
            data=subset_,
            x="vertical",
            y=feature,
            palette=palette_,
            inner="quartile",
            ax=ax,
            cut=0,
        )
        ax.set_title(f"{feature}\n{title_suffix}")
        ax.set_xlabel("Vertical")
        ax.set_ylabel(feature)
        ax.tick_params(axis="x", rotation=30)

    fig1.suptitle(
        f"Engagement Rate Distribution: YouTube Shorts vs. Standard Videos",
        y=1.02,
    )
    sns.despine()
    plt.tight_layout()
    if save_figure_name is not None:
        base_, ext_ = os.path.splitext(save_figure_name)
        _save_fig(plt, f"{base_}_violin{ext_}")
    plt.show()

    # Figure 2: median with 95% CI grouped bar chart
    fig2, ax3 = plt.subplots(figsize=(fig_w, fig_h))
    sns.barplot(
        data=df_plot,
        x="vertical",
        y=feature,
        hue="Content Type",
        palette=["#e55c00", "#2980b9"],
        estimator=np.median,
        errorbar=("ci", 95),
        ax=ax3,
    )
    ax3.set_title(f"Median {feature} by Vertical × Content Type")
    ax3.set_xlabel("Vertical")
    ax3.set_ylabel(f"Median {feature}")
    ax3.legend(title="Content Type")
    sns.despine()
    plt.tight_layout()
    if save_figure_name is not None:
        base_, ext_ = os.path.splitext(save_figure_name)
        _save_fig(plt, f"{base_}_bar{ext_}")
    plt.show()