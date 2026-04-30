import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

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
    IMAGE_PATH_ = "images/eda/"
    IMAGE_EXT_ = ".png"
    suffix = f"_{page:02d}" if page is not None else ""
    full_path = f"{IMAGE_PATH_}{figure_name}{suffix}{IMAGE_EXT_}"
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
