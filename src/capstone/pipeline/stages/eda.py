import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Configuration & State Management ---

def set_df(run, df_name: str):
    """
    Sets the active_eda_df on the PipelineRun object by attribute name.
    Allows switching between raw, clean, and engineered data contexts.
    """
    target_df = getattr(run, df_name)
    if target_df is None:
        raise ValueError(f"DataFrame '{df_name}' is not populated in PipelineRun.")
    run.active_eda_df = target_df

def set_fig_size(run, width: int, height: int):
    """Updates the persistent plotting configuration in the PipelineRun[cite: 2]."""
    run.eda_config["fig_size"] = (width, height)

def get_plt(run):
    """Returns the matplotlib.pyplot object for custom manual adjustments."""
    return plt

def _get_readable_df(run):
    """
    Internal Helper: Joins categorical labels (Vertical/Tier) back to 
    engineered data if they are missing, ensuring clean legends[cite: 1, 3].
    """
    if run.active_eda_df is None:
        raise RuntimeError("No active EDA DataFrame set. Call eda.set_df(run, 'df_name') first.")
    
    df = run.active_eda_df.copy()
    
    # If working with engineered data, categorical labels are often encoded or dropped.
    # Join them back from df_clean for visualization purposes[cite: 3].
    if "vertical" not in df.columns and run.df_clean is not None:
        labels = run.df_clean[['video_id', 'vertical', 'tier']]
        df = df.merge(labels, on='video_id', how='left')
    return df

# --- Plotting Functions ---

def plot_label_rates(run):
    """
    Visualizes the success rate of the 'above_baseline' target segmented 
    by Vertical and Tier to identify performance variations[cite: 1].
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
        palette=run.eda_config.get("palette", "viridis")
    )
    
    ax.set_title("Engagement Success Rate (Above Baseline) by Vertical & Tier")
    ax.set_ylabel("Success Rate (Mean)")
    ax.set_xlabel("Content Vertical")
    plt.legend(title="Channel Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_engagement_distribution(run):
    """
    Generates a grid of histograms with KDE overlays for all continuous 
    variables to analyze feature spread and skewness[cite: 1].
    """
    df = _get_readable_df(run)
    # Identify numeric columns, excluding ID and binary targets
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c not in ['video_id', 'above_baseline', 'is_short', 'is_long']]

    num_cols = 3
    num_rows = (len(cols) + num_cols - 1) // num_cols
    
    plt.figure(figsize=(run.eda_config["fig_size"][0], 4 * num_rows))
    sns.set_theme(style="ticks")

    for i, col in enumerate(cols):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(df[col], kde=True, color='teal')
        plt.title(f"Dist: {col}")
        plt.xlabel("")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def plot_feature_correlations(run, target='above_baseline'):
    """
    Heatmap of feature correlations. Includes a target leakage check by 
    highlighting relationships with the engagement label[cite: 1].
    """
    df = _get_readable_df(run).select_dtypes(include=[np.number])
    
    # Filter out very low variance features to improve legibility
    df = df.loc[:, df.var() > 0.01]
    
    corr = df.corr()
    
    plt.figure(figsize=run.eda_config["fig_size"])
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=False, 
        cmap='RdBu_r', 
        center=0,
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    
    plt.title(f"Feature Correlation Matrix (Target: {target})")
    plt.show()

def plot_vertical_segmentation(run, feature='view_count_24h'):
    """
    Uses subplots to compare the distributions of a specific metric 
    across different content verticals using boxplots[cite: 1].
    """
    df = _get_readable_df(run)
    
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in active DataFrame.")
        return

    plt.figure(figsize=run.eda_config["fig_size"])
    
    # Subplot 1: Distribution comparison
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='vertical', y=feature, palette='Set2')
    plt.title(f"{feature} by Vertical")
    plt.xticks(rotation=45)

    # Subplot 2: Density comparison
    plt.subplot(1, 2, 2)
    for v in df['vertical'].unique():
        sns.kdeplot(df[df['vertical'] == v][feature], label=v, fill=True)
    plt.title(f"{feature} Density per Vertical")
    plt.legend(title="Vertical")

    plt.tight_layout()
    plt.show()