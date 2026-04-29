# EDA Module design
**Author:** Jelani Gould-Bailey
**LLM Co-Author:** Gemini Thinking
**Creation Date:** April 29, 2026  

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
    
    # Persistent plotting config (e.g., {'fig_size': (12, 6), 'style': 'whitegrid'})
    eda_config: dict = field(default_factory=lambda: {
        "fig_size": (10, 6),
        "palette": "viridis",
    })

    # ... [Existing methods: assert_ready_for, summary] ...
```

---

## 2. The `eda.py` Module Design

The module is designed as a suite of functional "tools." It abstracts the complex joining logic required to get human-readable labels (Vertical/Tier) back onto engineered data.

### `pipeline/stages/eda.py`

#### **Configuration & State Management**
*   **`set_df(run, df_name)`**: Sets `run.active_eda_df`, allowing for overrides at any point in the 
eda execution flow (for example switching from a pre- feature engineering DF used in initial analysis,
to a post-engineering DF used to interpret results).
*   **`set_fig_size(run, width, height)`**: Updates the matplotlib `figSize()` for the active plot.
*   **`_get_readable_df(run)`**: *Internal Helper.* If the active DF is engineered data, this function performs a left-join back to `df_clean` on `video_id` to restore categorical "Vertical" and "Tier" labels for legends.[cite: 2, 3]
*   **`get_plt(run)`**: Returns the active `plt` object, e.g. if additional Matplotlib interactions are needed that 
aren't covered by existing eda module functionality.

#### **Plotting Functions**
*   **`plot_label_rates(run)`**: 
    *   Calculates the success rate of the `above_baseline` target.
    *   Produces a grouped bar chart segmented by **Vertical** and **Tier**.[cite: 1]
*   **`plot_engagement_distribution(run)`**: 
    *   Generates a grid of histograms with KDE overlays.
    *   Automatically identifies continuous variables in the active DF.[cite: 1]
*   **`plot_feature_correlations(run, target='above_baseline')`**: 
    *   **Leakage Check:** Specifically highlights correlations between features and the target.
    *   Filters out low-variance features to keep the heatmap legible.[cite: 1]
*   **`plot_vertical_segmentation(run, feature='view_count_24h')`**: 
    *   Utilizes subplots to compare distributions (Boxplots/Violin plots) of a specific metric across different verticals.[cite: 1]

---

## 3. Example Implementation (Code Snippet)

This demonstrates how the internal logic handles your request for granular functions and automatic label restoration.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_df(run, df_name: str):
    """Context switcher for EDA."""
    target_df = getattr(run, df_name)
    if target_df is None:
        raise ValueError(f"DataFrame '{df_name}' is not populated in PipelineRun.")
    run.active_eda_df = target_df

def plot_label_rates(run):
    """Bar chart of target success rates segmented by Tier and Vertical."""
    # Restore original labels for the legend
    df = _get_readable_df(run) 
    
    plt.figure(figsize=run.eda_config["fig_size"])
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x="vertical", 
        y="above_baseline", 
        hue="tier",
        palette=run.eda_config["palette"]
    )
    
    ax.set_title("Engagement Success Rate (Above Baseline) by Vertical & Tier")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Content Vertical")
    plt.legend(title="Channel Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def _get_readable_df(run):
    """Joins categorical labels back to engineered data if necessary."""
    df = run.active_eda_df.copy()
    # Check if we need to join Vertical/Tier back from df_clean
    if "vertical" not in df.columns and run.df_clean is not None:
        labels = run.df_clean[['video_id', 'vertical', 'tier']]
        df = df.merge(labels, on='video_id', how='left')
    return df
```

---

## 4. Notebook Workflow Comparison

With this design, your notebook logic remains extremely clean while maintaining full control over the EDA state.

**Old Way (Cluttered):**
```python
# Join labels manually
df_plt = run.df_engineered.merge(run.df_clean[['video_id', 'vertical']], on='video_id')
plt.figure(figsize=(15, 5))
sns.barplot(data=df_plt, x='vertical', y='above_baseline')
```

**New Way (Abstracted):**
```python
# Context: Pre-engineered wide data
eda.set_df(run, "df_clean")
eda.plot_engagement_distribution(run)

# Context: Post-engineered (Automatically joins labels for the legend)
eda.set_df(run, "df_engineered")
eda.set_fig_size(run, 14, 7)
eda.plot_label_rates(run)
eda.plot_feature_correlations(run) # Includes leakage check
```

---