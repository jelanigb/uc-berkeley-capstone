"""
Data cleanup and merging for YouTube engagement prediction.
Pivots per-poll snapshots into one-row-per-video format
and joins channel baseline medians.
"""

import pandas as pd
import numpy as np


# --- Columns that are the same across all polls for a given video ---
STATIC_COLS = [
    'video_id', 'channel_id', 'channel_handle',
    'title', 'description', 'tags',
    'duration_seconds', 'category_id', 'category_name',
    'published_at', 'vertical', 'tier',
    'contains_synthetic_media',
]

# --- Columns that change at each poll interval ---
METRIC_COLS = [
    'view_count', 'like_count', 'comment_count', 'subscriber_count',
]

# --- Thumbnail features (captured at poll time but stable) ---
THUMBNAIL_COLS = [
    'face_count', 'brightness', 'colorfulness',
]


def pivot_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot video_snapshots from long format (up to 3 rows per video)
    to wide format (1 row per video) with columns like:
      views_upload, views_24h, views_7d, likes_upload, etc.

    Only includes videos that have all three poll labels.
    """
    # --- Filter to complete triplets only ---
    poll_counts = df.groupby('video_id')['poll_label'].nunique()
    complete_ids = poll_counts[poll_counts == 3].index
    df_complete = df[df['video_id'].isin(complete_ids)].copy()

    dropped = df['video_id'].nunique() - len(complete_ids)
    print(f"  Videos with all 3 polls: {len(complete_ids)} "
          f"(dropped {dropped} incomplete)")

    # --- Static columns: take from the upload row ---
    df_upload_static = df_complete[
        df_complete['poll_label'] == 'upload'
    ][STATIC_COLS + THUMBNAIL_COLS].copy()

    # --- Pivot the metric columns per poll label ---
    label_suffixes = {'upload': 'upload', '24h': '24h', '7d': '7d'}

    pivoted_parts = []
    for poll_label, suffix in label_suffixes.items():
        df_poll = df_complete[df_complete['poll_label'] == poll_label].copy()

        rename_map = {col: f"{col}_{suffix}" for col in METRIC_COLS}
        rename_map['hours_since_publish'] = f"hours_since_publish_{suffix}"
        rename_map['poll_timestamp'] = f"poll_timestamp_{suffix}"

        cols_to_keep = ['video_id'] + list(rename_map.keys())
        df_poll = df_poll[cols_to_keep].rename(columns=rename_map)
        pivoted_parts.append(df_poll)

    # --- Merge everything ---
    df_wide = df_upload_static.copy()
    for part in pivoted_parts:
        df_wide = df_wide.merge(part, on='video_id', how='inner')

    print(f"  Pivoted shape: {df_wide.shape}")
    return df_wide


def join_baselines(
    df_wide: pd.DataFrame, df_medians: pd.DataFrame
) -> pd.DataFrame:
    """
    Join channel baseline medians onto the pivoted video table.
    Baseline columns are prefixed with 'baseline_' to avoid collisions.
    """
    rename_map = {
        col: f"baseline_{col}"
        for col in df_medians.columns
        if col != 'channel_id'
    }
    df_medians_prefixed = df_medians.rename(columns=rename_map)

    df_joined = df_wide.merge(df_medians_prefixed, on='channel_id', how='left')

    baseline_cols = [c for c in df_joined.columns if c.startswith('baseline_')]
    if baseline_cols:
        unmatched = df_joined[df_joined[baseline_cols[0]].isna()].shape[0]
        print(f"  Baseline join: {len(df_joined) - unmatched}/{len(df_joined)} "
              f"videos matched a channel median")

    return df_joined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural cleanup: whitespace trimming, type fixes, etc.
    Expand as needed.
    """
    df = df.copy()

    # --- Whitespace cleanup on text fields ---
    for col in ['title', 'description', 'channel_handle']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --- Ensure tags is a list (Parquet may deserialize differently) ---
    df['tags'] = df['tags'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # --- Clamp any negative metric values ---
    metric_patterns = ['view_count_', 'like_count_', 'comment_count_',
                       'subscriber_count_', 'duration_seconds']
    for col in df.columns:
        if any(col.startswith(p) for p in metric_patterns):
            df[col] = df[col].clip(lower=0)

    print(f"  Cleaned: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def build_clean_dataset(
    df_snapshots: pd.DataFrame,
    df_medians: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full cleanup pipeline: pivot → join baselines → clean.

    Usage:
        df_clean = build_clean_dataset(df_videos, df_medians)
    """
    print("=" * 60)
    print("Building clean dataset")
    print("=" * 60)

    print("\n[1/3] Pivoting snapshots...")
    df_wide = pivot_snapshots(df_snapshots)

    print("\n[2/3] Joining baseline medians...")
    df_joined = join_baselines(df_wide, df_medians)

    print("\n[3/3] Cleaning data...")
    df_clean = clean_data(df_joined)

    print(f"\n{'=' * 60}")
    print(f"Clean dataset: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    print(f"{'=' * 60}")

    return df_clean