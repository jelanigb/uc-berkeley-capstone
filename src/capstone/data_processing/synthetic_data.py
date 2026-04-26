"""
Synthetic data generation for YouTube engagement prediction.
Uses SDV's GaussianCopulaSynthesizer to generate realistic
synthetic rows from the model-ready table.

Synthetic rows are assigned to real channels proportionally,
and baseline medians are copied from the real channel data
(not synthesized) to ensure realistic target labels.
"""

import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


# Columns to synthesize — numeric metrics and categorical codes.
# Excludes: free text, IDs, timestamps, target variable, AND baselines
# (baselines are copied from real channels, not synthesized).
SYNTH_COLUMNS = [
    # Poll metrics
    'view_count_upload', 'like_count_upload', 'comment_count_upload', 'subscriber_count_upload',
    'view_count_24h', 'like_count_24h', 'comment_count_24h', 'subscriber_count_24h',
    'view_count_7d', 'like_count_7d', 'comment_count_7d', 'subscriber_count_7d',
    'hours_since_publish_upload', 'hours_since_publish_24h', 'hours_since_publish_7d',
    # Thumbnail
    'face_count', 'brightness', 'colorfulness',
    # Duration
    'duration_seconds',
    # Categorical groupings (real values, not engineered)
    'vertical', 'tier', 'category_id',
]

# Baseline columns that will be copied from real channels (not synthesized)
BASELINE_COLS = [
    'baseline_video_count',
    'baseline_median_views', 'baseline_median_likes',
    'baseline_median_comments', 'baseline_median_engagement_rate',
    'baseline_channel_handle',
]

# Columns that should never go negative
NON_NEGATIVE_COLS = [
    'view_count_upload', 'like_count_upload', 'comment_count_upload', 'subscriber_count_upload',
    'view_count_24h', 'like_count_24h', 'comment_count_24h', 'subscriber_count_24h',
    'view_count_7d', 'like_count_7d', 'comment_count_7d', 'subscriber_count_7d',
    'hours_since_publish_upload', 'hours_since_publish_24h', 'hours_since_publish_7d',
    'face_count', 'brightness', 'colorfulness', 'duration_seconds',
]

# Columns that should be rounded to integers
INTEGER_COLS = [
    'view_count_upload', 'like_count_upload', 'comment_count_upload', 'subscriber_count_upload',
    'view_count_24h', 'like_count_24h', 'comment_count_24h', 'subscriber_count_24h',
    'view_count_7d', 'like_count_7d', 'comment_count_7d', 'subscriber_count_7d',
    'face_count', 'duration_seconds',
]


def _prepare_for_sdv(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a subset of columns for SDV, fixing types."""
    df_synth = df[SYNTH_COLUMNS].copy()

    # Convert nullable Int64 to regular int
    for col in df_synth.select_dtypes(include=['Int64']).columns:
        df_synth[col] = df_synth[col].fillna(0).astype('int64')

    # Convert nullable Float64
    for col in df_synth.select_dtypes(include=['Float64']).columns:
        df_synth[col] = df_synth[col].fillna(0.0).astype('float64')

    return df_synth


def _assign_real_channels(df_synth: pd.DataFrame, df_real: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Assign synthetic rows to real channel IDs, weighted proportionally
    by each channel's video count in the real data. Then copy that
    channel's real baseline medians onto the synthetic row.
    """
    # Build channel weights from real data
    channel_counts = df_real['channel_id'].value_counts()
    channel_weights = channel_counts / channel_counts.sum()

    # Sample channel IDs proportionally
    sampled_channels = rng.choice(
        channel_weights.index,
        size=len(df_synth),
        p=channel_weights.values,
        replace=True,
    )
    df_synth['channel_id'] = sampled_channels

    # Build a lookup table: one row per channel with baseline columns
    baseline_lookup_cols = ['channel_id', 'channel_handle'] + [
        c for c in BASELINE_COLS if c in df_real.columns
    ]
    df_channel_lookup = (
        df_real[baseline_lookup_cols]
        .drop_duplicates(subset='channel_id')
        .set_index('channel_id')
    )

    # Copy real baselines onto synthetic rows
    for col in df_channel_lookup.columns:
        df_synth[col] = df_synth['channel_id'].map(df_channel_lookup[col])

    # Also copy vertical and tier from the real channel
    # (overrides SDV's synthesized vertical/tier to keep channel consistency)
    channel_vertical = (
        df_real[['channel_id', 'vertical', 'tier']]
        .drop_duplicates(subset='channel_id')
        .set_index('channel_id')
    )
    df_synth['vertical'] = df_synth['channel_id'].map(channel_vertical['vertical'])
    df_synth['tier'] = df_synth['channel_id'].map(channel_vertical['tier'])

    matched = df_synth['baseline_median_views'].notna().sum() if 'baseline_median_views' in df_synth.columns else 0
    print(f"  Assigned {len(df_synth)} synthetic rows to {df_synth['channel_id'].nunique()} real channels")
    print(f"  Baseline match: {matched}/{len(df_synth)}")

    return df_synth


def _postprocess_synthetic(df_synth: pd.DataFrame) -> pd.DataFrame:
    """Clean up synthetic rows: clamp negatives, round integers, enforce monotonicity."""

    # Clamp non-negative columns
    for col in NON_NEGATIVE_COLS:
        if col in df_synth.columns:
            df_synth[col] = df_synth[col].clip(lower=0)

    # Round integer columns
    for col in INTEGER_COLS:
        if col in df_synth.columns:
            df_synth[col] = df_synth[col].round().astype(int)

    # Ensure monotonic growth: upload <= 24h <= 7d for metrics
    for metric in ['view_count', 'like_count', 'comment_count']:
        col_u = f"{metric}_upload"
        col_24 = f"{metric}_24h"
        col_7d = f"{metric}_7d"
        if all(c in df_synth.columns for c in [col_u, col_24, col_7d]):
            df_synth[col_24] = df_synth[[col_u, col_24]].max(axis=1)
            df_synth[col_7d] = df_synth[[col_24, col_7d]].max(axis=1)

    # Ensure hours_since_publish ordering
    if all(c in df_synth.columns for c in [
        'hours_since_publish_upload', 'hours_since_publish_24h', 'hours_since_publish_7d'
    ]):
        df_synth['hours_since_publish_24h'] = df_synth[
            ['hours_since_publish_upload', 'hours_since_publish_24h']
        ].max(axis=1)
        df_synth['hours_since_publish_7d'] = df_synth[
            ['hours_since_publish_24h', 'hours_since_publish_7d']
        ].max(axis=1)

    # Add synthetic IDs and flag
    df_synth['video_id'] = [f'syn-{i:06d}' for i in range(len(df_synth))]
    df_synth['contains_synthetic_data'] = True

    # Placeholder text fields
    df_synth['title'] = '[Synthetic]'
    df_synth['description'] = '[Synthetic]'
    df_synth['tags'] = [[] for _ in range(len(df_synth))]
    df_synth['category_name'] = 'Synthetic'
    df_synth['published_at'] = pd.NaT
    df_synth['contains_synthetic_media'] = None

    # Placeholder poll timestamps
    df_synth['poll_timestamp_upload'] = pd.NaT
    df_synth['poll_timestamp_24h'] = pd.NaT
    df_synth['poll_timestamp_7d'] = pd.NaT

    return df_synth


def _recompute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute engineered features on synthetic rows.
    Must match the logic in feature_engineering.py.
    """
    # --- Target variable ---
    df['engagement_7d'] = np.where(
        df['view_count_7d'] > 0,
        (df['like_count_7d'] + df['comment_count_7d']) / df['view_count_7d'],
        0.0
    )
    baseline_comments = df.get(
        'baseline_median_comments',
        pd.Series(0, index=df.index)
    )
    df['baseline_engagement'] = np.where(
        df['baseline_median_views'] > 0,
        (df['baseline_median_likes'] + baseline_comments) / df['baseline_median_views'],
        0.0
    )
    df['above_baseline'] = (df['engagement_7d'] > df['baseline_engagement']).astype(int)

    # --- Velocity ---
    hours_elapsed = (
        df['hours_since_publish_24h'] - df['hours_since_publish_upload']
    ).clip(lower=0.1)
    for metric in ['view_count', 'like_count', 'comment_count', 'subscriber_count']:
        col_u = f"{metric}_upload"
        col_24 = f"{metric}_24h"
        if col_u in df.columns and col_24 in df.columns:
            df[f"{metric}_velocity_24h"] = (df[col_24] - df[col_u]) / hours_elapsed

    # --- Subscriber-normalized ---
    sub_count = df['subscriber_count_upload'].clip(lower=1)
    for suffix in ['upload', '24h', '7d']:
        df[f"views_per_sub_{suffix}"] = df[f"view_count_{suffix}"] / sub_count
        df[f"likes_per_sub_{suffix}"] = df[f"like_count_{suffix}"] / sub_count
        df[f"comments_per_sub_{suffix}"] = df[f"comment_count_{suffix}"] / sub_count

    # --- Text features (placeholder values for synthetic) ---
    df['title_length'] = 0
    df['title_word_count'] = 0
    df['title_category'] = 0  # neutral
    df['desc_length'] = 0
    df['desc_link_count'] = 0
    df['desc_hashtag_count'] = 0
    df['desc_category'] = 0  # neutral
    df['tag_count'] = 0
    df['has_tags'] = 0

    # --- Temporal features (not meaningful for synthetic) ---
    df['publish_hour'] = 0
    df['publish_dayofweek'] = 0
    df['publish_is_weekend'] = 0

    # --- Duration features ---
    df['is_short'] = (df['duration_seconds'] <= 60).astype(int)
    df['duration_minutes'] = df['duration_seconds'] / 60

    return df


def generate_synthetic_data(
    df_model: pd.DataFrame,
    num_rows: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic rows modeled on the feature-engineered model table.

    - Synthesizes numeric/categorical columns (NOT baselines)
    - Assigns synthetic rows to real channels proportionally
    - Copies real baseline medians from assigned channels
    - Enforces monotonic metric growth (upload <= 24h <= 7d)
    - Recomputes target and derived features from synthesized values
    - Marks all rows with contains_synthetic_data = True

    Args:
        df_model: Feature-engineered modeling table (from engineer_features)
        num_rows: Number of synthetic rows to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame of synthetic rows with same columns as df_model
    """
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print(f"Generating {num_rows} synthetic rows")
    print("=" * 60)

    # Prepare data for SDV
    print("\n[1/5] Preparing data for synthesis...")
    df_for_sdv = _prepare_for_sdv(df_model)
    print(f"  Input shape: {df_for_sdv.shape}")

    # Fit synthesizer
    print("\n[2/5] Fitting GaussianCopulaSynthesizer...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_for_sdv)

    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
    )
    synthesizer.fit(df_for_sdv)
    print(f"  Fit complete")

    # Generate
    print("\n[3/5] Sampling synthetic rows...")
    np.random.seed(seed)
    df_synth = synthesizer.sample(num_rows=num_rows)
    print(f"  Generated {len(df_synth)} rows")

    # Assign real channels and copy baselines
    print("\n[4/5] Assigning real channels and baselines...")
    df_synth = _assign_real_channels(df_synth, df_model, rng)

    # Postprocess and recompute features
    print("\n[5/5] Postprocessing and recomputing features...")
    df_synth = _postprocess_synthetic(df_synth)
    df_synth = _recompute_engineered_features(df_synth)

    # Align columns with real data
    for col in df_model.columns:
        if col not in df_synth.columns:
            df_synth[col] = None
    # Ensure contains_synthetic_data is included
    output_cols = [c for c in df_model.columns if c in df_synth.columns]
    if 'contains_synthetic_data' not in output_cols:
        output_cols.append('contains_synthetic_data')
    df_synth = df_synth[output_cols]

    target_dist = df_synth['above_baseline'].value_counts().to_dict()
    print(f"\n{'=' * 60}")
    print(f"Synthetic data: {len(df_synth)} rows")
    print(f"Assigned to {df_synth['channel_id'].nunique()} real channels")
    print(f"Target balance: {target_dist}")
    print(f"{'=' * 60}")

    return df_synth


def combine_real_and_synthetic(
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine real and synthetic DataFrames with explicit flagging.

    Args:
        df_real: Real model-ready DataFrame
        df_synthetic: Synthetic DataFrame from generate_synthetic_data

    Returns:
        Combined DataFrame with contains_synthetic_data column
        (True for synthetic, False for real — no NULLs)
    """
    df_real = df_real.copy()
    df_real['contains_synthetic_data'] = False

    # Align columns
    all_cols = list(dict.fromkeys(
        df_real.columns.tolist() + df_synthetic.columns.tolist()
    ))
    for col in all_cols:
        if col not in df_real.columns:
            df_real[col] = None
        if col not in df_synthetic.columns:
            df_synthetic[col] = None

    df_combined = pd.concat(
        [df_real[all_cols], df_synthetic[all_cols]],
        ignore_index=True,
    )

    real_count = len(df_real)
    synth_count = len(df_synthetic)
    print(f"Combined dataset: {len(df_combined)} rows "
          f"({real_count} real + {synth_count} synthetic, "
          f"{synth_count / len(df_combined) * 100:.0f}% synthetic)")

    return df_combined