"""
Feature engineering for YouTube engagement prediction.
Takes a clean, merged DataFrame and adds target variable,
velocity metrics, and structural features.
"""

import pandas as pd
import numpy as np


# =========================================================================
# Category dictionaries for title and description pattern encoding
# =========================================================================

TITLE_CATEGORIES = {
    'question':     1,   # contains ?
    'exclamation':  2,   # contains !
    'listicle':     3,   # starts with or contains "Top N", "N things", etc.
    'how_to':       4,   # starts with "How to" or "How To"
    'emoji_heavy':  5,   # 3+ emoji
    'all_caps':     6,   # >50% uppercase letters
    'clickbait':    7,   # combines ! + emoji + caps
    'neutral':      0,   # none of the above
}

DESC_CATEGORIES = {
    'has_timestamps':   1,   # contains MM:SS patterns (chapter markers)
    'has_links':        2,   # contains URLs
    'link_heavy':       3,   # 5+ URLs
    'has_hashtags':     4,   # contains #hashtags
    'minimal':          5,   # <50 characters
    'long_form':        6,   # >2000 characters
    'neutral':          0,   # none of the above
}


# =========================================================================
# Target variable
# =========================================================================

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary target: is this video's 7d engagement
    above its channel's baseline median?

    Engagement = (likes + comments) / views at the 7d mark.
    """
    df = df.copy()

    # Video's 7d engagement rate
    df['engagement_7d'] = np.where(
        df['view_count_7d'] > 0,
        (df['like_count_7d'] + df['comment_count_7d']) / df['view_count_7d'],
        0.0
    )

    # Channel's baseline median engagement rate
    # Adjust column names to match your channel_baseline_medians view
    if 'baseline_median_likes' in df.columns and 'baseline_median_views' in df.columns:
        baseline_comments = df.get(
            'baseline_median_comments',
            pd.Series(0, index=df.index)
        )
        df['baseline_engagement'] = np.where(
            df['baseline_median_views'] > 0,
            (df['baseline_median_likes'] + baseline_comments)
            / df['baseline_median_views'],
            0.0
        )
    else:
        available = [c for c in df.columns if 'baseline' in c]
        print(f"  WARNING: baseline_median_views/likes not found.")
        print(f"  Available baseline columns: {available}")
        df['baseline_engagement'] = 0.0

    # Binary target
    df['above_baseline'] = (
        df['engagement_7d'] > df['baseline_engagement']
    ).astype(int)

    pos_rate = df['above_baseline'].mean()
    print(f"  Target distribution: {pos_rate:.1%} above baseline, "
          f"{1 - pos_rate:.1%} below")

    return df


# =========================================================================
# Velocity features
# =========================================================================

VELOCITY_METRICS = [
    'view_count', 'like_count', 'comment_count', 'subscriber_count',
]


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity, acceleration, and normalized velocity features.

    Sections:
      1. Upload→24h velocity for all metrics (core rate-of-change)
      2. Publish→upload velocity for views/likes (early momentum signal)
      3. Subscriber-normalized velocity (cross-tier comparability)
      4. Velocity ratio vs baseline median (velocity in channel context)
      5. Velocity acceleration (ratio of 24h velocity to upload velocity)
    """
    df = df.copy()

    sub_count = df['subscriber_count_upload'].clip(lower=1)
    baseline_views = df['baseline_median_views'].clip(lower=1)

    # Hours in the upload→24h window
    hours_24h = (
        df['hours_since_publish_24h'] - df['hours_since_publish_upload']
    ).clip(lower=0.1)

    # Hours from publish to the upload poll (used for early momentum)
    hours_upload = df['hours_since_publish_upload'].clip(lower=0.1)

    # --- 1. Upload→24h velocity for all metrics ---
    for metric in VELOCITY_METRICS:
        col_u, col_24 = f"{metric}_upload", f"{metric}_24h"
        if col_u in df.columns and col_24 in df.columns:
            df[f"{metric}_velocity_24h"] = (df[col_24] - df[col_u]) / hours_24h

    # --- 2. Publish→upload velocity (how fast views/likes arrived pre-upload poll) ---
    df['view_velocity_upload'] = df['view_count_upload'] / hours_upload
    df['like_velocity_upload'] = df['like_count_upload'] / hours_upload

    # --- 3. Subscriber-normalized 24h velocity (removes channel-size bias) ---
    df['view_velocity_per_sub_24h'] = df['view_count_velocity_24h'] / sub_count
    df['like_velocity_per_sub_24h'] = df['like_count_velocity_24h'] / sub_count

    # --- 4. Velocity ratio: 24h view velocity vs channel's typical view count ---
    # Captures whether momentum is fast *for this channel*, not just in absolute terms
    df['view_velocity_ratio'] = df['view_count_velocity_24h'] / baseline_views

    # --- 5. Acceleration: did momentum increase from upload window to 24h window? ---
    df['view_velocity_acceleration'] = (
        df['view_count_velocity_24h'] / df['view_velocity_upload'].clip(lower=0.01)
    )
    df['like_velocity_acceleration'] = (
        df['like_count_velocity_24h'] / df['like_velocity_upload'].clip(lower=0.01)
    )

    print(f"  Computed velocity, upload momentum, normalized velocity, and acceleration features")
    return df


# =========================================================================
# Ratio and baseline-normalization features
# =========================================================================

def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that express counts as ratios or relative to channel norms.

    Sections:
      1. Upload-time performance vs channel baseline (early traction signal)
      2. Like rate at each snapshot (engagement quality independent of scale)
    """
    df = df.copy()

    # --- 1. Upload-time counts vs channel baseline median ---
    # Early traction signal: how is this video doing vs the channel's norm at upload?
    df['view_count_upload_vs_baseline'] = (
        df['view_count_upload'] / df['baseline_median_views'].clip(lower=1)
    )
    df['like_count_upload_vs_baseline'] = (
        df['like_count_upload'] / df['baseline_median_likes'].clip(lower=1)
    )

    # --- 2. Like rate (likes / views) at upload and 24h ---
    # Captures engagement quality independent of view volume
    df['like_rate_upload'] = np.where(
        df['view_count_upload'] > 0,
        df['like_count_upload'] / df['view_count_upload'],
        0.0,
    )
    df['like_rate_24h'] = np.where(
        df['view_count_24h'] > 0,
        df['like_count_24h'] / df['view_count_24h'],
        0.0,
    )

    print(f"  Computed ratio and baseline-normalized features")
    return df


# =========================================================================
# Subscriber-normalized features
# =========================================================================

def compute_subscriber_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize view/like/comment counts by subscriber count."""
    df = df.copy()
    sub_count = df['subscriber_count_upload'].clip(lower=1)

    for suffix in ['upload', '24h', '7d']:
        df[f"views_per_sub_{suffix}"] = df[f"view_count_{suffix}"] / sub_count
        df[f"likes_per_sub_{suffix}"] = df[f"like_count_{suffix}"] / sub_count
        df[f"comments_per_sub_{suffix}"] = df[f"comment_count_{suffix}"] / sub_count

    print(f"  Computed subscriber-normalized metrics for upload/24h/7d")
    return df


# =========================================================================
# Title category encoding
# =========================================================================

def _classify_title(title: str) -> int:
    """Classify a title into a single category using priority ordering."""
    if not isinstance(title, str):
        return TITLE_CATEGORIES['neutral']

    alpha_chars = [c for c in title if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    )
    emoji_count = sum(
        1 for c in title
        if '\U0001F600' <= c <= '\U0001F9FF'
        or '\U00002702' <= c <= '\U000027B0'
        or '\U0001FA00' <= c <= '\U0001FA6F'
    )
    has_question = '?' in title
    has_exclamation = '!' in title

    # Priority: most specific → least specific
    if has_exclamation and emoji_count >= 1 and caps_ratio > 0.3:
        return TITLE_CATEGORIES['clickbait']
    if caps_ratio > 0.5 and len(alpha_chars) > 5:
        return TITLE_CATEGORIES['all_caps']
    if emoji_count >= 3:
        return TITLE_CATEGORIES['emoji_heavy']
    if title.lower().startswith('how to'):
        return TITLE_CATEGORIES['how_to']
    if any(
        p in title.lower()
        for p in ['top ', ' best ', ' ways ', ' things ', ' tips ']
    ):
        return TITLE_CATEGORIES['listicle']
    if has_question:
        return TITLE_CATEGORIES['question']
    if has_exclamation:
        return TITLE_CATEGORIES['exclamation']

    return TITLE_CATEGORIES['neutral']


def _classify_description(desc: str) -> int:
    """Classify a description into a single category using priority ordering."""
    if not isinstance(desc, str):
        return DESC_CATEGORIES['neutral']

    length = len(desc)
    has_timestamps = bool(
        pd.Series([desc]).str.contains(r'\d{1,2}:\d{2}', regex=True).iloc[0]
    )
    link_count = desc.count('http://') + desc.count('https://')
    has_hashtags = '#' in desc

    # Priority: most specific → least specific
    if link_count >= 5:
        return DESC_CATEGORIES['link_heavy']
    if has_timestamps:
        return DESC_CATEGORIES['has_timestamps']
    if link_count > 0:
        return DESC_CATEGORIES['has_links']
    if has_hashtags:
        return DESC_CATEGORIES['has_hashtags']
    if length < 50:
        return DESC_CATEGORIES['minimal']
    if length > 2000:
        return DESC_CATEGORIES['long_form']

    return DESC_CATEGORIES['neutral']


def compute_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode title and description patterns as categorical features
    using the TITLE_CATEGORIES and DESC_CATEGORIES dictionaries.
    """
    df = df.copy()

    df['title_category'] = df['title'].apply(_classify_title)
    df['desc_category'] = df['description'].apply(_classify_description)

    # Reverse maps for readability
    title_reverse = {v: k for k, v in TITLE_CATEGORIES.items()}
    desc_reverse = {v: k for k, v in DESC_CATEGORIES.items()}

    title_dist = df['title_category'].map(title_reverse).value_counts()
    desc_dist = df['desc_category'].map(desc_reverse).value_counts()

    print(f"  Title categories:\n{title_dist.to_string()}")
    print(f"  Description categories:\n{desc_dist.to_string()}")

    return df


# =========================================================================
# Structural text features (numeric, language-agnostic)
# =========================================================================

def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute numeric structural features from title/description/tags."""
    df = df.copy()

    # --- Title ---
    df['title_length'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()

    # --- Description ---
    df['desc_length'] = df['description'].str.len()
    df['desc_link_count'] = df['description'].str.count(r'https?://')
    df['desc_hashtag_count'] = df['description'].str.count(r'#\w+')

    # --- Tags ---
    df['tag_count'] = df['tags'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df['has_tags'] = (df['tag_count'] > 0).astype(int)

    print(f"  Computed text structural features")
    return df


# =========================================================================
# Temporal features
# =========================================================================

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from publish timestamp."""
    df = df.copy()

    df['publish_hour'] = df['published_at'].dt.hour
    df['publish_dayofweek'] = df['published_at'].dt.dayofweek  # 0=Mon, 6=Sun
    df['publish_is_weekend'] = (df['publish_dayofweek'] >= 5).astype(int)

    print(f"  Computed temporal features")
    return df


# =========================================================================
# Thumbnail features
# =========================================================================

def compute_thumbnail_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from thumbnail CV signals."""
    df = df.copy()

    # Binary flag: face presence drives CTR more reliably than raw count
    df['has_face'] = (df['face_count'] > 0).astype(int)

    print(f"  Computed thumbnail features")
    return df


# =========================================================================
# Duration features
# =========================================================================

# Ordinal length buckets: S=0, M=1, L=2, XL=3
# Thresholds in seconds: Shorts / standard short-form / mid-form / long-form
_DURATION_THRESHOLDS = [(60, 0), (600, 1), (1200, 2)]


def _duration_bucket(secs: float) -> int:
    for threshold, code in _DURATION_THRESHOLDS:
        if secs <= threshold:
            return code
    return 3  # XL: > 1200s (20+ min)


def compute_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive duration-related features."""
    df = df.copy()

    df['is_short'] = (df['duration_seconds'] <= 60).astype(int)   # YouTube Shorts territory
    df['is_long'] = (df['duration_seconds'] > 1200).astype(int)   # 20+ min long-form

    # Ordinal bucket useful for trees; captures non-linear length effects
    df['duration_bucket'] = df['duration_seconds'].apply(_duration_bucket)

    print(f"  Computed duration features")
    return df


# =========================================================================
# Master pipeline
# =========================================================================

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Takes a clean, merged DataFrame and returns a modeling-ready table.

    Usage:
        df_model = engineer_features(df_clean)
    """
    print("=" * 60)
    print("Engineering features")
    print("=" * 60)

    print("\n[1/9] Computing target variable...")
    df = compute_target(df_clean)

    print("\n[2/9] Computing velocity features...")
    df = compute_velocity_features(df)

    print("\n[3/9] Computing ratio and baseline-normalized features...")
    df = compute_ratio_features(df)

    print("\n[4/9] Computing subscriber-normalized metrics...")
    df = compute_subscriber_normalized(df)

    print("\n[5/9] Computing categorical features...")
    df = compute_categorical_features(df)

    print("\n[6/9] Computing text structural features...")
    df = compute_text_features(df)

    print("\n[7/9] Computing temporal features...")
    df = compute_temporal_features(df)

    print("\n[8/9] Computing duration features...")
    df = compute_duration_features(df)

    print("\n[9/9] Computing thumbnail features...")
    df = compute_thumbnail_features(df)

    print(f"\n{'=' * 60}")
    print(f"Modeling table: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Target balance: {df['above_baseline'].value_counts().to_dict()}")
    print(f"{'=' * 60}")

    return df