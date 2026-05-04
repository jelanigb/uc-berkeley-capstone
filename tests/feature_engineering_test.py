"""Unit tests for standalone functions in data_processing/feature_engineering.py."""

import pandas as pd
import numpy as np
import pytest

from data_processing.feature_engineering import (
    _classify_title,
    _classify_description,
    _duration_bucket,
    compute_target,
    compute_velocity_features,
    compute_ratio_features,
    compute_subscriber_normalized,
    compute_categorical_features,
    compute_text_features,
    compute_temporal_features,
    compute_thumbnail_features,
    compute_duration_features,
    TITLE_CATEGORIES,
    DESC_CATEGORIES,
)


# =========================================================================
# Helpers
# =========================================================================

def velocity_row_() -> dict:
    """Minimal row supplying every column compute_velocity_features reads."""
    return dict(
        subscriber_count_upload=10_000,
        baseline_median_views=2_000.0,
        hours_since_publish_upload=1.0,
        hours_since_publish_24h=25.0,
        view_count_upload=100, view_count_24h=1_100,
        like_count_upload=10,  like_count_24h=110,
        comment_count_upload=5, comment_count_24h=55,
        subscriber_count_upload_=None,  # unused alias check
        subscriber_count_24h=10_010,
    )


def base_row_() -> dict:
    """Row supplying all columns needed by compute_* functions."""
    return dict(
        # target
        view_count_7d=5_000, like_count_7d=500, comment_count_7d=250,
        baseline_median_views=2_000.0, baseline_median_likes=200.0,
        baseline_median_comments=10.0,
        # velocity
        subscriber_count_upload=10_000, subscriber_count_24h=10_010,
        hours_since_publish_upload=1.0, hours_since_publish_24h=25.0,
        view_count_upload=100, view_count_24h=1_100,
        like_count_upload=10,  like_count_24h=110,
        comment_count_upload=5, comment_count_24h=55,
        # text / categorical
        title="Some title here",
        description="A description that is reasonably long enough to be neutral.",
        tags=["t1", "t2"],
        # temporal
        published_at=pd.Timestamp("2026-01-01 12:00:00"),
        # thumbnail
        face_count=1,
        # duration
        duration_seconds=300,
    )


def df_(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# =========================================================================
# _classify_title
# =========================================================================

@pytest.mark.parametrize("title,expected_key", [
    ("A regular video title", "neutral"),
    ("Did you watch this video?", "question"),
    ("This is amazing!", "exclamation"),
    ("How to make money online", "how_to"),
    ("Top 10 ways to succeed", "listicle"),
    ("Best tips for beginners", "listicle"),
    ("Fun video 😀😂😊", "emoji_heavy"),       # 3 emoji in U+1F600–1F9FF range
    ("HELLO WORLD IS GREAT TODAY", "all_caps"),
    ("WOW! Amazing 😀 VIDEO", "clickbait"),    # ! + emoji in range + high caps
])
def test_classify_title(title, expected_key):
    assert _classify_title(title) == TITLE_CATEGORIES[expected_key]


def test_classify_title_non_string_returns_neutral():
    assert _classify_title(None) == TITLE_CATEGORIES["neutral"]
    assert _classify_title(42) == TITLE_CATEGORIES["neutral"]


# =========================================================================
# _classify_description
# =========================================================================

@pytest.mark.parametrize("desc,expected_key", [
    ("Short", "minimal"),
    ("a" * 2001, "long_form"),
    ("check out https://example.com for more info", "has_links"),
    (
        "https://a.com https://b.com https://c.com https://d.com https://e.com",
        "link_heavy",
    ),
    ("0:00 Intro 1:30 Deep dive 5:00 Outro", "has_timestamps"),
    ("Great content #subscribe #like #share", "has_hashtags"),
    ("A regular description about the content of this video.", "neutral"),
])
def test_classify_description(desc, expected_key):
    assert _classify_description(desc) == DESC_CATEGORIES[expected_key]


def test_classify_description_non_string_returns_neutral():
    assert _classify_description(None) == DESC_CATEGORIES["neutral"]


# =========================================================================
# _duration_bucket
# =========================================================================

@pytest.mark.parametrize("secs,expected", [
    (0,    0),   # S: <= 60
    (60,   0),   # S: boundary
    (61,   1),   # M: > 60
    (600,  1),   # M: boundary
    (601,  2),   # L: > 600
    (1200, 2),   # L: boundary
    (1201, 3),   # XL: > 1200
    (3600, 3),   # XL: well above
])
def test_duration_bucket(secs, expected):
    assert _duration_bucket(secs) == expected


# =========================================================================
# compute_target
# =========================================================================

def test_compute_target_above_baseline():
    row = base_row_()
    # engagement_7d = (500+250)/5000 = 0.15
    # baseline_engagement = (200+10)/2000 = 0.105
    # 0.15 > 0.105 → above_baseline = 1
    df = df_(row)
    out = compute_target(df)
    assert out["above_baseline"].iloc[0] == 1


def test_compute_target_below_baseline():
    row = base_row_()
    row.update(like_count_7d=1, comment_count_7d=0, view_count_7d=5_000)
    # engagement_7d = 1/5000 = 0.0002 < 0.105
    df = df_(row)
    out = compute_target(df)
    assert out["above_baseline"].iloc[0] == 0


def test_compute_target_zero_views_gives_zero_engagement():
    row = base_row_()
    row["view_count_7d"] = 0
    out = compute_target(df_(row))
    assert out["engagement_7d"].iloc[0] == 0.0


def test_compute_target_missing_baseline_cols_defaults_to_zero():
    row = {k: v for k, v in base_row_().items()
           if "baseline_median" not in k}
    out = compute_target(df_(row))
    assert out["baseline_engagement"].iloc[0] == 0.0


def test_compute_target_returns_independent_copy():
    df = df_(base_row_())
    out = compute_target(df)
    assert out is not df


# =========================================================================
# compute_velocity_features
# =========================================================================

def test_compute_velocity_features_produces_expected_columns():
    df = df_(base_row_())
    out = compute_velocity_features(df)
    for col in [
        "view_count_velocity_24h", "like_count_velocity_24h",
        "view_velocity_upload", "like_velocity_upload",
        "view_velocity_per_sub_24h", "like_velocity_per_sub_24h",
        "view_velocity_ratio", "view_velocity_acceleration",
        "like_velocity_acceleration",
    ]:
        assert col in out.columns, f"Missing column: {col}"


def test_compute_velocity_features_arithmetic():
    row = base_row_()
    # hours_24h = 25 - 1 = 24; view delta = 1100 - 100 = 1000 → velocity = 1000/24
    out = compute_velocity_features(df_(row))
    expected_velocity = (1100 - 100) / (25.0 - 1.0)
    assert abs(out["view_count_velocity_24h"].iloc[0] - expected_velocity) < 1e-9


def test_compute_velocity_features_zero_hour_gap_does_not_divide_by_zero():
    row = base_row_()
    row["hours_since_publish_upload"] = 0.0
    row["hours_since_publish_24h"] = 0.0
    out = compute_velocity_features(df_(row))
    assert out["view_count_velocity_24h"].notna().all()


# =========================================================================
# compute_ratio_features
# =========================================================================

def test_compute_ratio_features_produces_expected_columns():
    out = compute_ratio_features(df_(base_row_()))
    for col in [
        "view_count_upload_vs_baseline",
        "like_count_upload_vs_baseline",
        "like_rate_upload",
        "like_rate_24h",
    ]:
        assert col in out.columns


def test_compute_ratio_features_zero_views_gives_zero_like_rate():
    row = base_row_()
    row["view_count_upload"] = 0
    row["view_count_24h"] = 0
    out = compute_ratio_features(df_(row))
    assert out["like_rate_upload"].iloc[0] == 0.0
    assert out["like_rate_24h"].iloc[0] == 0.0


# =========================================================================
# compute_subscriber_normalized
# =========================================================================

def test_compute_subscriber_normalized_produces_nine_columns():
    out = compute_subscriber_normalized(df_(base_row_()))
    for suffix in ["upload", "24h", "7d"]:
        for metric in ["views", "likes", "comments"]:
            assert f"{metric}_per_sub_{suffix}" in out.columns


def test_compute_subscriber_normalized_zero_subscribers_does_not_divide_by_zero():
    row = base_row_()
    row["subscriber_count_upload"] = 0
    out = compute_subscriber_normalized(df_(row))
    assert out["views_per_sub_upload"].notna().all()


# =========================================================================
# compute_categorical_features
# =========================================================================

def test_compute_categorical_features_adds_title_and_desc_category():
    out = compute_categorical_features(df_(base_row_()))
    assert "title_category" in out.columns
    assert "desc_category" in out.columns


def test_compute_categorical_features_maps_to_known_values():
    out = compute_categorical_features(df_(base_row_()))
    assert out["title_category"].iloc[0] in TITLE_CATEGORIES.values()
    assert out["desc_category"].iloc[0] in DESC_CATEGORIES.values()


# =========================================================================
# compute_text_features
# =========================================================================

def test_compute_text_features_title_length_and_word_count():
    row = base_row_()
    row["title"] = "hello world test"
    out = compute_text_features(df_(row))
    assert out["title_length"].iloc[0] == len("hello world test")
    assert out["title_word_count"].iloc[0] == 3


def test_compute_text_features_tag_count_from_list():
    row = base_row_()
    row["tags"] = ["a", "b", "c"]
    out = compute_text_features(df_(row))
    assert out["tag_count"].iloc[0] == 3
    assert out["has_tags"].iloc[0] == 1


def test_compute_text_features_tag_count_non_list_is_zero():
    row = base_row_()
    row["tags"] = None
    out = compute_text_features(df_(row))
    assert out["tag_count"].iloc[0] == 0
    assert out["has_tags"].iloc[0] == 0


# =========================================================================
# compute_temporal_features
# =========================================================================

def test_compute_temporal_features_weekday():
    row = base_row_()
    row["published_at"] = pd.Timestamp("2026-01-05 09:00:00")  # Monday
    out = compute_temporal_features(df_(row))
    assert out["publish_hour"].iloc[0] == 9
    assert out["publish_dayofweek"].iloc[0] == 0  # Monday
    assert out["publish_is_weekend"].iloc[0] == 0


def test_compute_temporal_features_weekend():
    row = base_row_()
    row["published_at"] = pd.Timestamp("2026-01-04 15:00:00")  # Sunday
    out = compute_temporal_features(df_(row))
    assert out["publish_is_weekend"].iloc[0] == 1


# =========================================================================
# compute_thumbnail_features
# =========================================================================

def test_compute_thumbnail_features_face_present():
    row = base_row_()
    row["face_count"] = 2
    out = compute_thumbnail_features(df_(row))
    assert out["has_face"].iloc[0] == 1


def test_compute_thumbnail_features_no_face():
    row = base_row_()
    row["face_count"] = 0
    out = compute_thumbnail_features(df_(row))
    assert out["has_face"].iloc[0] == 0


# =========================================================================
# compute_duration_features
# =========================================================================

@pytest.mark.parametrize("secs,is_short,is_long,bucket", [
    (30,   1, 0, 0),   # S
    (60,   1, 0, 0),   # S boundary
    (300,  0, 0, 1),   # M
    (1500, 0, 1, 3),   # XL
])
def test_compute_duration_features(secs, is_short, is_long, bucket):
    row = base_row_()
    row["duration_seconds"] = secs
    out = compute_duration_features(df_(row))
    assert out["is_short"].iloc[0] == is_short
    assert out["is_long"].iloc[0] == is_long
    assert out["duration_bucket"].iloc[0] == bucket
