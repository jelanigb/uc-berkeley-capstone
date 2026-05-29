"""Unit tests for data_processing/data_cleanup.py."""

import pandas as pd
import numpy as np
import pytest

from data_processing.data_cleanup import (
    pivot_snapshots,
    join_baselines,
    clean_data,
    STATIC_COLS,
    METRIC_COLS,
    THUMBNAIL_COLS,
)


# =========================================================================
# Helpers
# =========================================================================

def snapshot_row_(
    video_id: str,
    poll_label: str,
    channel_id: str = "ch1",
    view_count: int = 100,
    hours_since_publish: float = 1.0,
) -> dict:
    """One row of the long-format video_snapshots table."""
    return dict(
        video_id=video_id,
        channel_id=channel_id,
        channel_handle="@handle",
        title="Test title",
        description="Test description",
        tags=["t1"],
        duration_seconds=300,
        category_id=1,
        category_name="Music",
        published_at=pd.Timestamp("2026-01-01"),
        vertical="Tech",
        tier="M",
        contains_synthetic_media=False,
        poll_label=poll_label,
        view_count=view_count,
        like_count=10,
        comment_count=5,
        subscriber_count=10_000,
        hours_since_publish=hours_since_publish,
        poll_timestamp=pd.Timestamp("2026-01-02"),
        face_count=1,
        brightness=0.5,
        colorfulness=0.3,
    )


def complete_triplet_(video_id: str, channel_id: str = "ch1") -> list[dict]:
    return [
        snapshot_row_(video_id, "upload",  channel_id, view_count=100,   hours_since_publish=1.0),
        snapshot_row_(video_id, "24h",     channel_id, view_count=1_000,  hours_since_publish=24.0),
        snapshot_row_(video_id, "7d",      channel_id, view_count=5_000,  hours_since_publish=168.0),
    ]


# =========================================================================
# pivot_snapshots
# =========================================================================

def test_pivot_snapshots_complete_triplet_produces_one_row():
    df = pd.DataFrame(complete_triplet_("v1"))
    out = pivot_snapshots(df)
    assert len(out) == 1


def test_pivot_snapshots_drops_incomplete_video():
    rows = complete_triplet_("v1") + [snapshot_row_("v2", "upload")]
    df = pd.DataFrame(rows)
    out = pivot_snapshots(df)
    assert set(out["video_id"]) == {"v1"}


def test_pivot_snapshots_column_suffixes_present():
    df = pd.DataFrame(complete_triplet_("v1"))
    out = pivot_snapshots(df)
    for metric in METRIC_COLS:
        for suffix in ["upload", "24h", "7d"]:
            assert f"{metric}_{suffix}" in out.columns


def test_pivot_snapshots_hours_columns_present():
    df = pd.DataFrame(complete_triplet_("v1"))
    out = pivot_snapshots(df)
    for suffix in ["upload", "24h", "7d"]:
        assert f"hours_since_publish_{suffix}" in out.columns


def test_pivot_snapshots_static_cols_taken_from_upload_row():
    rows = complete_triplet_("v1")
    # Change the title on the 24h row to verify upload row is used
    rows[1]["title"] = "DIFFERENT TITLE"
    df = pd.DataFrame(rows)
    out = pivot_snapshots(df)
    assert out["title"].iloc[0] == "Test title"


def test_pivot_snapshots_multiple_complete_videos():
    rows = complete_triplet_("v1") + complete_triplet_("v2")
    df = pd.DataFrame(rows)
    out = pivot_snapshots(df)
    assert len(out) == 2
    assert set(out["video_id"]) == {"v1", "v2"}


def test_pivot_snapshots_returns_independent_copy():
    df = pd.DataFrame(complete_triplet_("v1"))
    out = pivot_snapshots(df)
    assert out is not df


# =========================================================================
# join_baselines
# =========================================================================

def test_join_baselines_prefixes_non_channel_id_columns():
    df_wide = pd.DataFrame([{"video_id": "v1", "channel_id": "ch1"}])
    df_medians = pd.DataFrame([{
        "channel_id": "ch1",
        "median_views": 1_000,
        "median_likes": 100,
    }])
    out = join_baselines(df_wide, df_medians)
    assert "baseline_median_views" in out.columns
    assert "baseline_median_likes" in out.columns
    assert "channel_id" in out.columns  # key col not prefixed


def test_join_baselines_matched_channel_has_no_nulls():
    df_wide = pd.DataFrame([{"video_id": "v1", "channel_id": "ch1"}])
    df_medians = pd.DataFrame([{"channel_id": "ch1", "median_views": 500}])
    out = join_baselines(df_wide, df_medians)
    assert out["baseline_median_views"].iloc[0] == 500


def test_join_baselines_unmatched_channel_has_null_baseline():
    df_wide = pd.DataFrame([{"video_id": "v1", "channel_id": "ch_unknown"}])
    df_medians = pd.DataFrame([{"channel_id": "ch1", "median_views": 500}])
    out = join_baselines(df_wide, df_medians)
    assert pd.isna(out["baseline_median_views"].iloc[0])


def test_join_baselines_no_column_collision():
    df_wide = pd.DataFrame([{"video_id": "v1", "channel_id": "ch1", "views": 99}])
    df_medians = pd.DataFrame([{"channel_id": "ch1", "views": 500}])
    out = join_baselines(df_wide, df_medians)
    # Original video column kept, baseline prefixed
    assert "views" in out.columns
    assert "baseline_views" in out.columns


# =========================================================================
# clean_data
# =========================================================================

def test_clean_data_strips_whitespace_from_text_cols():
    df = pd.DataFrame([{
        "title": "  Hello  ",
        "description": "  World  ",
        "channel_handle": "  @handle  ",
        "tags": ["t1"],
    }])
    out = clean_data(df)
    assert out["title"].iloc[0] == "Hello"
    assert out["description"].iloc[0] == "World"
    assert out["channel_handle"].iloc[0] == "@handle"


def test_clean_data_tags_non_list_becomes_empty_list():
    df = pd.DataFrame([{
        "title": "t", "description": "d", "channel_handle": "h",
        "tags": None,
    }])
    out = clean_data(df)
    assert out["tags"].iloc[0] == []


def test_clean_data_tags_list_preserved():
    df = pd.DataFrame([{
        "title": "t", "description": "d", "channel_handle": "h",
        "tags": ["a", "b"],
    }])
    out = clean_data(df)
    assert out["tags"].iloc[0] == ["a", "b"]


def test_clean_data_clamps_negative_metric_values():
    df = pd.DataFrame([{
        "title": "t", "description": "d", "channel_handle": "h",
        "tags": [],
        "view_count_upload": -10,
        "like_count_24h": -5,
        "duration_seconds": -1,
    }])
    out = clean_data(df)
    assert out["view_count_upload"].iloc[0] == 0
    assert out["like_count_24h"].iloc[0] == 0
    assert out["duration_seconds"].iloc[0] == 0


def test_clean_data_positive_metrics_unchanged():
    df = pd.DataFrame([{
        "title": "t", "description": "d", "channel_handle": "h",
        "tags": [],
        "view_count_upload": 500,
    }])
    out = clean_data(df)
    assert out["view_count_upload"].iloc[0] == 500


def test_clean_data_drops_null_contains_synthetic_media():
    df = pd.DataFrame([
        {"title": "a", "description": "d", "channel_handle": "h", "tags": [],
         "contains_synthetic_media": False},
        {"title": "b", "description": "d", "channel_handle": "h", "tags": [],
         "contains_synthetic_media": None},
        {"title": "c", "description": "d", "channel_handle": "h", "tags": [],
         "contains_synthetic_media": True},
    ])
    out = clean_data(df)
    assert len(out) == 2
    assert out["contains_synthetic_media"].notna().all()
    assert set(out["title"]) == {"a", "c"}


def test_clean_data_without_synthetic_media_col_is_unaffected():
    # The drop is guarded — frames lacking the column pass through unchanged.
    df = pd.DataFrame([{
        "title": "t", "description": "d", "channel_handle": "h", "tags": [],
    }])
    out = clean_data(df)
    assert len(out) == 1


def test_clean_data_returns_independent_copy():
    df = pd.DataFrame([{
        "title": "  trimmed  ", "description": "d", "channel_handle": "h",
        "tags": [],
    }])
    out = clean_data(df)
    assert out is not df
    assert df["title"].iloc[0] == "  trimmed  "
