"""Unit tests for data_processing/synthetic_data.py.

generate_synthetic_data() requires a fitted SDV synthesizer and is not
tested here. The three helpers that contain all post-generation logic —
_postprocess_synthetic, _recompute_engineered_features, and
combine_real_and_synthetic — are pure DataFrame operations and are
exercised below.
"""

import pandas as pd
import numpy as np
import pytest

from data_processing.synthetic_data import (
    _postprocess_synthetic,
    _recompute_engineered_features,
    combine_real_and_synthetic,
    NON_NEGATIVE_COLS,
    INTEGER_COLS,
)


# =========================================================================
# Helpers
# =========================================================================

def synth_row_() -> dict:
    """Minimal row with every column _postprocess_synthetic may touch."""
    return dict(
        # poll metrics — all non-negative and integer after post-process
        view_count_upload=100,    view_count_24h=800,    view_count_7d=4_000,
        like_count_upload=10,     like_count_24h=80,     like_count_7d=400,
        comment_count_upload=5,   comment_count_24h=50,  comment_count_7d=250,
        subscriber_count_upload=10_000, subscriber_count_24h=10_010,
        subscriber_count_7d=10_020,
        hours_since_publish_upload=1.0, hours_since_publish_24h=24.0,
        hours_since_publish_7d=168.0,
        face_count=1, brightness=0.5, colorfulness=0.3,
        duration_seconds=300,
        vertical="Tech", tier="M", category_id=1,
        # baselines
        baseline_median_views=2_000.0, baseline_median_likes=200.0,
        baseline_median_comments=10.0,
        baseline_median_engagement_rate=0.1,
        baseline_channel_handle="@ch", baseline_video_count=50,
        channel_id="ch1",
    )


def synth_df_(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).copy()


# =========================================================================
# _postprocess_synthetic — non-negative clamping
# =========================================================================

def test_postprocess_clamps_negative_values_to_zero():
    row = synth_row_()
    # duration_seconds and brightness have no monotonic override — safe to test
    row["duration_seconds"] = -10
    row["brightness"] = -0.5
    out = _postprocess_synthetic(synth_df_(row))
    assert out["duration_seconds"].iloc[0] == 0
    assert out["brightness"].iloc[0] == 0


def test_postprocess_leaves_positive_values_unchanged():
    row = synth_row_()
    row["view_count_upload"] = 200
    out = _postprocess_synthetic(synth_df_(row))
    assert out["view_count_upload"].iloc[0] == 200


# =========================================================================
# _postprocess_synthetic — integer rounding
# =========================================================================

def test_postprocess_rounds_integer_cols():
    row = synth_row_()
    row["view_count_upload"] = 99.7
    row["face_count"] = 1.4
    out = _postprocess_synthetic(synth_df_(row))
    assert out["view_count_upload"].iloc[0] == 100
    assert out["face_count"].iloc[0] == 1


# =========================================================================
# _postprocess_synthetic — metric monotonicity
# =========================================================================

def test_postprocess_enforces_view_monotonicity():
    row = synth_row_()
    # Make 24h < upload (violation)
    row["view_count_upload"] = 1_000
    row["view_count_24h"] = 500     # violation: < upload
    row["view_count_7d"] = 200      # violation: < 24h
    out = _postprocess_synthetic(synth_df_(row))
    assert out["view_count_24h"].iloc[0] >= out["view_count_upload"].iloc[0]
    assert out["view_count_7d"].iloc[0] >= out["view_count_24h"].iloc[0]


def test_postprocess_enforces_hours_monotonicity():
    row = synth_row_()
    row["hours_since_publish_upload"] = 5.0
    row["hours_since_publish_24h"] = 3.0    # violation
    row["hours_since_publish_7d"] = 1.0     # violation
    out = _postprocess_synthetic(synth_df_(row))
    assert out["hours_since_publish_24h"].iloc[0] >= out["hours_since_publish_upload"].iloc[0]
    assert out["hours_since_publish_7d"].iloc[0] >= out["hours_since_publish_24h"].iloc[0]


# =========================================================================
# _postprocess_synthetic — synthetic markers
# =========================================================================

def test_postprocess_sets_synthetic_flag():
    out = _postprocess_synthetic(synth_df_(synth_row_()))
    assert out["contains_synthetic_data"].iloc[0] == True


def test_postprocess_video_ids_are_syn_prefixed():
    rows = [synth_row_() for _ in range(3)]
    out = _postprocess_synthetic(synth_df_(*rows))
    assert all(v.startswith("syn-") for v in out["video_id"])


def test_postprocess_video_ids_are_unique():
    rows = [synth_row_() for _ in range(5)]
    out = _postprocess_synthetic(synth_df_(*rows))
    assert out["video_id"].nunique() == 5


def test_postprocess_sets_placeholder_text_fields():
    out = _postprocess_synthetic(synth_df_(synth_row_()))
    assert out["title"].iloc[0] == "[Synthetic]"
    assert out["description"].iloc[0] == "[Synthetic]"
    assert out["tags"].iloc[0] == []


# =========================================================================
# _recompute_engineered_features
# =========================================================================

def test_recompute_adds_above_baseline_column():
    out = _recompute_engineered_features(synth_df_(synth_row_()))
    assert "above_baseline" in out.columns
    assert out["above_baseline"].iloc[0] in (0, 1)


def test_recompute_adds_velocity_columns():
    out = _recompute_engineered_features(synth_df_(synth_row_()))
    for metric in ["view_count", "like_count", "comment_count"]:
        assert f"{metric}_velocity_24h" in out.columns


def test_recompute_adds_subscriber_normalized_columns():
    out = _recompute_engineered_features(synth_df_(synth_row_()))
    for suffix in ["upload", "24h", "7d"]:
        assert f"views_per_sub_{suffix}" in out.columns


def test_recompute_above_baseline_is_one_when_engagement_exceeds_channel():
    row = synth_row_()
    # engagement_7d = (400 + 250) / 4000 = 0.1625
    # baseline_engagement = (200 + 10) / 2000 = 0.105
    row.update(
        like_count_7d=400, comment_count_7d=250, view_count_7d=4_000,
        baseline_median_likes=200.0, baseline_median_comments=10.0,
        baseline_median_views=2_000.0,
    )
    out = _recompute_engineered_features(synth_df_(row))
    assert out["above_baseline"].iloc[0] == 1


# =========================================================================
# combine_real_and_synthetic
# =========================================================================

def real_df_(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": f"real_{i}", "feature_a": i, "feature_b": float(i)}
        for i in range(n)
    ])


def fake_synth_df_(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": f"syn-{i:06d}", "feature_a": 100 + i, "feature_b": float(100 + i),
         "contains_synthetic_data": True}
        for i in range(n)
    ])


def test_combine_total_row_count():
    out = combine_real_and_synthetic(real_df_(3), fake_synth_df_(2))
    assert len(out) == 5


def test_combine_real_rows_flagged_false():
    out = combine_real_and_synthetic(real_df_(3), fake_synth_df_(2))
    real_flags = out[out["video_id"].str.startswith("real_")]["contains_synthetic_data"]
    assert (real_flags == False).all()


def test_combine_synthetic_rows_flagged_true():
    out = combine_real_and_synthetic(real_df_(3), fake_synth_df_(2))
    synth_flags = out[out["video_id"].str.startswith("syn-")]["contains_synthetic_data"]
    assert (synth_flags == True).all()


def test_combine_no_nulls_in_shared_columns():
    out = combine_real_and_synthetic(real_df_(3), fake_synth_df_(2))
    assert out["feature_a"].notna().all()
    assert out["feature_b"].notna().all()


def test_combine_does_not_mutate_real_df():
    real = real_df_(3)
    original_cols = list(real.columns)
    combine_real_and_synthetic(real, fake_synth_df_(2))
    assert list(real.columns) == original_cols
    assert "contains_synthetic_data" not in real.columns
