"""Unit tests for FeatureEngineerLogic.

The tests target the four pieces FeatureEngineerLogic owns directly:
drop-bad-baselines, encode-categoricals, fill-missing, feature_cols. The
end-to-end `run` is covered by a couple of small integration-style tests
that drive a minimal real-row DataFrame through the full chain (including
the underlying `engineer_features`) so the wiring is exercised too.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.stages.feature_engineer import (
    TARGET_COL_,
    VERTICAL_ORDER_,
    FeatureEngineerLogic,
    derive_feature_cols,
)


def _real_row(**overrides) -> dict:
    """One realistic, fully-populated real-data row.

    Includes every column `engineer_features` reads, so the resulting row
    survives the full pipeline. Tests override only the fields under test.
    """
    base = dict(
        video_id="v1",
        channel_id="c1",
        channel_handle="@a",
        title="Some title",
        description="Some description that is reasonably long.",
        tags=["t1", "t2"],
        category_id=1,
        category_name="Music",
        published_at=pd.Timestamp("2026-01-01 12:00:00"),
        poll_timestamp_upload=pd.Timestamp("2026-01-01 13:00:00"),
        poll_timestamp_24h=pd.Timestamp("2026-01-02 12:00:00"),
        poll_timestamp_7d=pd.Timestamp("2026-01-08 12:00:00"),
        view_count_upload=100, like_count_upload=10, comment_count_upload=5,
        view_count_24h=1000, like_count_24h=100, comment_count_24h=50,
        view_count_7d=5000, like_count_7d=500, comment_count_7d=250,
        subscriber_count_upload=10_000, subscriber_count_24h=10_010,
        hours_since_publish_upload=1.0, hours_since_publish_24h=24.0,
        baseline_median_views=2000.0, baseline_median_likes=200.0,
        baseline_median_comments=10.0, baseline_median_engagement_rate=0.1,
        baseline_channel_handle="@a", baseline_video_count=50,
        duration_seconds=300, duration_minutes=5.0,
        face_count=1,
        vertical="Tech", tier="M",
        contains_synthetic_data=False, contains_synthetic_media=False,
    )
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


@pytest.fixture
def logic() -> FeatureEngineerLogic:
    return FeatureEngineerLogic()


# ---------- _drop_bad_baselines ----------

def test_drop_bad_baselines_keeps_clean_rows(logic):
    df = _df(_real_row(video_id="a"), _real_row(video_id="b"))
    out = logic._drop_bad_baselines(df, label="train")
    assert list(out["video_id"]) == ["a", "b"]


@pytest.mark.parametrize("col", [
    "baseline_median_views",
    "baseline_median_likes",
    "baseline_median_comments",
])
def test_drop_bad_baselines_drops_nan(logic, col):
    df = _df(
        _real_row(video_id="ok"),
        _real_row(video_id="bad", **{col: np.nan}),
    )
    out = logic._drop_bad_baselines(df, label="train")
    assert list(out["video_id"]) == ["ok"]


def test_drop_bad_baselines_drops_zero_engagement_rate(logic):
    df = _df(
        _real_row(video_id="ok"),
        _real_row(video_id="bad", baseline_median_engagement_rate=0.0),
    )
    out = logic._drop_bad_baselines(df, label="train")
    assert list(out["video_id"]) == ["ok"]


def test_drop_bad_baselines_keeps_zero_comments(logic):
    """Zero comments is valid — channels with no commenting still have a
    non-zero engagement_rate if they have likes. Not a degenerate label."""
    df = _df(
        _real_row(video_id="ok"),
        _real_row(video_id="also_ok", baseline_median_comments=0),
    )
    out = logic._drop_bad_baselines(df, label="train")
    assert list(out["video_id"]) == ["ok", "also_ok"]


def test_drop_bad_baselines_returns_independent_copy(logic):
    df = _df(_real_row())
    out = logic._drop_bad_baselines(df, label="train")
    assert out is not df


# ---------- _encode_categoricals ----------

def test_encode_tier_is_ordinal(logic):
    df = _df(
        _real_row(video_id="s", tier="S"),
        _real_row(video_id="m", tier="M"),
        _real_row(video_id="l", tier="L"),
    )
    out = logic._encode_categoricals(df)
    assert out["tier_encoded"].tolist() == [0, 1, 2]


def test_encode_vertical_is_one_hot(logic):
    df = _df(
        _real_row(video_id="ed", vertical="Education"),
        _real_row(video_id="ls", vertical="Lifestyle"),
        _real_row(video_id="te", vertical="Tech"),
    )
    out = logic._encode_categoricals(df)

    one_hot = out[[f"vertical_{v}" for v in VERTICAL_ORDER_]]
    assert (one_hot.sum(axis=1) == 1).all()

    expected = pd.DataFrame(
        np.eye(3, dtype=int),
        columns=[f"vertical_{v}" for v in VERTICAL_ORDER_],
    )
    pd.testing.assert_frame_equal(
        one_hot.reset_index(drop=True), expected, check_dtype=False
    )


def test_encode_unknown_vertical_is_all_zero(logic):
    df = _df(_real_row(vertical="Unknown"))
    out = logic._encode_categoricals(df)
    one_hot = out[[f"vertical_{v}" for v in VERTICAL_ORDER_]]
    assert (one_hot.sum(axis=1) == 0).all()


# ---------- _fill_missing ----------

def test_fill_missing_default_replaces_nan_and_inf(logic):
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.inf, -np.inf]})
    out = logic._fill_missing(df)
    assert out["a"].tolist() == [1.0, 0.0]
    assert out["b"].tolist() == [0.0, 0.0]


def test_fill_missing_subset_only_touches_subset(logic):
    df = pd.DataFrame({"a": [np.nan, 1.0], "b": [np.nan, 2.0]})
    out = logic._fill_missing(df, fill_value=0, subset=["a"])
    assert out["a"].tolist() == [0.0, 1.0]
    assert pd.isna(out["b"].iloc[0])
    assert out["b"].iloc[1] == 2.0


def test_fill_missing_empty_subset_is_treated_as_all(logic):
    df = pd.DataFrame({"a": [np.nan, 1.0], "b": [np.nan, 2.0]})
    out = logic._fill_missing(df, fill_value=-1, subset=[])
    assert out["a"].tolist() == [-1.0, 1.0]
    assert out["b"].tolist() == [-1.0, 2.0]


def test_fill_missing_returns_independent_copy(logic):
    df = pd.DataFrame({"a": [np.nan]})
    out = logic._fill_missing(df)
    assert out is not df
    assert pd.isna(df.loc[0, "a"])


def test_fill_missing_custom_value(logic):
    df = pd.DataFrame({"a": [np.nan, 1.0]})
    out = logic._fill_missing(df, fill_value=-999)
    assert out["a"].tolist() == [-999.0, 1.0]


# ---------- feature_cols ----------

def test_feature_cols_excludes_target(logic):
    df = pd.DataFrame({"above_baseline": [0], "feature_x": [1]})
    cols = derive_feature_cols(df)
    assert "above_baseline" not in cols
    assert "feature_x" in cols


def test_feature_cols_excludes_seven_d_suffixed(logic):
    df = pd.DataFrame({
        "view_count_7d": [0],
        "view_count_24h": [0],
        "feature_x": [1],
    })
    cols = derive_feature_cols(df)
    assert "view_count_7d" not in cols
    assert "view_count_24h" in cols
    assert "feature_x" in cols


def test_feature_cols_excludes_raw_categoricals_keeps_encoded(logic):
    df = pd.DataFrame({
        "vertical": ["Tech"],
        "tier": ["M"],
        "tier_encoded": [1],
        "vertical_Tech": [1],
    })
    cols = derive_feature_cols(df)
    assert "vertical" not in cols
    assert "tier" not in cols
    assert "tier_encoded" in cols
    assert "vertical_Tech" in cols


def test_feature_cols_excludes_raw_baselines_and_intermediates(logic):
    df = pd.DataFrame({
        "baseline_median_views": [0],
        "baseline_engagement": [0],
        "engagement_7d": [0],
        "feature_x": [1],
    })
    cols = derive_feature_cols(df)
    assert "baseline_median_views" not in cols
    assert "baseline_engagement" not in cols
    assert "engagement_7d" not in cols
    assert "feature_x" in cols


# ---------- run end-to-end ----------

def test_engineer_returns_engineered_df_with_expected_cols(logic):
    df_train = _df(_real_row(video_id="t1"), _real_row(video_id="t2"))
    df_test = _df(_real_row(video_id="te1"))
    df_val = _df(_real_row(video_id="v1"))

    out_train = logic.engineer(df_train, label="train")
    out_test = logic.engineer(df_test, label="test")
    out_val = logic.engineer(df_val, label="val")

    for out in (out_train, out_test, out_val):
        assert TARGET_COL_ in out.columns
        assert "tier_encoded" in out.columns
        for v in VERTICAL_ORDER_:
            assert f"vertical_{v}" in out.columns

    assert set(out_train.columns) == set(out_test.columns) == set(out_val.columns)


def test_engineer_drops_bad_baselines(logic):
    df_nan = _df(
        _real_row(video_id="ok"),
        _real_row(video_id="bad_nan_likes", baseline_median_likes=np.nan),
    )
    df_zero_rate = _df(
        _real_row(video_id="ok"),
        _real_row(video_id="bad_zero_rate", baseline_median_engagement_rate=0.0),
    )

    assert list(logic.engineer(df_nan, label="nan")["video_id"]) == ["ok"]
    assert list(logic.engineer(df_zero_rate, label="zero_rate")["video_id"]) == ["ok"]


def test_engineer_feature_cols_excludes_target_and_seven_d(logic):
    df = _df(_real_row(video_id="v"))
    out = logic.engineer(df, label="all")

    cols = derive_feature_cols(out)
    assert TARGET_COL_ not in cols
    assert not any(c.endswith("_7d") for c in cols)
    assert "tier_encoded" in cols
