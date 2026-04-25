"""Unit tests for DataSplitter.

The headline invariant: no `video_id` from the recorded validation set
ever appears in df_train or df_test — across both the create path
(first run) and the load path (subsequent runs, including drift where
some recorded ids have vanished from df_videos and may reappear later).

Tests use an in-memory `HoldoutStore` so no GCS round-trip is needed.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.pipeline_run import PipelineRun
from pipeline.stages.data_splitter import DataSplitter, HoldoutStore


class InMemoryHoldoutStore(HoldoutStore):
    """Test double — keeps the recorded payload in a Python dict."""

    def __init__(self, payload: dict = None):
        self.payload = payload

    def exists(self) -> bool:
        return self.payload is not None

    def load(self) -> dict:
        return self.payload

    def save(self, payload: dict) -> None:
        self.payload = payload

    def location(self) -> str:
        return "<in-memory>"


def _make_videos_df(n_per_cell: int = 30, seed: int = 0) -> pd.DataFrame:
    """3 verticals x 3 tiers, n_per_cell rows each — enough for stratified splits."""
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for vertical in ["Education", "Lifestyle", "Tech"]:
        for tier in ["S", "M", "L"]:
            for _ in range(n_per_cell):
                rows.append({
                    "video_id": f"v{i:05d}",
                    "vertical": vertical,
                    "tier": tier,
                    "extra": rng.random(),
                })
                i += 1
    return pd.DataFrame(rows)


def _pipeline_run(df: pd.DataFrame) -> PipelineRun:
    run = PipelineRun(config=None)
    run.df_videos = df
    return run


# ---------- create path ----------

def test_create_path_writes_payload_and_no_leak():
    df = _make_videos_df(n_per_cell=20)
    store = InMemoryHoldoutStore()
    splitter = DataSplitter(
        config=None, store=store, confirm=lambda: True, seed=42,
    )

    out = splitter.run(_pipeline_run(df))

    val_ids = set(out.df_val["video_id"])
    train_ids = set(out.df_train["video_id"])
    test_ids = set(out.df_test["video_id"])

    assert val_ids.isdisjoint(train_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids | train_ids | test_ids == set(df["video_id"])

    assert store.payload is not None
    assert set(store.payload["video_ids"]) == val_ids
    assert store.payload["total_val_rows"] == len(val_ids)


def test_create_path_decline_raises_and_does_not_persist():
    df = _make_videos_df(n_per_cell=10)
    store = InMemoryHoldoutStore()
    splitter = DataSplitter(
        config=None, store=store, confirm=lambda: False, seed=42,
    )

    with pytest.raises(RuntimeError, match="aborted"):
        splitter.run(_pipeline_run(df))

    assert store.payload is None


def test_create_path_drops_rows_missing_vertical_or_tier():
    df = _make_videos_df(n_per_cell=20)
    bad = pd.DataFrame([
        {"video_id": "bad_v", "vertical": np.nan, "tier": "M", "extra": 0.0},
        {"video_id": "bad_t", "vertical": "Tech", "tier": np.nan, "extra": 0.0},
    ])
    df = pd.concat([df, bad], ignore_index=True)

    store = InMemoryHoldoutStore()
    splitter = DataSplitter(
        config=None, store=store, confirm=lambda: True, seed=42,
    )
    out = splitter.run(_pipeline_run(df))

    seen = set(out.df_train["video_id"]) | set(out.df_test["video_id"]) | set(out.df_val["video_id"])
    assert "bad_v" not in seen
    assert "bad_t" not in seen


# ---------- load path ----------

def test_load_path_no_leak_when_all_recorded_ids_present():
    df = _make_videos_df(n_per_cell=30)
    recorded = df.groupby(["vertical", "tier"]).head(10)["video_id"].tolist()
    store = InMemoryHoldoutStore(payload={
        "created_at": "2026-04-01T00:00:00",
        "seed": 42,
        "total_val_rows": len(recorded),
        "rows_per_cell": {},
        "video_ids": recorded,
    })

    splitter = DataSplitter(config=None, store=store, seed=42)
    out = splitter.run(_pipeline_run(df))

    recorded_set = set(recorded)
    assert set(out.df_val["video_id"]) == recorded_set
    assert recorded_set.isdisjoint(set(out.df_train["video_id"]))
    assert recorded_set.isdisjoint(set(out.df_test["video_id"]))


def test_load_path_excludes_all_recorded_ids_under_drift():
    """The locked-forever invariant under drift.

    Half the recorded ids have vanished from df_videos. df_val shrinks
    accordingly, but the train/test pool must still exclude every
    recorded id — including the surviving ones (so they can't leak now)
    and the vanished ones (so they can't leak when they reappear later).
    """
    df = _make_videos_df(n_per_cell=30)
    recorded = df.groupby(["vertical", "tier"]).head(10)["video_id"].tolist()
    vanished = set(recorded[: len(recorded) // 2])
    df_present = df[~df["video_id"].isin(vanished)].reset_index(drop=True)

    store = InMemoryHoldoutStore(payload={
        "created_at": "2026-04-01T00:00:00",
        "seed": 42,
        "total_val_rows": len(recorded),
        "rows_per_cell": {},
        "video_ids": recorded,
    })

    splitter = DataSplitter(config=None, store=store, seed=42)
    out = splitter.run(_pipeline_run(df_present))

    recorded_set = set(recorded)
    train_ids = set(out.df_train["video_id"])
    test_ids = set(out.df_test["video_id"])
    val_ids = set(out.df_val["video_id"])

    assert recorded_set.isdisjoint(train_ids)
    assert recorded_set.isdisjoint(test_ids)
    assert val_ids == recorded_set - vanished


def test_load_path_does_not_mutate_store():
    df = _make_videos_df(n_per_cell=30)
    recorded = df.groupby(["vertical", "tier"]).head(10)["video_id"].tolist()
    original_payload = {
        "created_at": "2026-04-01T00:00:00",
        "seed": 42,
        "total_val_rows": len(recorded),
        "rows_per_cell": {},
        "video_ids": list(recorded),
    }
    store = InMemoryHoldoutStore(payload=dict(original_payload))

    splitter = DataSplitter(config=None, store=store, seed=42)
    splitter.run(_pipeline_run(df.iloc[: len(df) // 2].reset_index(drop=True)))

    assert store.payload == original_payload


# ---------- determinism ----------

def test_same_seed_produces_same_splits():
    df = _make_videos_df(n_per_cell=20)

    store_a = InMemoryHoldoutStore()
    store_b = InMemoryHoldoutStore()
    out_a = DataSplitter(config=None, store=store_a, confirm=lambda: True, seed=42).run(
        _pipeline_run(df)
    )
    out_b = DataSplitter(config=None, store=store_b, confirm=lambda: True, seed=42).run(
        _pipeline_run(df)
    )

    assert list(out_a.df_val["video_id"]) == list(out_b.df_val["video_id"])
    assert list(out_a.df_train["video_id"]) == list(out_b.df_train["video_id"])
    assert list(out_a.df_test["video_id"]) == list(out_b.df_test["video_id"])
