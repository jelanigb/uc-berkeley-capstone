"""Unit tests for DataSplitter and create_holdout.

DataSplitter is load-only: it assumes the validation holdout already exists in
the store and raises with a pointer to create_validation_set.py if it does not.
One-time holdout creation is exercised via create_holdout() directly.

The headline invariant: no video_id from the recorded validation set ever
appears in df_train or df_test — across all load-path runs, including drift
where some recorded ids have vanished from df_engineered and may reappear later.

Tests operate on df_engineered (post-feature-engineering, 1 row per video with
an above_baseline column), matching the real DataSplitter contract.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.pipeline_run import PipelineRun
from pipeline.stages.data_splitter import DataSplitter, HoldoutStore, create_holdout


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


def _make_df(n_per_cell: int = 30, seed: int = 0) -> pd.DataFrame:
    """3 verticals × 3 tiers, n_per_cell rows per (vertical, tier).

    above_baseline alternates 0/1 within each cell so the 18-cell
    stratification key (vertical × tier × above_baseline) is always populated
    with at least n_per_cell / 2 rows per sub-cell.
    """
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for vertical in ["Education", "Lifestyle", "Tech"]:
        for tier in ["S", "M", "L"]:
            for j in range(n_per_cell):
                rows.append({
                    "video_id": f"v{i:05d}",
                    "vertical": vertical,
                    "tier": tier,
                    "above_baseline": j % 2,
                    "extra": rng.random(),
                })
                i += 1
    return pd.DataFrame(rows)


def _pipeline_run(df: pd.DataFrame) -> PipelineRun:
    run = PipelineRun(config=None)
    run.df_engineered = df
    return run


def _payload(video_ids: list) -> dict:
    return {
        "created_at": "2026-04-01T00:00:00",
        "seed": 42,
        "total_val_rows": len(video_ids),
        "rows_per_cell": {},
        "video_ids": list(video_ids),
    }


# ---------- DataSplitter — raises when holdout missing ----------

def test_splitter_raises_when_holdout_missing():
    df = _make_df(n_per_cell=20)
    store = InMemoryHoldoutStore()           # exists() → False
    splitter = DataSplitter(config=None, store=store, seed=42)
    with pytest.raises(RuntimeError, match="create_validation_set"):
        splitter.run(_pipeline_run(df))


# ---------- create_holdout ----------

def test_create_holdout_writes_payload():
    df = _make_df(n_per_cell=20)
    store = InMemoryHoldoutStore()

    payload = create_holdout(df, frac=0.30, store=store, seed=42)

    assert store.payload is not None
    val_ids = set(payload["video_ids"])
    assert set(store.payload["video_ids"]) == val_ids
    assert store.payload["total_val_rows"] == len(val_ids)
    # Fraction is within 5% of target
    assert abs(len(val_ids) / len(df) - 0.30) < 0.05
    # All returned IDs come from the input
    assert val_ids <= set(df["video_id"])


def test_create_holdout_decline_raises_and_does_not_persist():
    df = _make_df(n_per_cell=20)
    store = InMemoryHoldoutStore()
    with pytest.raises(RuntimeError, match="aborted"):
        create_holdout(df, frac=0.30, store=store, seed=42, confirm=lambda: False)
    assert store.payload is None


def test_create_holdout_then_splitter_produces_disjoint_splits():
    """Full workflow: create holdout, then run DataSplitter — all splits disjoint."""
    df = _make_df(n_per_cell=20)
    store = InMemoryHoldoutStore()

    create_holdout(df, frac=0.30, store=store, seed=42)
    out = DataSplitter(config=None, store=store, seed=42).run(_pipeline_run(df))

    val_ids = set(out.df_val["video_id"])
    train_ids = set(out.df_train["video_id"])
    test_ids = set(out.df_test["video_id"])

    assert val_ids.isdisjoint(train_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids | train_ids | test_ids == set(df["video_id"])


# ---------- load path ----------

def test_load_path_no_leak_when_all_recorded_ids_present():
    df = _make_df(n_per_cell=30)
    recorded = df.groupby(["vertical", "tier"]).head(10)["video_id"].tolist()
    store = InMemoryHoldoutStore(payload=_payload(recorded))

    out = DataSplitter(config=None, store=store, seed=42).run(_pipeline_run(df))

    recorded_set = set(recorded)
    assert set(out.df_val["video_id"]) == recorded_set
    assert recorded_set.isdisjoint(set(out.df_train["video_id"]))
    assert recorded_set.isdisjoint(set(out.df_test["video_id"]))


def test_load_path_excludes_all_recorded_ids_under_drift():
    """The locked-forever invariant under drift.

    Half the recorded ids have vanished from df_engineered. df_val shrinks
    accordingly, but the train/test pool must still exclude every recorded id —
    including surviving ones (can't leak now) and vanished ones (can't leak
    when they reappear later).
    """
    df = _make_df(n_per_cell=30)
    recorded = df.groupby(["vertical", "tier"]).head(10)["video_id"].tolist()
    vanished = set(recorded[: len(recorded) // 2])
    df_present = df[~df["video_id"].isin(vanished)].reset_index(drop=True)

    out = DataSplitter(config=None, store=InMemoryHoldoutStore(payload=_payload(recorded)), seed=42).run(
        _pipeline_run(df_present)
    )

    recorded_set = set(recorded)
    assert recorded_set.isdisjoint(set(out.df_train["video_id"]))
    assert recorded_set.isdisjoint(set(out.df_test["video_id"]))
    assert set(out.df_val["video_id"]) == recorded_set - vanished


def test_load_path_does_not_mutate_store():
    df = _make_df(n_per_cell=30)
    recorded = df.groupby(["vertical", "tier"]).head(10)["video_id"].tolist()
    original_payload = _payload(recorded)
    store = InMemoryHoldoutStore(payload=dict(original_payload))

    DataSplitter(config=None, store=store, seed=42).run(_pipeline_run(df))

    assert store.payload == original_payload


# ---------- determinism ----------

def test_same_seed_produces_same_splits():
    df = _make_df(n_per_cell=20)
    recorded = df.head(10)["video_id"].tolist()

    out_a = DataSplitter(config=None, store=InMemoryHoldoutStore(payload=_payload(recorded)), seed=42).run(
        _pipeline_run(df)
    )
    out_b = DataSplitter(config=None, store=InMemoryHoldoutStore(payload=_payload(recorded)), seed=42).run(
        _pipeline_run(df)
    )

    assert list(out_a.df_val["video_id"]) == list(out_b.df_val["video_id"])
    assert list(out_a.df_train["video_id"]) == list(out_b.df_train["video_id"])
    assert list(out_a.df_test["video_id"]) == list(out_b.df_test["video_id"])
