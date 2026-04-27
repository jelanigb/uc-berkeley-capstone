"""
DataSplitter — filter the locked validation set out of df_engineered, then
split the remaining rows into train and test.

Behavior
--------
DataSplitter is load-only: it assumes the validation holdout record already
exists in the HoldoutStore. If the record does not exist the stage raises with
a clear pointer to `scripts/create_validation_set.py`, which must be run once
before any pipeline scenario.

The holdout record is a list of `video_id` values locked at creation time.
DataSplitter filters df_engineered (post-engineering, 1 row per video) against
that list to produce df_val, then splits the remaining rows into df_train and
df_test. It also derives X/y matrices from each split — unscaled — ready for
the Scaler stage.

Stratification
--------------
Both the runtime train/test split and the one-time holdout creation use the
same `stratify_key(df)` function defined here: vertical × tier × above_baseline
(18 cells for 3 verticals × 3 tiers × 2 target classes). Using above_baseline
here requires that feature engineering has already computed the target, which
is why DataSplitter runs after FeatureEngineer in the new architecture.

`stratify_key` and `create_holdout` are exported so `create_validation_set.py`
can reuse them without duplicating logic.

Load path notes
---------------
The train/test pool excludes ALL recorded video_ids — not just surviving ones.
A video that disappears today and reappears later must never flip into train,
or it would leak between val and train across runs.

If some recorded ids are missing from df_engineered (deleted / privated), df_val
shrinks and a warning is printed; the stored record is left untouched so the
historical creation count is preserved.
"""

import json
from collections import Counter
from datetime import datetime
from typing import Callable

import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

from constants import BUCKET_NAME
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import VersionConfig
from pipeline.stages.feature_engineer import TARGET_COL_, derive_feature_cols


# ── public helpers ────────────────────────────────────────────────────────────

def stratify_key(df: pd.DataFrame) -> pd.Series:
    """18-cell stratification key: vertical × tier × above_baseline.

    Both DataSplitter (runtime train/test split) and create_holdout (one-time
    validation creation) use this function so their stratification is identical.
    Requires df to have vertical, tier, and above_baseline columns.
    """
    return (
        df["vertical"].astype(str)
        + "_" + df["tier"].astype(str)
        + "_" + df[TARGET_COL_].astype(str)
    )


def create_holdout(
    df_engineered: pd.DataFrame,
    frac: float,
    store: "HoldoutStore",
    seed: int = 42,
    confirm: Callable[[], bool] = None,
) -> dict:
    """Stratified holdout split for one-time validation set creation.

    Splits `frac` of df_engineered into a validation set, saves the video_ids
    to `store`, and returns the payload dict. Raises if any stratification cell
    has fewer than 2 rows (can't split). Never called during normal pipeline
    runs — only by scripts/create_validation_set.py.

    confirm: callable returning bool. If None, proceeds without prompting.
    """
    if len(df_engineered) == 0:
        raise ValueError(
            "df_engineered is empty — 0 rows available for holdout creation.\n"
            "Likely causes:\n"
            "  1. No videos with all 3 poll labels: check 'Videos with all 3 polls:'"
            " in the pivot_snapshots output above.\n"
            "  2. All rows failed baseline filtering: look for '_drop_bad_baselines"
            " dropped N rows' in the FeatureEngineer output above, meaning the"
            " channel baseline join (Step 2) matched no channels."
        )

    key = stratify_key(df_engineered)
    cell_counts = key.value_counts()
    small_cells = cell_counts[cell_counts < 2]
    if not small_cells.empty:
        raise ValueError(
            f"Cannot stratify — {len(small_cells)} cell(s) have fewer than 2 rows:\n"
            + small_cells.to_string()
        )

    df_val, df_remaining = train_test_split(
        df_engineered,
        test_size=1 - frac,
        stratify=key,
        random_state=seed,
    )

    rows_per_cell = dict(Counter(stratify_key(df_val)))
    total = len(df_engineered)

    print("=" * 60)
    print("VALIDATION SPLIT — first-time creation")
    print("=" * 60)
    print(f"Total rows: {total}")
    print(f"  df_val:       {len(df_val):>6,} rows ({len(df_val) / total:.1%})")
    print(f"  df_remaining: {len(df_remaining):>6,} rows ({len(df_remaining) / total:.1%})")
    print()
    print("Validation rows per cell (vertical_tier_class):")
    for cell, count in sorted(rows_per_cell.items()):
        print(f"  {cell}: {count}")
    print()
    print(f"Will write to: {store.location()}")
    print("Once written, this validation set is LOCKED FOREVER.")
    print()

    if confirm is not None and not confirm():
        raise RuntimeError("Validation split creation aborted by user.")

    payload = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "total_val_rows": len(df_val),
        "rows_per_cell": rows_per_cell,
        "video_ids": df_val["video_id"].tolist(),
    }
    store.save(payload)
    print(f"Wrote {store.location()}")
    return payload


# ── holdout store ─────────────────────────────────────────────────────────────

class HoldoutStore:
    """Persistence interface for the locked validation `video_id` record."""

    def exists(self) -> bool:
        raise NotImplementedError

    def load(self) -> dict:
        raise NotImplementedError

    def save(self, payload: dict) -> None:
        raise NotImplementedError

    def location(self) -> str:
        raise NotImplementedError


class GcsHoldoutStore(HoldoutStore):
    """Default production backend: a single JSON blob in GCS."""

    DEFAULT_PATH = "splits/validation_ids.json"

    def __init__(self, bucket_name: str = BUCKET_NAME, path: str = DEFAULT_PATH):
        self.bucket_name = bucket_name
        self.path = path
        self.blob_ = storage.Client().bucket(bucket_name).blob(path)

    def exists(self) -> bool:
        return self.blob_.exists()

    def load(self) -> dict:
        return json.loads(self.blob_.download_as_text())

    def save(self, payload: dict) -> None:
        self.blob_.upload_from_string(
            json.dumps(payload, indent=2),
            content_type="application/json",
        )

    def location(self) -> str:
        return f"gs://{self.bucket_name}/{self.path}"


class InMemoryHoldoutStore(HoldoutStore):
    """In-memory store for dry-run / testing. Never touches GCS."""

    _GCS_PATH = f"gs://{BUCKET_NAME}/splits/validation_ids.json"

    def __init__(self):
        self.payload_ = None

    def exists(self) -> bool:
        return False

    def load(self) -> dict:
        raise RuntimeError("InMemoryHoldoutStore.load() should never be called.")

    def save(self, payload: dict) -> None:
        self.payload_ = payload

    def location(self) -> str:
        return self._GCS_PATH


def _prompt_confirm() -> bool:
    return input("Proceed with write? Type 'yes' to confirm: ").strip().lower() == "yes"


# ── stage ─────────────────────────────────────────────────────────────────────

class DataSplitter:
    """Stage 4 — load holdout ids, filter df_engineered, split train/test, derive X/y."""

    TEST_FRAC_OF_REMAINING = 0.20    # 20% of non-holdout rows → df_test

    def __init__(
        self,
        config: VersionConfig,
        store: HoldoutStore = None,
        seed: int = 42,
    ):
        self.config = config
        self.seed = seed
        self.store = store or GcsHoldoutStore()

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.SPLIT)

        if not self.store.exists():
            raise RuntimeError(
                "Validation holdout not found at "
                f"{self.store.location()}.\n"
                "Run `python scripts/create_validation_set.py --yes` once to "
                "create and lock the holdout before running any pipeline scenario."
            )

        df = run.df_engineered.copy()
        feature_cols = derive_feature_cols(df)

        df_train, df_test, df_val = self._load_and_split(df)

        run.df_train = df_train
        run.df_test = df_test
        run.df_val = df_val

        run.X_train = df_train[feature_cols].copy()
        run.y_train = df_train[TARGET_COL_].copy()
        run.X_test = df_test[feature_cols].copy()
        run.y_test = df_test[TARGET_COL_].copy()
        run.X_val = df_val[feature_cols].copy()
        run.y_val = df_val[TARGET_COL_].copy()

        return run

    def _load_and_split(self, df: pd.DataFrame):
        payload = self.store.load()
        recorded_ids = set(payload["video_ids"])
        recorded_count = payload["total_val_rows"]

        df_val = df[df["video_id"].isin(recorded_ids)]

        # Exclude ALL recorded ids — not just surviving ones. A video that
        # disappears today and reappears later must never flip into train.
        df_remaining = df[~df["video_id"].isin(recorded_ids)]

        missing = recorded_count - len(df_val)
        if missing > 0:
            print(
                f"Warning: {missing} of {recorded_count} holdout ids missing "
                f"from current df_engineered (deleted / private?). "
                f"df_val has {len(df_val)} rows."
            )

        df_train, df_test = train_test_split(
            df_remaining,
            test_size=self.TEST_FRAC_OF_REMAINING,
            stratify=stratify_key(df_remaining),
            random_state=self.seed,
        )

        total = len(df)
        print(f"DataSplitter — loaded holdout ({len(df_val):,} val rows):")
        print(f"  df_val:   {len(df_val):>6,} rows ({len(df_val) / total:.1%})")
        print(f"  df_train: {len(df_train):>6,} rows ({len(df_train) / total:.1%})")
        print(f"  df_test:  {len(df_test):>6,} rows ({len(df_test) / total:.1%})")

        return df_train, df_test, df_val
