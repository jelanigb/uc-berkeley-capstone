"""
DataSplitter — carve out (or load) the locked validation set, then split
the remaining real data into train and test.

Behavior
--------
The validation set is recorded as a list of `video_id`s in a `HoldoutStore`
(GCS at `splits/validation_ids.json` in production). The first run creates
that record; every run afterwards loads it. Once written, the record is
treated as immutable — there is no force-recreate path here. That is
intentional: the validation set must be locked across the entire project
lifetime so model comparisons remain apples-to-apples.

Create path (record does not yet exist):
  - 30/70 stratified split on the `vertical_tier` cell carves out df_val.
  - The remaining 70% is split 80/20 (also stratified) into df_train / df_test
    so the final ratios are 30% val, 56% train, 14% test.
  - The split summary (rows per cell, totals) is printed and confirmation
    is requested before anything is persisted. Declining aborts.

Load path (record exists):
  - df_val is the intersection of current df_videos with the recorded
    video_ids. If some recorded ids no longer appear in df_videos
    (deleted / privated / stripped from BQ) df_val shrinks accordingly
    and a warning is printed; the record itself is left untouched so the
    historical creation count is preserved.
  - The train/test pool excludes ALL recorded ids — not just the ones
    still surviving in df_videos. This is deliberate: a video that
    disappears today and reappears later must never flip into train,
    or it would leak between val and train across runs.

Stratification
--------------
Stratifies on the `vertical_tier` cell (vertical x tier crossed) for
both the val carve-out and the train/test re-split. Rows missing
`vertical` or `tier` are dropped before splitting since both are
critical modeling features.

Dependency injection
--------------------
The persistence backend (`HoldoutStore`) and the confirmation prompt
(`confirm`) are constructor-injected so tests can swap in an in-memory
store and an auto-yes/no callable. Production defaults wire to GCS and
`input(...)` respectively.
"""

import json
from collections import Counter
from datetime import datetime
from typing import Callable

from google.cloud import storage
from sklearn.model_selection import train_test_split

from constants import BUCKET_NAME
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.run_config import RunConfig


class HoldoutStore:
    """Persistence interface for the locked validation `video_id` record."""

    def exists(self) -> bool:
        raise NotImplementedError

    def load(self) -> dict:
        raise NotImplementedError

    def save(self, payload: dict) -> None:
        raise NotImplementedError

    def location(self) -> str:
        """Human-readable location string used in console output."""
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


def _prompt_confirm() -> bool:
    return input("Proceed with write? Type 'yes' to confirm: ").strip().lower() == "yes"


class DataSplitter:
    """Stage 2 — carve out / load the locked val set, then split train / test."""

    VAL_FRAC = 0.30                  # 30% of total -> df_val
    TEST_FRAC_OF_REMAINING = 0.20    # 20% of remaining 70% -> 14% of total

    def __init__(
        self,
        config: RunConfig,
        store: HoldoutStore = None,
        confirm: Callable[[], bool] = None,
        seed: int = 42,
    ):
        self.config = config
        self.seed = seed
        self.store = store or GcsHoldoutStore()
        self.confirm = confirm or _prompt_confirm

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.SPLIT)

        df = run.df_videos.copy()

        before = len(df)
        df = df.dropna(subset=["vertical", "tier"])
        dropped = before - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with NaN vertical or tier")

        df["cell_"] = df["vertical"].astype(str) + "_" + df["tier"].astype(str)

        if self.store.exists():
            df_train, df_test, df_val = self._load_and_split(df)
        else:
            df_train, df_test, df_val = self._create_and_split(df)

        run.df_train = df_train.drop(columns=["cell_"])
        run.df_test = df_test.drop(columns=["cell_"])
        run.df_val = df_val.drop(columns=["cell_"])
        return run

    def _create_and_split(self, df):
        df_val, df_remaining = train_test_split(
            df,
            test_size=1 - self.VAL_FRAC,
            stratify=df["cell_"],
            random_state=self.seed,
        )
        df_train, df_test = train_test_split(
            df_remaining,
            test_size=self.TEST_FRAC_OF_REMAINING,
            stratify=df_remaining["cell_"],
            random_state=self.seed,
        )

        rows_per_cell = dict(Counter(df_val["cell_"]))
        total = len(df)

        print("=" * 60)
        print("VALIDATION SPLIT — first-time creation")
        print("=" * 60)
        print(f"Total rows after NaN filter: {total}")
        print(f"  df_val:   {len(df_val):>5} rows ({len(df_val) / total:.1%})")
        print(f"  df_train: {len(df_train):>5} rows ({len(df_train) / total:.1%})")
        print(f"  df_test:  {len(df_test):>5} rows ({len(df_test) / total:.1%})")
        print()
        print("Validation rows per cell (vertical_tier):")
        for cell, count in sorted(rows_per_cell.items()):
            print(f"  {cell}: {count}")
        print()
        print(f"Will write to: {self.store.location()}")
        print("Once written, this validation set is LOCKED FOREVER.")
        print()

        if not self.confirm():
            raise RuntimeError("Validation split creation aborted by user.")

        payload = {
            "created_at": datetime.now().isoformat(),
            "seed": self.seed,
            "total_val_rows": len(df_val),
            "rows_per_cell": rows_per_cell,
            "video_ids": df_val["video_id"].tolist(),
        }
        self.store.save(payload)
        print(f"Wrote {self.store.location()}")

        return df_train, df_test, df_val

    def _load_and_split(self, df):
        payload = self.store.load()
        recorded_ids = set(payload["video_ids"])
        recorded_count = payload["total_val_rows"]

        df_val = df[df["video_id"].isin(recorded_ids)]

        # Train/test pool excludes ALL recorded ids, not only the surviving
        # ones. A video that's missing today might reappear later; if it
        # does it must NOT land in train, or it would flip between val and
        # train across runs and leak.
        df_remaining = df[~df["video_id"].isin(recorded_ids)]

        missing = recorded_count - len(df_val)
        if missing > 0:
            print(
                f"Warning: {missing} of {recorded_count} holdout ids missing "
                f"from current df_videos (deleted / private?). "
                f"df_val has {len(df_val)} rows."
            )

        df_train, df_test = train_test_split(
            df_remaining,
            test_size=self.TEST_FRAC_OF_REMAINING,
            stratify=df_remaining["cell_"],
            random_state=self.seed,
        )
        return df_train, df_test, df_val
