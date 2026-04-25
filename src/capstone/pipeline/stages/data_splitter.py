"""
DataSplitter — carve out (or load) the locked validation set, then split
the remaining real data into train and test.

Behavior
--------
The validation set is recorded as a list of `video_id`s in GCS at
`splits/validation_ids.json`. The first run creates that file; every run
afterwards loads it. Once written, the file is treated as immutable —
there is no force-recreate path here. That is intentional: the validation
set must be locked across the entire project lifetime so model comparisons
remain apples-to-apples.

Create path (file does not yet exist):
  - 30/70 stratified split on the `vertical_tier` cell carves out df_val.
  - The remaining 70% is split 80/20 (also stratified) into df_train / df_test
    so the final ratios are 30% val, 56% train, 14% test.
  - The split summary (rows per cell, totals) is printed and confirmation
    is requested before anything is written to GCS. Declining aborts.

Load path (file exists):
  - df_val is the intersection of current df_videos with the recorded
    video_ids. If some recorded ids no longer appear in df_videos
    (deleted / privated / stripped from BQ) df_val shrinks accordingly
    and a warning is printed; the JSON itself is left untouched so the
    historical record of the original creation count is preserved.
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
"""

import json
from collections import Counter
from datetime import datetime

from google.cloud import storage
from sklearn.model_selection import train_test_split

from constants import BUCKET_NAME
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.run_config import RunConfig


class DataSplitter:
    """Stage 2 — carve out / load the locked val set, then split train / test."""

    GCS_VALIDATION_IDS_PATH = "splits/validation_ids.json"

    VAL_FRAC = 0.30                  # 30% of total -> df_val
    TEST_FRAC_OF_REMAINING = 0.20    # 20% of remaining 70% -> 14% of total

    def __init__(self, config: RunConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.client_ = storage.Client()
        self.bucket_ = self.client_.bucket(BUCKET_NAME)

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.SPLIT)

        df = run.df_videos.copy()

        before = len(df)
        df = df.dropna(subset=["vertical", "tier"])
        dropped = before - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with NaN vertical or tier")

        df["cell_"] = df["vertical"].astype(str) + "_" + df["tier"].astype(str)

        blob = self.bucket_.blob(self.GCS_VALIDATION_IDS_PATH)
        if blob.exists():
            df_train, df_test, df_val = self._load_and_split(df, blob)
        else:
            df_train, df_test, df_val = self._create_and_split(df, blob)

        run.df_train = df_train.drop(columns=["cell_"])
        run.df_test = df_test.drop(columns=["cell_"])
        run.df_val = df_val.drop(columns=["cell_"])
        return run

    def _create_and_split(self, df, blob):
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
        print(f"Will write to: gs://{BUCKET_NAME}/{self.GCS_VALIDATION_IDS_PATH}")
        print("Once written, this validation set is LOCKED FOREVER.")
        print()

        response = input("Proceed with write? Type 'yes' to confirm: ").strip().lower()
        if response != "yes":
            raise RuntimeError("Validation split creation aborted by user.")

        payload = {
            "created_at": datetime.now().isoformat(),
            "seed": self.seed,
            "total_val_rows": len(df_val),
            "rows_per_cell": rows_per_cell,
            "video_ids": df_val["video_id"].tolist(),
        }
        blob.upload_from_string(
            json.dumps(payload, indent=2),
            content_type="application/json",
        )
        print(f"Wrote gs://{BUCKET_NAME}/{self.GCS_VALIDATION_IDS_PATH}")

        return df_train, df_test, df_val

    def _load_and_split(self, df, blob):
        payload = json.loads(blob.download_as_text())
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
