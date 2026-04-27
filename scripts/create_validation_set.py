#!/usr/bin/env python3
"""
create_validation_set.py

Pulls current data from BigQuery, runs the full data-preparation and
feature-engineering pipeline, creates the locked validation holdout at
gs://<bucket>/splits/validation_ids.json, and writes a versioned raw GCS
parquet snapshot.

Intended to be run ONCE, before the first training run. Uses the fullest
available dataset from BQ (not a stale GCS snapshot) so the holdout is
stratified on the maximum possible rows.

Stratification key: vertical × tier × above_baseline (18 cells) — the same
key used by DataSplitter at runtime, so all future train/test splits are
consistent with the holdout.

Usage
-----
    # Dry run (default) — compute plan, print counts + sample IDs, write nothing:
    python scripts/create_validation_set.py

    # Explicit dry run (same as above):
    python scripts/create_validation_set.py --dry-run

    # Execute writes to GCS:
    python scripts/create_validation_set.py --yes

Dry run always runs first (even with --yes) so the plan is printed before
any GCS write begins.

# TODO(print-flag-cleanup): dry-run currently prints some messages that are
# also printed by live mode; consolidate so each message appears once and
# only under the correct flag.
"""

import argparse
import os
import sys

# ── path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "capstone"))

from constants import BUCKET_NAME
from pipeline.pipeline_run import PipelineRun
from pipeline.stages.data_loader import DataLoader
from pipeline.stages.data_preprocessor import DataPreprocessor
from pipeline.stages.feature_engineer import FeatureEngineer
from pipeline.stages.data_splitter import (
    GcsHoldoutStore,
    InMemoryHoldoutStore,
    create_holdout,
)
from pipeline.version_config import VersionConfig
from utils.snapshot_data import save_baselines_snapshot, save_video_snapshot


# ── formatting helpers ────────────────────────────────────────────────────────

_LINE = "─" * 68

def _banner(title: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {title}")
    print(f"{'═' * 68}")

def _section(title: str) -> None:
    print(f"\n{_LINE}")
    print(f"  {title}")
    print(_LINE)


def print_plan(
    run: PipelineRun,
    dry_store: InMemoryHoldoutStore,
    config: VersionConfig,
    live: bool = False,
) -> None:
    """Print the full dry-run plan: BQ counts, GCS paths, split sizes, sample IDs."""

    if live:
        _banner("LIVE RUN - WRITES TO GCS")    
    else:
        _banner("DRY RUN — NO WRITES TO GCS")
    

    _section("BQ pull + preprocess")
    print(f"  Videos (raw poll rows):   {len(run.df_videos):>7,}")
    v_chans = run.df_videos['channel_id'].nunique() if 'channel_id' in run.df_videos.columns else '?'
    print(f"  Unique channels:          {v_chans:>7}")
    print(f"  Baselines:                {len(run.df_baselines):>7,} rows")
    print(f"  Medians:                  {len(run.df_medians):>7,} rows")

    if run.df_clean is not None:
        print(f"  Post-pivot (1/video):     {len(run.df_clean):>7,}")
    if run.df_engineered is not None:
        print(f"  Post-engineering:         {len(run.df_engineered):>7,}")
        if 'vertical' in run.df_engineered.columns:
            v_counts = run.df_engineered['vertical'].value_counts().to_dict()
            for v, n in sorted(v_counts.items()):
                print(f"    {v}: {n:,}")

    _section(f"Raw snapshot  →  version '{config.raw_version}'. Will {'' if live else 'not'} write to GCS")
    ts_placeholder = "<YYYYmmdd_HHMMSS>"
    bucket = BUCKET_NAME
    nrows = len(run.df_videos) if run.df_videos is not None else "?"
    nb = len(run.df_baselines) if run.df_baselines is not None else "?"
    nm = len(run.df_medians) if run.df_medians is not None else "?"
    print(f"{'Writing:' if live else 'Simulating:'}")
    print(f"  gs://{bucket}/snapshots/snapshots_{config.raw_version}_{nrows}rows_{ts_placeholder}.parquet")
    print(f"  gs://{bucket}/snapshots/snapshots_{config.raw_version}_{nrows}rows_{ts_placeholder}_meta.json")
    print(f"  gs://{bucket}/snapshots/baselines_{config.raw_version}_{nb}rows_{ts_placeholder}.parquet")
    print(f"  gs://{bucket}/snapshots/medians_{config.raw_version}_{nm}rows_{ts_placeholder}.parquet")
    print(f"  gs://{bucket}/snapshots/baselines_{config.raw_version}_{ts_placeholder}_meta.json")

    payload = dry_store.payload_
    if payload:
        _section(f"Validation holdout  →  {dry_store.location()}. Will {'' if live else 'not'} write to GCS")
        total = len(run.df_engineered)
        n_val = payload["total_val_rows"]
        n_remaining = total - n_val
        print(f"  df_val:       {n_val:>6,} rows  ({n_val / total:.1%})")
        print(f"  df_remaining: {n_remaining:>6,} rows  ({n_remaining / total:.1%})")

        print(f"\n  Rows per cell (vertical × tier × above_baseline):")
        for cell, count in sorted(payload["rows_per_cell"].items()):
            print(f"    {cell:<30}  {count:,}")

        sample_ids = payload["video_ids"][:10]
        print(f"\n  Sample validation video_ids (first 10 of {len(payload['video_ids']):,}):")
        for vid in sample_ids:
            print(f"    {vid}")

    _section(f"{'' if live else 'simulated'} Version bump")
    cur_data = config.state_["data"]
    print(f"  raw_version:  v{cur_data['major']}.{cur_data['minor']}_{cur_data['raw_suffix']}  →  {config.raw_version}")
    print(f"  (config.commit() will be called if --yes flag is used)")

    plan_msg_ = ('Will write to GCS' if live else 'Dry run; will not write to GCS')
    _banner(f"Plan complete. {plan_msg_}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pull BQ data, engineer features, create locked validation holdout.\n"
            "Default is dry-run; pass --yes to execute writes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--yes",
        action="store_true",
        help="Execute all GCS writes. Dry-run plan is printed first.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Compute and print plan without writing anything (default).",
    )
    args = parser.parse_args()
    live = args.yes

    # ── config ──────────────────────────────────────────────────────────────
    config = VersionConfig.load().snapshot_raw().build()
    run = PipelineRun(config)

    # ── guard: holdout must not already exist ────────────────────────────────
    real_store = GcsHoldoutStore()
    if real_store.exists():
        print(
            f"\n⚠️  Holdout already exists at {real_store.location()}.\n"
            "    This script is intended for first-time holdout creation only.\n"
            "    To re-validate against existing models, use the pipeline's\n"
            "    validate_current scenario in debug.ipynb instead.\n"
        )
        if live:
            sys.exit(1)
        else:
            print("    (Continuing dry run to show what a NEW split would look like.)\n")

    # ── step 1: pull from BQ ────────────────────────────────────────────────
    print("\nStep 1 — Loading data from BigQuery...")
    DataLoader(config, source=DataLoader.SOURCE_BQ).run(run)
    print(f"  → {len(run.df_videos):,} video poll rows, {len(run.df_medians):,} baseline median rows")

    # ── step 2: preprocess (pivot + baseline-join) ───────────────────────────
    print("\nStep 2 — Preprocessing (pivot + baseline join)...")
    DataPreprocessor(config).run(run)
    print(f"  → {len(run.df_clean):,} rows in df_clean (one per complete-triplet video)")

    # ── step 3: feature engineering ──────────────────────────────────────────
    print("\nStep 3 — Engineering features...")
    FeatureEngineer(config).run(run)
    print(f"  → {len(run.df_engineered):,} rows in df_engineered")

    # ── step 4: dry-run holdout split ────────────────────────────────────────
    print("\nStep 4 — Computing validation split (dry run)...")
    dry_store = InMemoryHoldoutStore()
    create_holdout(
        run.df_engineered,
        frac=0.30,
        store=dry_store,
        seed=42,
        confirm=None,  # no prompt in script context
    )

    # ── always print the plan ───────────────────────────────────────────────
    print_plan(run, dry_store, config, live)

    if not live:
        sys.exit(0)

    # ── step 5: write raw snapshot ───────────────────────────────────────────
    print("Step 5 — Writing raw snapshot to GCS...")
    save_video_snapshot(run.df_videos, config.raw_version)
    save_baselines_snapshot(run.df_baselines, run.df_medians, config.raw_version)

    # ── step 6: write validation holdout ────────────────────────────────────
    print("\nStep 6 — Writing validation holdout to GCS...")
    create_holdout(
        run.df_engineered,
        frac=0.30,
        store=real_store,
        seed=42,
        confirm=None,  # --yes is the confirmation
    )

    # ── step 7: commit version bump ──────────────────────────────────────────
    print("\nStep 7 — Committing version bump...")
    config.commit()

    print(
        f"\n✅  Done.\n"
        f"    Raw snapshot:  {config.raw_version}\n"
        f"    Holdout:       {real_store.location()}\n"
        f"    Versions committed. Next VersionConfig.load() will see the new counters.\n"
    )


if __name__ == "__main__":
    main()
