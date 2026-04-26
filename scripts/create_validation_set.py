#!/usr/bin/env python3
"""
create_validation_set.py

Pulls current data from BigQuery, writes a versioned raw GCS parquet
snapshot, and creates the locked validation holdout at
gs://<bucket>/splits/validation_ids.json.

Intended to be run ONCE, before the first training run, using the fullest
available dataset from BQ (not a stale GCS snapshot).

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
"""

import argparse
import os
import sys

# ── path setup ───────────────────────────────────────────────────────────────
# Script lives at repo/scripts/; source root is repo/src/capstone/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "capstone"))

from constants import BUCKET_NAME
from pipeline.pipeline_run import PipelineRun
from pipeline.stages.data_loader import DataLoader
from pipeline.stages.data_splitter import DataSplitter, GcsHoldoutStore, HoldoutStore
from pipeline.version_config import VersionConfig
from utils.snapshot_data import save_baselines_snapshot, save_video_snapshot


# ── dry-run holdout store ─────────────────────────────────────────────────────

class DryRunHoldoutStore(HoldoutStore):
    """In-memory store used during dry-run. Never touches GCS.

    Always reports not-exists so DataSplitter takes the create path,
    capturing the computed payload without writing it.
    """

    _GCS_PATH = f"gs://{BUCKET_NAME}/splits/validation_ids.json"

    def __init__(self):
        self.payload_ = None

    def exists(self) -> bool:
        return False

    def load(self) -> dict:
        raise RuntimeError("DryRunHoldoutStore.load() should never be called.")

    def save(self, payload: dict) -> None:
        self.payload_ = payload

    def location(self) -> str:
        return self._GCS_PATH


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


def print_plan(run: PipelineRun, dry_store: DryRunHoldoutStore, config: VersionConfig) -> None:
    """Print the full dry-run plan: BQ counts, GCS paths, split sizes, sample IDs."""

    _banner("DRY RUN — NO WRITES TO GCS")

    _section("BQ pull")
    print(f"  Videos:           {len(run.df_videos):>7,} rows")
    v_chans = run.df_videos['channel_id'].nunique() if 'channel_id' in run.df_videos.columns else '?'
    print(f"  Unique channels:  {v_chans:>7}")
    if 'vertical' in run.df_videos.columns:
        v_counts = run.df_videos['vertical'].value_counts().to_dict()
        for v, n in sorted(v_counts.items()):
            print(f"    {v}: {n:,}")
    print(f"  Baselines:        {len(run.df_baselines):>7,} rows")
    print(f"  Medians:          {len(run.df_medians):>7,} rows")

    _section(f"Raw snapshot  →  version '{config.raw_version}'  (would write)")
    ts_placeholder = "<YYYYmmdd_HHMMSS>"
    bucket = BUCKET_NAME
    print(f"  gs://{bucket}/snapshots/snapshots_{config.raw_version}_{len(run.df_videos)}rows_{ts_placeholder}.parquet")
    print(f"  gs://{bucket}/snapshots/snapshots_{config.raw_version}_{len(run.df_videos)}rows_{ts_placeholder}_meta.json")
    print(f"  gs://{bucket}/snapshots/baselines_{config.raw_version}_{len(run.df_baselines)}rows_{ts_placeholder}.parquet")
    print(f"  gs://{bucket}/snapshots/medians_{config.raw_version}_{len(run.df_medians)}rows_{ts_placeholder}.parquet")
    print(f"  gs://{bucket}/snapshots/baselines_{config.raw_version}_{ts_placeholder}_meta.json")

    payload = dry_store.payload_
    if payload:
        _section(f"Validation holdout  →  {dry_store.location()}  (would write)")
        total = len(run.df_videos)
        n_val   = len(run.df_val)
        n_train = len(run.df_train)
        n_test  = len(run.df_test)
        print(f"  df_val:    {n_val:>6,} rows  ({n_val / total:.1%})")
        print(f"  df_train:  {n_train:>6,} rows  ({n_train / total:.1%})")
        print(f"  df_test:   {n_test:>6,} rows  ({n_test / total:.1%})")

        print(f"\n  Rows per cell (vertical × tier):")
        for cell, count in sorted(payload["rows_per_cell"].items()):
            print(f"    {cell:<20}  {count:,}")

        sample_ids = payload["video_ids"][:10]
        print(f"\n  Sample validation video_ids (first 10 of {len(payload['video_ids']):,}):")
        for vid in sample_ids:
            print(f"    {vid}")

    _section("Version bump")
    cur_data = config.state_["data"]
    print(f"  raw_version:  v{cur_data['major']}.{cur_data['minor']}_{cur_data['raw_suffix']}  →  {config.raw_version}")
    print(f"  (config.commit() will be called on --yes run)")

    print(f"\n{'═' * 68}")
    print("  Re-run with --yes to execute all writes.")
    print(f"{'═' * 68}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pull BQ data, write raw GCS snapshot, create locked validation holdout.\n"
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
    # snapshot_raw() schedules a raw data minor bump; build() computes the
    # new version string but does NOT write to GCS yet.
    config = VersionConfig.load().snapshot_raw().build()
    run = PipelineRun(config)

    # ── guard: holdout must not already exist for this to make sense ────────
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
    loader = DataLoader(config, source=DataLoader.SOURCE_BQ)
    loader.run(run)

    # ── step 2: compute the split ───────────────────────────────────────────
    print("\nStep 2 — Computing validation split...")
    dry_store = DryRunHoldoutStore()
    # auto-confirm in script context (--yes is the confirmation; dry-run never writes)
    dry_splitter = DataSplitter(config, store=dry_store, confirm=lambda: True)
    dry_splitter.run(run)

    # ── always print the plan ───────────────────────────────────────────────
    print_plan(run, dry_store, config)

    if not live:
        sys.exit(0)

    # ── step 3: execute writes ───────────────────────────────────────────────
    print("Step 3 — Writing raw snapshot to GCS...")
    save_video_snapshot(run.df_videos, config.raw_version)
    save_baselines_snapshot(run.df_baselines, run.df_medians, config.raw_version)

    print("\nStep 4 — Writing validation holdout to GCS...")
    live_splitter = DataSplitter(config, store=real_store, confirm=lambda: True)
    live_splitter.run(run)

    print("\nStep 5 — Committing version bump...")
    config.commit()

    print(
        f"\n✅  Done.\n"
        f"    Raw snapshot:  {config.raw_version}\n"
        f"    Holdout:       {real_store.location()}\n"
        f"    Versions committed. Next VersionConfig.load() will see the new counters.\n"
    )


if __name__ == "__main__":
    main()
