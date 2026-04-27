"""
DataPreprocessor — pivot long-format snapshots to wide, join baseline medians.

Sits between DataLoader and FeatureEngineer. DataLoader returns raw long-format
data (1 row per (video_id, poll_label)); DataPreprocessor collapses those rows
into a single wide row per completed-triplet video and joins the channel baseline
medians needed to compute the target.

Delegates entirely to `build_clean_dataset` in data_processing.data_cleanup —
no transformation logic lives here. That function:

  1. pivot_snapshots  — drops videos missing any of the 3 poll labels and
                        produces wide columns: view_count_upload, view_count_24h,
                        view_count_7d, etc.
  2. join_baselines   — left-joins channel median baseline values (prefixed
                        baseline_*) needed by compute_target downstream.
  3. clean_data       — structural cleanup (whitespace, type fixes, clamp negatives).

After this stage, `run.df_clean` holds 1 row per video, baseline columns present,
ready for FeatureEngineer.
"""

from data_processing.data_cleanup import build_clean_dataset
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import VersionConfig


class DataPreprocessor:
    """Stage 2 — pivot + baseline-join + structural cleanup."""

    def __init__(self, config: VersionConfig):
        self.config = config

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.PREPROCESS)
        run.df_clean = build_clean_dataset(run.df_videos, run.df_medians)
        return run
