"""
RawSnapshotter — persist run.df_videos, df_baselines, and df_medians to GCS.

Write-side counterpart to DataLoader. Gated by config.take_snapshot_raw;
a no-op when that flag is False so the notebook can always call it without
branching. Delegates to utils.snapshot_data.save_video_snapshot and
save_baselines_snapshot — does not re-pull from BigQuery.

Never modifies PipelineRun state — reads only.
"""

from pipeline.pipeline_run import PipelineRun
from pipeline.version_config import VersionConfig
from utils.snapshot_data import save_baselines_snapshot, save_video_snapshot


class RawSnapshotter:
    """Persist raw-data frames from a completed DataLoader run to GCS."""

    def __init__(self, config: VersionConfig):
        self.config = config

    def run(self, run: PipelineRun) -> PipelineRun:
        if not self.config.take_snapshot_raw:
            print("[RawSnapshotter] take_snapshot_raw=False — skipping.")
            return run

        missing = [
            f for f in ("df_videos", "df_baselines", "df_medians")
            if getattr(run, f) is None
        ]
        if missing:
            raise RuntimeError(
                f"RawSnapshotter: run fields {missing} are None. "
                "Did DataLoader.run() complete?"
            )

        version_tag = self.config.raw_version
        save_video_snapshot(run.df_videos, version_tag)
        save_baselines_snapshot(run.df_baselines, run.df_medians, version_tag)
        print(f"[RawSnapshotter] Raw snapshot '{version_tag}' saved to GCS.")
        return run
