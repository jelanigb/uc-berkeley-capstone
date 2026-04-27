"""
DataLoader — read-only stage that populates `run.df_videos`, `run.df_baselines`,
and `run.df_medians` from either BigQuery or a GCS parquet snapshot.

Source selection
----------------
The factory wires `DataLoader(config, source=...)` with `source="bq"` for
`full_run` and `source="gcs"` for every other scenario. Keeping source as an
explicit constructor argument (rather than deriving it from a `scenario` field
on `VersionConfig`) preserves the orthogonality between version intent and
pipeline topology — the factory is the single source of truth for which
scenario implies which source.

Read-only
---------
Writing back to GCS is the responsibility of `RawSnapshotter` (next pass),
gated by `config.take_snapshot_raw`. `DataLoader` never writes.

Delegation
----------
GCS reads delegate to `utils.snapshot_data.load_videos` / `load_baselines`,
which are the source of truth for the GCS layout. BQ reads use the queries
exposed by `utils.snapshot_data` (`BASELINE_QUERY`, `BASELINE_MEDIANS_QUERY`)
plus an inline video query that mirrors the one inside
`utils.snapshot_data.snapshot_video_data` (the side-effecting writer).
"""

import pandas as pd
from google.cloud import bigquery

from constants import PROJECT_ID
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import VersionConfig
from utils.snapshot_data import (
    BASELINE_QUERY,
    BASELINE_MEDIANS_QUERY,
    DATASET_ID,
    load_baselines,
    load_videos,
)


VIDEO_QUERY_ = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.video_snapshots`
    WHERE poll_label IS NOT NULL
"""


class DataLoader:
    """Stage 1 — read videos, baselines, and medians into `PipelineRun`."""

    SOURCE_BQ = "bq"
    SOURCE_GCS = "gcs"
    VALID_SOURCES_ = (SOURCE_BQ, SOURCE_GCS)

    def __init__(self, config: VersionConfig, source: str):
        if source not in self.VALID_SOURCES_:
            raise ValueError(
                f"DataLoader source must be one of {self.VALID_SOURCES_}, "
                f"got {source!r}."
            )
        self.config = config
        self.source = source

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.LOAD)

        if self.source == self.SOURCE_BQ:
            df_videos, df_baselines, df_medians = self._load_from_bq()
        else:
            df_videos, df_baselines, df_medians = self._load_from_gcs(
                self.config.raw_version
            )

        run.df_videos = df_videos
        run.df_baselines = df_baselines
        run.df_medians = df_medians
        return run

    def _load_from_bq(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        bq_client = bigquery.Client(project=PROJECT_ID)

        print(f"Pulling video_snapshots from BQ ({PROJECT_ID}.{DATASET_ID})...")
        df_videos = bq_client.query(VIDEO_QUERY_).to_dataframe()
        print(f"  {len(df_videos)} video rows")

        print("Pulling channel_baseline_videos from BQ...")
        df_baselines = bq_client.query(BASELINE_QUERY).to_dataframe()
        print(
            f"  {len(df_baselines)} baseline rows "
            f"({df_baselines['channel_id'].nunique()} channels)"
        )

        print("Pulling channel_baseline_medians from BQ...")
        df_medians = bq_client.query(BASELINE_MEDIANS_QUERY).to_dataframe()
        print(f"  {len(df_medians)} baseline median rows")

        return df_videos, df_baselines, df_medians

    def _load_from_gcs(
        self,
        raw_version: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_videos, _ = load_videos(raw_version)
        df_baselines, df_medians, _ = load_baselines(raw_version)
        return df_videos, df_baselines, df_medians
