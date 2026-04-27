"""
FeatureEngineer — derive all features on a single cleaned DataFrame.

Architecture
------------
FeatureEngineerLogic does the column-level work:
  - drop rows with bad (NaN/0) baseline medians
  - call engineer_features (computes target + all derived features)
  - encode categoricals (tier ordinal, vertical one-hots)
  - fill NaN/inf with 0

FeatureEngineer (outer) is a thin stage wrapper: reads run.df_clean,
calls logic.engineer, writes run.df_engineered. No split awareness, no
scaling — those are DataSplitter and Scaler's responsibility.

Bad-baseline rows
-----------------
A row with NaN in baseline_median_views/likes/comments is bad data; targets are
dependnet on baseline engagement data and cannot be calculated for these rows.
Likewise if a channel has a median_engagement_rate of exactly 0, then it should 
be excluded (as ANY engagement will put the channel above median). We
drop those rows before engineer_features runs. This is a defensive check;
DataPreprocessor's left-join already surfaces channels with no baseline
match as NaN in those columns.

derive_feature_cols (module-level)
-----------------------------------
Returns the list of feature columns from an engineered df by excluding
target, IDs, timestamps, raw baselines, and 7d-suffixed metric columns.
Exported here (not in DataSplitter) because the exclusion list is
conceptually part of the feature engineering contract. DataSplitter
imports and calls it to derive X column names without re-implementing the
logic.
"""

import numpy as np
import pandas as pd

from data_processing.feature_engineering import engineer_features
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.version_config import VersionConfig


TARGET_COL_ = "above_baseline"

# Tier is genuinely ordinal — channel size grows S < M < L.
TIER_ORDER_ = ["S", "M", "L"]
TIER_ENCODING_ = {tier: i for i, tier in enumerate(TIER_ORDER_)}

# Vertical is nominal — one-hot to avoid imposing a false ordering on
# Education / Lifestyle / Tech (would distort LR coefficients).
VERTICAL_ORDER_ = ["Education", "Lifestyle", "Tech"]

# Columns where NaN corrupts the target computation.
BASELINE_NON_NAN_COLS_ = [
    "baseline_median_views",
    "baseline_median_likes",
    "baseline_median_comments",
    "baseline_median_engagement_rate",
]

# Columns excluded from X. Kept on df for post-engineering EDA so
# plots can still group by vertical/tier and inspect raw baselines.
EXCLUDE_COLS_ = [
    # IDs and text
    "video_id", "channel_id", "channel_handle", "title", "description", "tags",
    "category_id", "category_name",
    # Timestamps
    "published_at", "poll_timestamp_upload", "poll_timestamp_24h", "poll_timestamp_7d",
    # Target and intermediate
    "above_baseline", "engagement_7d", "baseline_engagement",
    # Metadata flags
    "contains_synthetic_data", "contains_synthetic_media",
    # Raw baseline values — encoded into baseline_engagement; raw cols would leak.
    "baseline_channel_handle", "baseline_video_count",
    "baseline_median_views", "baseline_median_likes",
    "baseline_median_comments", "baseline_median_engagement_rate",
    # String categoricals — use tier_encoded and vertical_* one-hots instead.
    "vertical", "tier",
    # Redundant with duration_seconds
    "duration_minutes",
    # Harvest-process metadata, not predictive
    "hours_since_publish_upload", "hours_since_publish_24h",
    # Superseded by has_face (binary flag is the more reliable signal)
    "face_count",
]


def derive_feature_cols(df: pd.DataFrame) -> list:
    """Return the list of feature columns from an engineered DataFrame.

    Excludes the columns listed in EXCLUDE_COLS_ plus any column whose name
    ends in '_7d' (raw 7-day metric values that would leak the target).

    Single source of truth imported by DataSplitter to derive X column names.
    """
    seven_d = [c for c in df.columns if c.endswith("_7d")]
    excludes = set(EXCLUDE_COLS_) | set(seven_d)
    return [c for c in df.columns if c not in excludes]


class FeatureEngineerLogic:
    """Stateless column-level feature engineering — testable with just DataFrames.

    Public so SyntheticAugmenter can route new rows through the same chain:
    engineer() is idempotent for already-engineered columns because each step
    overwrites derived columns from the same raw source columns. Categorical
    encoding and fill-missing are NOT applied inside the synth generator, so
    running engineer() on synth rows correctly aligns them with real X_train.
    """

    def engineer(self, df: pd.DataFrame, label: str = "external") -> pd.DataFrame:
        """Apply the full engineering chain to a single DataFrame."""
        df = self._drop_bad_baselines(df, label=label)
        df = engineer_features(df)
        df = self._encode_categoricals(df)
        df = self._fill_missing(df, fill_value=0)
        return df

    def _drop_bad_baselines(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        is_nan = df[BASELINE_NON_NAN_COLS_].isna().any(axis=1)
        is_zero = df['baseline_median_engagement_rate'] == 0.0
        bad = is_nan | is_zero
        if bad.any():
            print(
                f"  {label}: dropped {int(bad.sum())} rows with NaN in a "
                f"baseline_median_* column or 0.0 baseline_median_engagement_rate"
            )
        return df.loc[~bad].copy()

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tier_encoded"] = df["tier"].map(TIER_ENCODING_)
        for vertical in VERTICAL_ORDER_:
            df[f"vertical_{vertical}"] = (df["vertical"] == vertical).astype(int)
        return df

    def _fill_missing(
        self,
        df: pd.DataFrame,
        fill_value=0,
        subset: list = None,
    ) -> pd.DataFrame:
        """Fill NaN/inf in `subset` columns with `fill_value`.

        `subset=None` means every column. Inf is mapped to NaN first so a
        future median fill works correctly. Nullable boolean columns (BooleanDtype)
        are cast to Int8 first so fillna(0) is accepted.
        """
        df = df.copy()
        cols = df.columns.tolist() if not subset else subset
        bool_cols = [c for c in cols if isinstance(df[c].dtype, pd.BooleanDtype)]
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype(pd.Int8Dtype())
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        return df


class FeatureEngineer:
    """Stage 3 — engineer features on df_clean, write df_engineered."""

    def __init__(self, config: VersionConfig, logic: FeatureEngineerLogic = None):
        self.config = config
        self.logic = logic or FeatureEngineerLogic()

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.ENGINEER)
        run.df_engineered = self.logic.engineer(run.df_clean, label="all")
        return run
