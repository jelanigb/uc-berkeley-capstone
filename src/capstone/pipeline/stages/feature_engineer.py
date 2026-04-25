"""
FeatureEngineer — engineer features on each split, derive feature_cols,
fit a scaler on train, and produce X/y for train, test, and val.

Architecture
------------
FeatureEngineerLogic does the column-level work and is fit-free today
(`engineer_features` is a stack of stateless row-wise transforms). Its
contract is to apply identical transformations to all three splits and
return engineered DataFrames. If a stateful transform is added later
(target encoders, KNN imputers, etc.) it should be fit on df_train only
and applied to the others.

FeatureEngineer (outer) handles X/y split, feature_cols derivation, and
scaling. Scaling is applied to every model — RF and XGBoost are
scale-invariant so it does not hurt them, and head-to-head model
comparison is cleaner when every model sees the same X.

Bad-baseline rows
-----------------
A real row with NaN or 0 in baseline_median_views/likes/comments has a
corrupted target: `compute_target` falls back to baseline_engagement = 0,
so above_baseline degenerates to engagement_7d > 0. We drop those rows
in FeatureEngineerLogic *before* `engineer_features` runs so the bad
target is never produced in the first place.

Synthetic data is never present at this stage — it is added downstream
by SyntheticAugmenter, which applies the same drop policy in its own
domain.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_processing.feature_engineering import engineer_features
from pipeline.pipeline_run import PipelineRun, Stage
from pipeline.run_config import RunConfig


TARGET_COL_ = "above_baseline"

# Tier is genuinely ordinal — channel size grows S < M < L.
TIER_ORDER_ = ["S", "M", "L"]
TIER_ENCODING_ = {tier: i for i, tier in enumerate(TIER_ORDER_)}

# Vertical is nominal — one-hot to avoid imposing a false ordering on
# Education / Lifestyle / Tech (would distort LR coefficients).
VERTICAL_ORDER_ = ["Education", "Lifestyle", "Tech"]

# Baseline columns whose absence corrupts the target. See module docstring.
BASELINE_REQUIRED_COLS_ = [
    "baseline_median_views",
    "baseline_median_likes",
    "baseline_median_comments",
]

# Columns excluded from X. Kept on df_model for post-engineering EDA so
# plots can still group by vertical/tier and inspect raw baselines.
# Note: 7d-suffixed metric columns are added dynamically in `feature_cols`.
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


class FeatureEngineerLogic:
    """Stateless column-level feature engineering — testable with just DataFrames."""

    def run(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train_eng = self._engineer(df_train, label="train")
        df_test_eng = self._engineer(df_test, label="test")
        df_val_eng = self._engineer(df_val, label="val")
        return df_train_eng, df_test_eng, df_val_eng

    def feature_cols(self, df: pd.DataFrame) -> list:
        """Derive feature columns from an engineered df by exclusion."""
        seven_d = [c for c in df.columns if c.endswith("_7d")]
        excludes = set(EXCLUDE_COLS_) | set(seven_d)
        return [c for c in df.columns if c not in excludes]

    def _engineer(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        df = self._drop_bad_baselines(df, label=label)
        df = engineer_features(df)
        df = self._encode_categoricals(df)
        # TODO: revisit per-column fill strategy (e.g. median for thumbnail
        # features like brightness/colorfulness) once the data is more stable.
        df = self._fill_missing(df, fill_value=0)
        return df

    def _drop_bad_baselines(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        is_nan = df[BASELINE_REQUIRED_COLS_].isna().any(axis=1)
        is_zero = (df[BASELINE_REQUIRED_COLS_] == 0).any(axis=1)
        bad = is_nan | is_zero
        if bad.any():
            print(
                f"  {label}: dropped {int(bad.sum())} rows with NaN or 0 in "
                f"baseline_median_*"
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

        `subset=None` (or empty) means every column. Inf is mapped to NaN
        first so a future median fill works correctly.
        """
        df = df.copy()
        cols = df.columns.tolist() if not subset else subset
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        return df


class FeatureEngineer:
    """Stage 4 — outer wrapper. Builds X/y and fits the scaler on train."""

    def __init__(self, config: RunConfig, logic: FeatureEngineerLogic = None):
        self.config = config
        self.logic = logic or FeatureEngineerLogic()
        self.feature_cols_: list = []
        self.scaler_: StandardScaler = None

    def run(self, run: PipelineRun) -> PipelineRun:
        run.assert_ready_for(Stage.ENGINEER)

        df_train, df_test, df_val = self.logic.run(
            run.df_train, run.df_test, run.df_val
        )
        self.feature_cols_ = self.logic.feature_cols(df_train)

        X_train, y_train = self._split_xy(df_train)
        X_test, y_test = self._split_xy(df_test)
        X_val, y_val = self._split_xy(df_val)

        self.scaler_ = StandardScaler()
        X_train = self._scale(X_train, fit=True)
        X_test = self._scale(X_test, fit=False)
        X_val = self._scale(X_val, fit=False)

        run.X_train, run.y_train = X_train, y_train
        run.X_test, run.y_test = X_test, y_test
        run.X_val, run.y_val = X_val, y_val
        # df_model: real train, post-engineering, unscaled — for EDA grouping
        # by vertical/tier and inspecting features in original units.
        run.df_model = df_train
        return run

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df[self.feature_cols_].copy(), df[TARGET_COL_].copy()

    def _scale(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        arr = self.scaler_.fit_transform(X) if fit else self.scaler_.transform(X)
        return pd.DataFrame(arr, columns=self.feature_cols_, index=X.index)
