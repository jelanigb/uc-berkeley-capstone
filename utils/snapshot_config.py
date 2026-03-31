"""
SnapshotConfig — builder-pattern orchestrator for all GCS snapshot operations.

Manages snapshot flags, tuning flags, default hyperparameters, and version
numbers in one place. Reads/writes `config/versions.json` in GCS as the
single source of truth for the current version state.

Usage (notebook config cell):

    from utils.snapshot_config import SnapshotConfig

    # Use a preset:
    config = SnapshotConfig.load().preset("feature_engineering").build()

    # Or chain methods explicitly:
    config = SnapshotConfig.load().snapshot_mixed().snapshot_models().tune().build()

    # Dry run (no snapshots, no tuning — just loads current versions for reading):
    config = SnapshotConfig.load().build()

Available presets:
    "dry_run"             — nothing; loads current versions for reading
    "model_tuning"        — tune + models + hyperparams        → minor bump
    "feature_engineering" — mixed + tune + models + hyperparams → minor bump
    "new_raw_data"        — all five                           → major bump

After all snapshots succeed, call config.commit() to persist the new version
number back to GCS so the next session picks it up automatically.
"""

import json
import re
from datetime import datetime

from google.cloud import storage

PROJECT_ID     = "maduros-dolce"
BUCKET_NAME    = "maduros-dolce-capstone-data"
VERSIONS_BLOB_ = "config/versions.json"

PRESETS_ = {
    "dry_run":             [],
    "model_tuning":        ["tune", "models", "hyperparams"],
    "feature_engineering": ["mixed", "tune", "models", "hyperparams"],
    "new_raw_data":        ["raw", "mixed", "tune", "models", "hyperparams"],
}

# Default hyperparameters — used when no GCS hyperparam snapshot exists yet
DEFAULT_LR_PARAMS_ = {
    'C': 1.0, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 5000,
}
DEFAULT_RF_PARAMS_ = {
    'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 5,
    'min_samples_split': 2, 'max_features': 'sqrt',
}
DEFAULT_XGB_PARAMS_ = {
    'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
}

# Default state used when no versions.json exists in GCS yet
DEFAULT_STATE_ = {
    "version":      "3.1",
    "raw_suffix":   "real",
    "mixed_suffix": "mixed_80real",
    "last_updated": None,
    "last_snapshot_types": [],
}


class SnapshotConfig:

    def __init__(self, state: dict):
        self.state_        = state
        self.flags_        = {k: False for k in ["raw", "mixed", "tune", "models", "hyperparams"]}
        self.built_        = False
        self.new_version_  = None
        self.search_strategy = "random"  # default; override with .tune(strategy=...)
        self.search_n_iter   = 50
        self.search_cv       = 5
        self.search_scoring  = "roc_auc"
        self.new_grids       = {}        # model class name → param grid override

        # Public version strings — set by build()
        self.raw_version        = None
        self.mixed_version      = None
        self.model_version      = None
        self.hyperparam_version = None

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def load(cls) -> "SnapshotConfig":
        """
        Load current version state from GCS.
        Falls back to defaults if no versions.json exists yet.
        """
        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)
        blob = bucket.blob(VERSIONS_BLOB_)

        if blob.exists():
            state = json.loads(blob.download_as_text())
        else:
            state = DEFAULT_STATE_.copy()
            print(
                f"No versions.json found in GCS — using defaults (v{DEFAULT_STATE_['version']}). "
                "Call config.commit() after your first snapshot to persist versions."
            )

        print(
            f"SnapshotConfig loaded: v{state['version']} "
            f"(raw_suffix='{state['raw_suffix']}', mixed_suffix='{state['mixed_suffix']}')"
        )
        return cls(state)

    # =========================================================================
    # Builder methods — each returns self for chaining
    # =========================================================================

    def snapshot_raw(self, suffix: str = None) -> "SnapshotConfig":
        """Mark raw BQ pull for snapshotting. Triggers a major version bump."""
        self.flags_["raw"] = True
        if suffix:
            self.state_["raw_suffix"] = suffix
        return self

    def snapshot_mixed(self, suffix: str = None) -> "SnapshotConfig":
        """Mark mixed (real + synthetic) dataset for snapshotting."""
        self.flags_["mixed"] = True
        if suffix:
            self.state_["mixed_suffix"] = suffix
        return self

    def snapshot_models(self) -> "SnapshotConfig":
        """Mark all three model snapshots."""
        self.flags_["models"] = True
        return self

    def snapshot_hyperparams(self) -> "SnapshotConfig":
        """Mark hyperparameter set for snapshotting."""
        self.flags_["hyperparams"] = True
        return self

    def tune(
        self,
        strategy: str = "random",
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = "roc_auc",
        new_grids: dict = None,
    ) -> "SnapshotConfig":
        """
        Enable hyperparameter tuning and configure the search.

        Parameters
        ----------
        strategy  : "random" | "halving" | "grid"
        n_iter    : number of random samples (RandomizedSearchCV only)
        cv        : cross-validation folds
        scoring   : sklearn scoring string
        new_grids : optional dict of {model_class_name: param_grid} overrides.
                    Any model not listed here uses get_default_param_grid().
                    Example:
                        new_grids={'XGBClassifier': {'max_depth': [3, 4, 5],
                                                     'learning_rate': [0.05, 0.1]}}
        """
        self.flags_["tune"] = True
        self.search_strategy = strategy
        self.search_n_iter   = n_iter
        self.search_cv       = cv
        self.search_scoring  = scoring
        self.new_grids       = new_grids or {}
        return self

    def preset(self, name: str) -> "SnapshotConfig":
        """
        Apply a named preset. See module docstring for available presets.
        Can be combined with explicit flag methods.

        Note: presets that include "tune" use default search settings
        (random, n_iter=50, cv=5). Chain .tune(...) after to override.
        """
        if name not in PRESETS_:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(PRESETS_.keys())}"
            )
        for flag in PRESETS_[name]:
            if flag == "tune":
                self.flags_["tune"] = True   # search settings stay at defaults
            else:
                self.flags_[flag] = True
        return self

    # =========================================================================
    # Build — computes version strings, does NOT write to GCS
    # =========================================================================

    def build(self) -> "SnapshotConfig":
        """
        Compute new version strings based on active flags.

        Bumping rules:
          - Raw snapshot active → major bump (X.Y → X+1.0)
          - Any other flag active → minor bump (X.Y → X.Y+1)
          - No flags active → version unchanged (dry run)
        """
        current = self.state_["version"]
        major, minor = self.parse_version_(current)
        any_active = any(self.flags_.values())

        if self.flags_["raw"]:
            new_version = f"{major + 1}.0"
        elif any_active:
            new_version = f"{major}.{minor + 1}"
        else:
            new_version = current

        raw_sfx   = self.state_["raw_suffix"]
        mixed_sfx = self.state_["mixed_suffix"]

        self.raw_version        = f"v{new_version}_{raw_sfx}"
        self.mixed_version      = f"v{new_version}_{mixed_sfx}"
        self.model_version      = f"v{new_version}"
        self.hyperparam_version = f"v{new_version}"
        self.new_version_       = new_version
        self.built_             = True

        active = [k for k, v in self.flags_.items() if v]
        print("\nSnapshotConfig ready:")
        print(f"  Active flags      : {active if active else ['none (dry run)']}")
        print(f"  Version           : v{current} → v{new_version}")
        print(f"  raw_version       : {self.raw_version}")
        print(f"  mixed_version     : {self.mixed_version}")
        print(f"  model_version     : {self.model_version}")
        print(f"  hyperparam_version: {self.hyperparam_version}")
        if self.flags_["tune"]:
            print(f"  Tuning            : strategy={self.search_strategy}, "
                  f"n_iter={self.search_n_iter}, cv={self.search_cv}, "
                  f"scoring={self.search_scoring}")
        if any_active:
            print("\n  Call config.commit() after all snapshots succeed.")

        return self

    # =========================================================================
    # Commit — write updated versions.json back to GCS
    # =========================================================================

    def commit(self):
        """
        Persist the new version number to GCS versions.json.
        Call this AFTER all snapshot operations complete successfully.
        """
        if not self.built_:
            raise RuntimeError("Call .build() before .commit().")
        if not any(self.flags_.values()):
            print("No snapshots were active — nothing to commit.")
            return

        new_state = {
            "version":             self.new_version_,
            "raw_suffix":          self.state_["raw_suffix"],
            "mixed_suffix":        self.state_["mixed_suffix"],
            "last_updated":        datetime.utcnow().isoformat(),
            "last_snapshot_types": [k for k, v in self.flags_.items() if v],
        }

        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)
        bucket.blob(VERSIONS_BLOB_).upload_from_string(
            json.dumps(new_state, indent=2),
            content_type="application/json",
        )
        print(f"Committed versions.json → v{self.new_version_}")

    # =========================================================================
    # Flag properties
    # =========================================================================

    @property
    def take_snapshot_raw(self) -> bool:
        return self.flags_["raw"]

    @property
    def take_snapshot_mixed(self) -> bool:
        return self.flags_["mixed"]

    @property
    def take_snapshot_models(self) -> bool:
        return self.flags_["models"]

    @property
    def take_snapshot_hyperparams(self) -> bool:
        return self.flags_["hyperparams"]

    @property
    def tune_models(self) -> bool:
        return self.flags_["tune"]

    @property
    def search_config(self) -> dict:
        """Returns the search config dict suitable for passing to save_hyperparams."""
        return {
            "strategy": self.search_strategy,
            "n_iter":   self.search_n_iter,
            "cv":       self.search_cv,
            "scoring":  self.search_scoring,
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def parse_version_(version_str: str) -> tuple:
        m = re.match(r"^(\d+)\.(\d+)$", str(version_str).strip())
        if not m:
            raise ValueError(
                f"Cannot parse version '{version_str}'. Expected format 'X.Y' (e.g. '3.1')."
            )
        return int(m.group(1)), int(m.group(2))

    def __repr__(self) -> str:
        if self.built_:
            active = [k for k, v in self.flags_.items() if v]
            return f"SnapshotConfig(v{self.new_version_}, active={active})"
        return "SnapshotConfig(not built — call .build())"
