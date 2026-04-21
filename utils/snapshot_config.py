"""
SnapshotConfig — builder-pattern orchestrator for all GCS snapshot operations.

Tracks three independently-versioned entities — base data, model, and
hyperparameters — each with its own major.minor counter in GCS at
`config/versions.json`. See docs/versioning.md for the full semantics.

Bump rules (summary):
    Base Data
      .snapshot_raw() / .snapshot_mixed()        → data minor bump (new pull, same schema)
      .snapshot_schema_change()                  → data major bump (columns changed)
    Model
      .snapshot_models()                         → model minor bump (props / hyperparams changed)
      .snapshot_models_new_data()                → model major bump (retrained on new data)
    Hyperparameters
      .snapshot_hyperparams()                    → hyperparams minor bump (values tweaked)
      .snapshot_hyperparams_new_grid()           → hyperparams major bump (new param added to grid)

Typical usage (notebook config cell):

    from utils.snapshot_config import SnapshotConfig

    config = (
        SnapshotConfig.load()
        .snapshot_models()
        .snapshot_hyperparams()
        .use_data_version("3.1")
        .build()
    )

After all snapshots succeed, call config.commit() to persist the new version
numbers back to GCS so the next session picks them up automatically.
"""

import copy
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
    "new_raw_data":        ["raw", "mixed", "tune", "models_new_data", "hyperparams"],
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
DEFAULT_ENSEMBLE_PARAMS_ = {
    'voting': 'soft',
}

# Default state used when no versions.json exists in GCS yet.
# Each entity carries its own major.minor counter.
DEFAULT_STATE_ = {
    "data": {
        "major":        3,
        "minor":        1,
        "raw_suffix":   "real",
        "mixed_suffix": "mixed_80real",
    },
    "model":       {"major": 3, "minor": 1},
    "hyperparams": {"major": 1, "minor": 0},
    "last_updated":        None,
    "last_snapshot_types": [],
}


class SnapshotConfig:

    def __init__(self, state: dict):
        self.state_ = state
        self.flags_ = {k: False for k in [
            "raw", "mixed", "data_major",
            "models", "model_major",
            "hyperparams", "hyperparams_major",
            "tune",
        ]}
        self.built_           = False
        self.search_strategy  = "random"  # default; override with .tune(strategy=...)
        self.search_n_iter    = 50
        self.search_cv        = 5
        self.search_scoring   = "roc_auc"
        self.new_grids        = {}        # model class name → param grid override

        # Pinning — set by use_*_version methods
        self.pinned_data_version_       = None
        self.pinned_model_version_      = None
        self.pinned_hyperparam_version_ = None

        # Computed new versions (major, minor) per entity — set by build()
        self.new_data_       = None
        self.new_model_      = None
        self.new_hyperparams_ = None

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

        Handles one-time migration from the legacy flat-schema versions.json
        (single `"version"` key) to the new nested schema with independent
        data/model/hyperparams counters. The migration runs in-memory only;
        GCS is rewritten on the next commit().
        """
        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)
        blob = bucket.blob(VERSIONS_BLOB_)

        if blob.exists():
            state = json.loads(blob.download_as_text())
            if "version" in state and "data" not in state:
                state = cls.migrate_flat_to_nested_(state)
                print(
                    "Migrated legacy versions.json → nested schema "
                    "(data/model/hyperparams). Next commit() will persist the new shape."
                )
        else:
            state = copy.deepcopy(DEFAULT_STATE_)
            print(
                "No versions.json found in GCS — using defaults. "
                "Call config.commit() after your first snapshot to persist versions."
            )

        d = state["data"]
        m = state["model"]
        h = state["hyperparams"]
        print(
            f"SnapshotConfig loaded:\n"
            f"  data:        v{d['major']}.{d['minor']} "
            f"(raw_suffix='{d['raw_suffix']}', mixed_suffix='{d['mixed_suffix']}')\n"
            f"  model:       v{m['major']}.{m['minor']}\n"
            f"  hyperparams: v{h['major']}.{h['minor']}"
        )
        return cls(state)

    @staticmethod
    def migrate_flat_to_nested_(flat: dict) -> dict:
        """
        Convert legacy {"version": "3.1", "raw_suffix": ..., "mixed_suffix": ...}
        to the new nested schema. Data and model inherit the flat version;
        hyperparams bootstraps to v1.0 (starting defaults under the new scheme).
        """
        major, minor = SnapshotConfig.parse_version_(flat["version"])
        return {
            "data": {
                "major":        major,
                "minor":        minor,
                "raw_suffix":   flat.get("raw_suffix",   "real"),
                "mixed_suffix": flat.get("mixed_suffix", "mixed_80real"),
            },
            "model":       {"major": major, "minor": minor},
            "hyperparams": {"major": 1,     "minor": 0},
            "last_updated":        flat.get("last_updated"),
            "last_snapshot_types": flat.get("last_snapshot_types", []),
        }

    # =========================================================================
    # Builder methods — each returns self for chaining
    # =========================================================================

    def snapshot_raw(self, suffix: str = None) -> "SnapshotConfig":
        """Mark raw BQ pull for snapshotting. Triggers a data minor bump (same schema)."""
        self.flags_["raw"] = True
        if suffix:
            self.state_["data"]["raw_suffix"] = suffix
        return self

    def snapshot_mixed(self, suffix: str = None) -> "SnapshotConfig":
        """Mark mixed (real + synthetic) dataset for snapshotting. Triggers a data minor bump."""
        self.flags_["mixed"] = True
        if suffix:
            self.state_["data"]["mixed_suffix"] = suffix
        return self

    def snapshot_schema_change(self) -> "SnapshotConfig":
        """
        Mark this data snapshot as a schema change (different columns). Upgrades
        any active data write (raw / mixed) to a data major bump with minor = 0.
        Call alongside .snapshot_raw() and/or .snapshot_mixed().
        """
        self.flags_["data_major"] = True
        return self

    def snapshot_models(self) -> "SnapshotConfig":
        """Mark model artifacts for snapshotting. Triggers a model minor bump (same data)."""
        self.flags_["models"] = True
        return self

    def snapshot_models_new_data(self) -> "SnapshotConfig":
        """Mark model artifacts for snapshotting AND trigger a model major bump
        (retrained on new data). Must be set explicitly — there is no automatic
        upgrade from a data version bump in the same session."""
        self.flags_["models"] = True
        self.flags_["model_major"] = True
        return self

    def snapshot_hyperparams(self) -> "SnapshotConfig":
        """Mark hyperparameter set for snapshotting. Triggers a hyperparams minor bump
        (existing params, new values)."""
        self.flags_["hyperparams"] = True
        return self

    def snapshot_hyperparams_new_grid(self) -> "SnapshotConfig":
        """Mark hyperparameter set for snapshotting AND trigger a hyperparams major bump
        (new parameter added to the grid)."""
        self.flags_["hyperparams"] = True
        self.flags_["hyperparams_major"] = True
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
        """
        self.flags_["tune"] = True
        self.search_strategy = strategy
        self.search_n_iter   = n_iter
        self.search_cv       = cv
        self.search_scoring  = scoring
        self.new_grids       = new_grids or {}
        return self

    def use_data_version(self, version: str) -> "SnapshotConfig":
        """Pin raw/mixed versions to an existing data snapshot. Model and
        hyperparam versions still bump independently based on their own flags.

        Raises if any data-write or schema-change flag is set — pinning means
        reusing an existing snapshot, which those flags would overwrite.
        """
        self.pinned_data_version_ = self.parse_pinned_version_(version)
        return self

    def use_model_version(self, version: str) -> "SnapshotConfig":
        """Pin the model version (symmetric with use_data_version). Rarely needed
        outside of overriding an existing model snapshot intentionally."""
        self.pinned_model_version_ = self.parse_pinned_version_(version)
        return self

    def use_hyperparam_version(self, version: str) -> "SnapshotConfig":
        """Pin the hyperparams version (symmetric with use_data_version)."""
        self.pinned_hyperparam_version_ = self.parse_pinned_version_(version)
        return self

    def preset(self, name: str) -> "SnapshotConfig":
        """
        Apply a named preset. See PRESETS_ at the top of the module.
        Can be combined with explicit flag methods.

        Presets default to minor bumps. Chain .snapshot_schema_change() or
        .snapshot_models_new_data() or .snapshot_hyperparams_new_grid()
        after to upgrade to major bumps where appropriate.
        """
        if name not in PRESETS_:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(PRESETS_.keys())}"
            )
        dispatch = {
            "raw":             lambda: self.flags_.__setitem__("raw", True),
            "mixed":           lambda: self.flags_.__setitem__("mixed", True),
            "models":          lambda: self.flags_.__setitem__("models", True),
            "models_new_data": lambda: (self.flags_.__setitem__("models", True),
                                        self.flags_.__setitem__("model_major", True)),
            "hyperparams":     lambda: self.flags_.__setitem__("hyperparams", True),
            "tune":            lambda: self.flags_.__setitem__("tune", True),
        }
        for flag in PRESETS_[name]:
            dispatch[flag]()
        return self

    # =========================================================================
    # Build — computes version strings, does NOT write to GCS
    # =========================================================================

    def build(self) -> "SnapshotConfig":
        """
        Compute new version strings for each entity based on active flags.
        Three entities bump independently:

          data:        raw/mixed       → minor; schema_change       → major
          model:       models          → minor; models_new_data     → major
          hyperparams: hyperparams     → minor; hyperparams_new_grid → major

        Pinned versions (via use_*_version()) override the bump for that entity.
        """
        self.validate_pin_conflicts_()

        self.new_data_        = self.compute_new_version_(
            current=(self.state_["data"]["major"], self.state_["data"]["minor"]),
            write_flag=self.flags_["raw"] or self.flags_["mixed"] or self.flags_["data_major"],
            major_flag=self.flags_["data_major"],
        )
        self.new_model_       = self.compute_new_version_(
            current=(self.state_["model"]["major"], self.state_["model"]["minor"]),
            write_flag=self.flags_["models"],
            major_flag=self.flags_["model_major"],
        )
        self.new_hyperparams_ = self.compute_new_version_(
            current=(self.state_["hyperparams"]["major"], self.state_["hyperparams"]["minor"]),
            write_flag=self.flags_["hyperparams"],
            major_flag=self.flags_["hyperparams_major"],
        )

        raw_sfx   = self.state_["data"]["raw_suffix"]
        mixed_sfx = self.state_["data"]["mixed_suffix"]

        data_v       = self.pinned_data_version_       or self.new_data_
        model_v      = self.pinned_model_version_      or self.new_model_
        hyperparam_v = self.pinned_hyperparam_version_ or self.new_hyperparams_

        self.raw_version        = f"v{data_v[0]}.{data_v[1]}_{raw_sfx}"
        self.mixed_version      = f"v{data_v[0]}.{data_v[1]}_{mixed_sfx}"
        self.model_version      = f"v{model_v[0]}.{model_v[1]}"
        self.hyperparam_version = f"v{hyperparam_v[0]}.{hyperparam_v[1]}"
        self.built_             = True

        self.print_build_summary_()
        return self

    def validate_pin_conflicts_(self):
        if self.pinned_data_version_ and (
            self.flags_["raw"] or self.flags_["mixed"] or self.flags_["data_major"]
        ):
            raise ValueError(
                "Cannot pin data version with raw/mixed/schema_change flags — "
                "pinning reuses an existing snapshot, which those flags would overwrite."
            )
        if self.pinned_model_version_ and self.flags_["models"]:
            raise ValueError(
                "Cannot pin model version while snapshot_models() is active — "
                "the pinned version would be overwritten."
            )
        if self.pinned_hyperparam_version_ and self.flags_["hyperparams"]:
            raise ValueError(
                "Cannot pin hyperparam version while snapshot_hyperparams() is active — "
                "the pinned version would be overwritten."
            )

    @staticmethod
    def compute_new_version_(current: tuple, write_flag: bool, major_flag: bool) -> tuple:
        """Apply bump rules to one entity's (major, minor) counter."""
        major, minor = current
        if not write_flag:
            return (major, minor)
        if major_flag:
            return (major + 1, 0)
        return (major, minor + 1)

    def print_build_summary_(self):
        active = [k for k, v in self.flags_.items() if v]
        d_cur = (self.state_["data"]["major"], self.state_["data"]["minor"])
        m_cur = (self.state_["model"]["major"], self.state_["model"]["minor"])
        h_cur = (self.state_["hyperparams"]["major"], self.state_["hyperparams"]["minor"])

        def arrow_(cur, new):
            return f"v{cur[0]}.{cur[1]} → v{new[0]}.{new[1]}" if cur != new else f"v{cur[0]}.{cur[1]} (unchanged)"

        print("\nSnapshotConfig ready:")
        print(f"  Active flags      : {active if active else ['none (dry run)']}")
        print(f"  data              : {arrow_(d_cur, self.new_data_)}"
              + (f"  [pinned → v{self.pinned_data_version_[0]}.{self.pinned_data_version_[1]}]"
                 if self.pinned_data_version_ else ""))
        print(f"  model             : {arrow_(m_cur, self.new_model_)}"
              + (f"  [pinned → v{self.pinned_model_version_[0]}.{self.pinned_model_version_[1]}]"
                 if self.pinned_model_version_ else ""))
        print(f"  hyperparams       : {arrow_(h_cur, self.new_hyperparams_)}"
              + (f"  [pinned → v{self.pinned_hyperparam_version_[0]}.{self.pinned_hyperparam_version_[1]}]"
                 if self.pinned_hyperparam_version_ else ""))
        print(f"  raw_version       : {self.raw_version}")
        print(f"  mixed_version     : {self.mixed_version}")
        print(f"  model_version     : {self.model_version}")
        print(f"  hyperparam_version: {self.hyperparam_version}")
        if self.flags_["tune"]:
            print(f"  Tuning            : strategy={self.search_strategy}, "
                  f"n_iter={self.search_n_iter}, cv={self.search_cv}, "
                  f"scoring={self.search_scoring}")
        if any(self.flags_.values()):
            print("\n  Call config.commit() after all snapshots succeed.")

    # =========================================================================
    # Commit — write updated versions.json back to GCS
    # =========================================================================

    def commit(self):
        """
        Persist new version numbers to GCS versions.json. Only entities whose
        flags were active are bumped; others round-trip unchanged.
        """
        if not self.built_:
            raise RuntimeError("Call .build() before .commit().")
        if not any(self.flags_.values()):
            print("No snapshots were active — nothing to commit.")
            return

        new_state = {
            "data": {
                "major":        self.new_data_[0],
                "minor":        self.new_data_[1],
                "raw_suffix":   self.state_["data"]["raw_suffix"],
                "mixed_suffix": self.state_["data"]["mixed_suffix"],
            },
            "model":       {"major": self.new_model_[0],       "minor": self.new_model_[1]},
            "hyperparams": {"major": self.new_hyperparams_[0], "minor": self.new_hyperparams_[1]},
            "last_updated":        datetime.utcnow().isoformat(),
            "last_snapshot_types": [k for k, v in self.flags_.items() if v],
        }

        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)
        bucket.blob(VERSIONS_BLOB_).upload_from_string(
            json.dumps(new_state, indent=2),
            content_type="application/json",
        )
        d = new_state["data"]
        m = new_state["model"]
        h = new_state["hyperparams"]
        print(
            f"Committed versions.json → "
            f"data v{d['major']}.{d['minor']}, "
            f"model v{m['major']}.{m['minor']}, "
            f"hyperparams v{h['major']}.{h['minor']}"
        )

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
        m = re.match(r"^v?(\d+)\.(\d+)$", str(version_str).strip())
        if not m:
            raise ValueError(
                f"Cannot parse version '{version_str}'. Expected format 'X.Y' or 'vX.Y'."
            )
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def parse_pinned_version_(version: str) -> tuple:
        return SnapshotConfig.parse_version_(version)

    def __repr__(self) -> str:
        if self.built_:
            active = [k for k, v in self.flags_.items() if v]
            return (f"SnapshotConfig(data={self.new_data_}, model={self.new_model_}, "
                    f"hyperparams={self.new_hyperparams_}, active={active})")
        return "SnapshotConfig(not built — call .build())"
