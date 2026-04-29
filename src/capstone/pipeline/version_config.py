"""
VersionConfig — builder-pattern orchestrator for a single pipeline run.

Carries four independently-versioned entities — base data, model, baselines 
(which includes medians) and hyperparameters — each with its own major.minor 
counter in GCS at `config/versions.json`. 

Also carries run-wide flags consumed by the stage classes (e.g. 
`use_synthetic`, `tune_models`) and the search config consumed by 
`PipelineFactory` when wiring the trainer.

Bump rules (summary):
    Base Data
      .snapshot_raw() / .snapshot_final()         -> data minor bump (new pull, same schema)
                                                     snapshot_raw() also triggers a baselines major bump
      .snapshot_schema_change()                   -> data major bump (columns changed)
    Model
      .snapshot_models()                          -> model minor bump (props / hyperparams changed)
      .snapshot_models_new_data()                 -> model major bump (retrained on new data)
    Baselines
      auto-bumped (major only) by snapshot_raw() and snapshot_models_new_data()
      baselines never do a minor bump — every change is a fresh pull
    Hyperparameters
      .snapshot_hyperparams()                     -> hyperparams minor bump (values tweaked)
      .snapshot_hyperparams_new_grid()            -> hyperparams major bump (new param added to grid)

Typical usage (notebook config cell):

    from pipeline.version_config import VersionConfig

    config = (
        VersionConfig.load(use_synthetic=True)
        .snapshot_models_new_data()
        .snapshot_hyperparams()
        .build()
    )

After all snapshots succeed, call `config.commit()` to persist the new
version numbers back to GCS so the next session picks them up automatically.
"""

import copy
import json
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from google.cloud import storage

from constants import PROJECT_ID, BUCKET_NAME

VERSIONS_BLOB_ = "config/versions.json"

DEFAULT_SYNTHETIC_REAL_PCT_ = 0.9


class Flag(Enum):
    """Internal flag identifiers for `VersionConfig.flags_`.

    Local to this module — external callers interact via the
    `take_snapshot_*` / `tune_models` properties, not the flag dict.
    """
    RAW = "raw"
    FINAL = "final"
    DATA_MAJOR = "data_major"
    MODELS = "models"
    MODEL_MAJOR = "model_major"
    BASELINES_MAJOR = "baselines"
    HYPERPARAMS = "hyperparams"
    HYPERPARAMS_MAJOR = "hyperparams_major"
    TUNE = "tune"

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
        "major": 3,
        "minor": 3,
        "raw_suffix": "real",
    },
    "baselines": {"major": 3, "minor": 3},
    "model": {"major": 5, "minor": 0},
    "hyperparams": {"major": 1, "minor": 0},
    "last_updated": None,
    "last_snapshot_types": [],
}


class VersionConfig:

    def __init__(self, state: dict, use_synthetic: bool = False, target_real_pct: float = None):
        self.state_ = state
        self.flags_ = {f: False for f in Flag}
        self.built_ = False
        self.search_strategy = "random"  # default; override with .tune(strategy=...)
        self.search_n_iter = 50
        self.search_cv = 5
        self.search_scoring = "roc_auc"
        self.new_grids = {}  # model class name -> param grid override

        # Run-wide flags (non-snapshot)
        self.use_synthetic_ = use_synthetic
        if use_synthetic and target_real_pct is None:
            self.target_real_pct_ = DEFAULT_SYNTHETIC_REAL_PCT_
        elif use_synthetic:
            if not (0 < target_real_pct < 1):
                raise ValueError(
                    f"target_real_pct must be in (0, 1), got {target_real_pct!r}."
                )
            self.target_real_pct_ = target_real_pct
        else:
            self.target_real_pct_ = None

        # Pinning — set by use_*_version methods
        self.pinned_data_version_ = None
        self.pinned_baselines_version_ = None
        self.pinned_raw_suffix_ = None
        self.pinned_model_version_ = None
        self.pinned_hyperparam_version_ = None

        # Computed new versions (major, minor) per entity — set by build()
        self.new_data_ = None
        self.new_baselines_ = None
        self.new_model_ = None
        self.new_hyperparams_ = None

        # Public version strings — set by build().
        # Unprefixed = current (pre-bump): what loaders read from GCS.
        # next_*     = target  (post-bump): what snapshotters write to GCS.
        self.raw_version = None
        self.baselines_version = None
        self.model_version = None
        self.hyperparam_version = None
        self.next_raw_version = None
        self.next_final_version = None       # no pre-bump final; nothing loads it
        self.next_model_version = None
        self.next_baselines_version = None
        self.next_hyperparam_version = None

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def load(cls, use_synthetic: bool = False, target_real_pct: float = None) -> "VersionConfig":
        """
        Load current version state from GCS.

        Parameters
        ----------
        use_synthetic : bool
            Whether SyntheticAugmenter should append synthetic rows to the
            train split. Defaults to False; pass True to enable.

        Handles one-time migration from the legacy flat-schema versions.json
        (single `"version"` key) to the new nested schema with independent
        data/model/hyperparams counters. Also renames the legacy
        `mixed_suffix` key to `final_suffix` on nested schemas written by
        older versions of this module. Migrations run in-memory only; GCS
        is rewritten on the next commit().
        """
        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)
        blob = bucket.blob(VERSIONS_BLOB_)

        if blob.exists():
            state = json.loads(blob.download_as_text())
            if "version" in state and "data" not in state:
                state = cls.migrate_flat_to_nested_(state)
                print(
                    "Migrated legacy versions.json -> nested schema "
                    "(data/model/hyperparams). Next commit() will persist the new shape."
                )
            elif "data" in state and "mixed_suffix" in state["data"] and "final_suffix" not in state["data"]:
                state["data"]["final_suffix"] = state["data"].pop("mixed_suffix")
                print(
                    "Renamed data.mixed_suffix -> data.final_suffix. "
                    "Next commit() will persist the new key name."
                )
        else:
            state = copy.deepcopy(DEFAULT_STATE_)
            print(
                "No versions.json found in GCS — using defaults. "
                "Call config.commit() after your first snapshot to persist versions."
            )

        d = state["data"]
        b = state["baselines"]
        m = state["model"]
        h = state["hyperparams"]
        _synth_pct = round((target_real_pct or DEFAULT_SYNTHETIC_REAL_PCT_) * 100) if use_synthetic else None
        print(
            f"VersionConfig loaded:\n"
            f"  data:          v{d['major']}.{d['minor']} (raw_suffix='{d['raw_suffix']}')\n"
            f"  baselines:     v{b['major']}.{b['minor']}\n"
            f"  model:         v{m['major']}.{m['minor']}\n"
            f"  hyperparams:   v{h['major']}.{h['minor']}\n"
            f"  use_synthetic: {use_synthetic}"
            + (f" (target_real_pct={_synth_pct}%)" if use_synthetic else "")
        )
        return cls(state, use_synthetic=use_synthetic, target_real_pct=target_real_pct)

    @staticmethod
    def migrate_flat_to_nested_(flat: dict) -> dict:
        """
        Convert legacy {"version": "3.1", "raw_suffix": ..., "mixed_suffix": ...}
        to the new nested schema. Data and model inherit the flat version;
        hyperparams bootstraps to v1.0 (starting defaults under the new scheme).
        The legacy `mixed_suffix` key is renamed to `final_suffix`.
        """
        major, minor = VersionConfig.parse_version_(flat["version"])
        return {
            "data": {
                "major": major,
                "minor": minor,
                "raw_suffix": flat.get("raw_suffix", "real"),
            },
            "model": {"major": major, "minor": minor},
            "baselines": {"major": major, "minor": 0},
            "hyperparams": {"major": 1, "minor": 0},
            "last_updated": flat.get("last_updated"),
            "last_snapshot_types": flat.get("last_snapshot_types", []),
        }

    # =========================================================================
    # Builder methods — each returns self for chaining
    # =========================================================================

    def snapshot_raw(self, suffix: str = None) -> "VersionConfig":
        """Mark raw BQ pull for snapshotting. Triggers a data minor bump (same schema)
        and also a major bump for baselines."""
        self.flags_[Flag.RAW] = True
        self.flags_[Flag.BASELINES_MAJOR] = True
        if suffix:
            self.state_["data"]["raw_suffix"] = suffix
        return self

    def snapshot_final(self, suffix: str = None) -> "VersionConfig":
        """Mark the final training dataset (feature-engineered, optionally with
        synthetic rows) for snapshotting. Triggers a data minor bump."""
        self.flags_[Flag.FINAL] = True
        if suffix:
            self.state_["data"]["final_suffix"] = suffix
        return self

    def snapshot_schema_change(self) -> "VersionConfig":
        """
        Mark this data snapshot as a schema change (different columns). Upgrades
        any active data write (raw / final) to a data major bump with minor = 0.
        Also bumps the baselines version the same way.
        Call alongside .snapshot_raw() and/or .snapshot_final().
        """
        self.flags_[Flag.DATA_MAJOR] = True
        self.flags_[Flag.BASELINES_MAJOR] = True
        return self

    def snapshot_models(self) -> "VersionConfig":
        """Mark model artifacts for snapshotting. Triggers a model minor bump (same data)."""
        self.flags_[Flag.MODELS] = True
        return self

    def snapshot_models_new_data(self) -> "VersionConfig":
        """Mark model artifacts for snapshotting AND trigger a model major bump
        (retrained on new data). Must be set explicitly — there is no automatic
        upgrade from a data version bump in the same session."""
        self.flags_[Flag.MODELS] = True
        self.flags_[Flag.MODEL_MAJOR] = True
        return self

    def snapshot_hyperparams(self) -> "VersionConfig":
        """Mark hyperparameter set for snapshotting. Triggers a hyperparams minor bump
        (existing params, new values)."""
        self.flags_[Flag.HYPERPARAMS] = True
        return self

    def snapshot_hyperparams_new_grid(self) -> "VersionConfig":
        """Mark hyperparameter set for snapshotting AND trigger a hyperparams major bump
        (new parameter added to the grid)."""
        self.flags_[Flag.HYPERPARAMS] = True
        self.flags_[Flag.HYPERPARAMS_MAJOR] = True
        return self

    def tune(
        self,
        strategy: str = "random",
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = "roc_auc",
        new_grids: dict = None,
    ) -> "VersionConfig":
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
        self.flags_[Flag.TUNE] = True
        self.search_strategy = strategy
        self.search_n_iter = n_iter
        self.search_cv = cv
        self.search_scoring = scoring
        self.new_grids = new_grids or {}
        return self

    def use_data_version(self, version: str, suffix: str = None) -> "VersionConfig":
        """Pin raw/final versions to an existing data snapshot. Model and
        hyperparam versions still bump independently based on their own flags.

        Parameters
        ----------
        suffix : str, optional
            Override the raw-version suffix (default: state raw_suffix, usually
            "real"). Pass the full suffix as it appears in the GCS path, e.g.
            "mixed_80real".

        Raises if any data-write or schema-change flag is set — pinning means
        reusing an existing snapshot, which those flags would overwrite.
        """
        self.pinned_data_version_ = self.parse_pinned_version_(version)
        if suffix is not None:
            self.pinned_raw_suffix_ = suffix
        return self

    def use_baselines_version(self, version:str) -> "VersionConfig":
        """
        Override method if for any reason baselines and medians versions are missing
        for a given data version.
        """
        self.pinned_baselines_version_ = self.parse_pinned_version_(version)
        return self

    def use_model_version(self, version: str) -> "VersionConfig":
        """Pin the model version (symmetric with use_data_version). Rarely needed
        outside of overriding an existing model snapshot intentionally."""
        self.pinned_model_version_ = self.parse_pinned_version_(version)
        return self

    def use_hyperparam_version(self, version: str) -> "VersionConfig":
        """Pin the hyperparams version (symmetric with use_data_version)."""
        self.pinned_hyperparam_version_ = self.parse_pinned_version_(version)
        return self

    # =========================================================================
    # Build — computes version strings, does NOT write to GCS
    # =========================================================================

    def build(self) -> "VersionConfig":
        """
        Compute new version strings for each entity based on active flags.
        Three entities bump independently:

          data:        raw/final       -> minor; schema_change        -> major
          baselines:   baselines and medians -> should mirror data major
          model:       models          -> minor; models_new_data      -> major
          hyperparams: hyperparams     -> minor; hyperparams_new_grid -> major

        Pinned versions (via use_*_version()) override the bump for that entity.
        """
        self.validate_pin_conflicts_()

        self.new_data_ = self.compute_new_version_(
            current=(self.state_["data"]["major"], self.state_["data"]["minor"]),
            write_flag=self.flags_[Flag.RAW] or self.flags_[Flag.FINAL] or self.flags_[Flag.DATA_MAJOR],
            major_flag=self.flags_[Flag.DATA_MAJOR],
        )
        self.new_model_ = self.compute_new_version_(
            current=(self.state_["model"]["major"], self.state_["model"]["minor"]),
            write_flag=self.flags_[Flag.MODELS],
            major_flag=self.flags_[Flag.MODEL_MAJOR],
        )
        # Baselines only ever do major bumps — minor is not meaningful.
        # Triggered by snapshot_raw() (new data pull) and snapshot_models_new_data().
        self.new_baselines_ = self.compute_new_version_(
            current=(self.state_["baselines"]["major"], self.state_["baselines"]["minor"]),
            write_flag=self.flags_[Flag.BASELINES_MAJOR],
            major_flag=True,
        )
        self.new_hyperparams_ = self.compute_new_version_(
            current=(self.state_["hyperparams"]["major"], self.state_["hyperparams"]["minor"]),
            write_flag=self.flags_[Flag.HYPERPARAMS],
            major_flag=self.flags_[Flag.HYPERPARAMS_MAJOR],
        )

        raw_sfx = self.pinned_raw_suffix_ or self.state_["data"]["raw_suffix"]
        if self.use_synthetic_:
            real_pct = round(self.target_real_pct_ * 100)
            final_sfx = f"mixed_{real_pct}real"
        else:
            final_sfx = "100real"

        # Post-bump (target) — what snapshotters write to.
        # Pinning overrides the bump; if pinned, next_* == current (no write will happen).
        data_v = self.pinned_data_version_ or self.new_data_
        model_v = self.pinned_model_version_ or self.new_model_
        baselines_v = self.pinned_baselines_version_ or self.new_baselines_
        hyperparam_v = self.pinned_hyperparam_version_ or self.new_hyperparams_

        self.next_raw_version = f"v{data_v[0]}.{data_v[1]}_{raw_sfx}"
        self.next_final_version = f"v{data_v[0]}.{data_v[1]}_{final_sfx}"
        self.next_model_version = f"v{model_v[0]}.{model_v[1]}"
        self.next_baselines_version = f"v{baselines_v[0]}.{baselines_v[1]}"
        self.next_hyperparam_version = f"v{hyperparam_v[0]}.{hyperparam_v[1]}"

        # Pre-bump (current) — what loaders read from GCS.
        # Uses state_ directly so it always reflects what actually exists.
        # Pinning overrides: a pinned version is the intended read target.
        d = self.state_["data"]
        m = self.state_["model"]
        b = self.state_["baselines"]
        h = self.state_["hyperparams"]
        cur_data = self.pinned_data_version_ or (d["major"], d["minor"])
        cur_model = self.pinned_model_version_ or (m["major"], m["minor"])
        cur_baselines = self.pinned_baselines_version_ or (b["major"], b["minor"])
        cur_hyperparams = self.pinned_hyperparam_version_ or (h["major"], h["minor"])

        self.raw_version = f"v{cur_data[0]}.{cur_data[1]}_{raw_sfx}"
        self.model_version = f"v{cur_model[0]}.{cur_model[1]}"
        self.baselines_version = f"v{cur_baselines[0]}.{cur_baselines[1]}"
        self.hyperparam_version = f"v{cur_hyperparams[0]}.{cur_hyperparams[1]}"

        self.built_ = True

        self.print_build_summary_()
        return self

    def validate_pin_conflicts_(self):
        if self.pinned_data_version_ and (
            self.flags_[Flag.RAW] or self.flags_[Flag.FINAL] or self.flags_[Flag.DATA_MAJOR]
        ):
            raise ValueError(
                "Cannot pin data version with raw/final/schema_change flags — "
                "pinning reuses an existing snapshot, which those flags would overwrite."
            )
        if self.pinned_model_version_ and self.flags_[Flag.MODELS]:
            raise ValueError(
                "Cannot pin model version while snapshot_models() is active — "
                "the pinned version would be overwritten."
            )
        if self.pinned_baselines_version_ and self.flags_[Flag.BASELINES_MAJOR]:
            raise ValueError(
                "Cannot pin baselines version while a baselines bump is active "
                "(triggered by snapshot_raw() or snapshot_models_new_data()) — "
                "the pinned version would be overwritten."
            )
        if self.pinned_hyperparam_version_ and self.flags_[Flag.HYPERPARAMS]:
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
        active = [k.value for k, v in self.flags_.items() if v]
        d_cur = (self.state_["data"]["major"], self.state_["data"]["minor"])
        b_cur = (self.state_["baselines"]["major"], self.state_["baselines"]["minor"])
        m_cur = (self.state_["model"]["major"], self.state_["model"]["minor"])
        h_cur = (self.state_["hyperparams"]["major"], self.state_["hyperparams"]["minor"])

        def arrow_(cur, new):
            return f"v{cur[0]}.{cur[1]} -> v{new[0]}.{new[1]}" if cur != new else f"v{cur[0]}.{cur[1]} (unchanged)"

        print("\nVersionConfig ready:")
        print(f"  Active flags      : {active if active else ['none (dry run)']}")
        print(f"  data              : {arrow_(d_cur, self.new_data_)}"
              + (f"  [pinned -> v{self.pinned_data_version_[0]}.{self.pinned_data_version_[1]}]"
                 if self.pinned_data_version_ else ""))
        print(f"  baselines         : {arrow_(b_cur, self.new_baselines_)}"
              + (f"  [pinned -> v{self.pinned_baselines_version_[0]}.{self.pinned_baselines_version_[1]}]"
                 if self.pinned_baselines_version_ else ""))
        print(f"  model             : {arrow_(m_cur, self.new_model_)}"
              + (f"  [pinned -> v{self.pinned_model_version_[0]}.{self.pinned_model_version_[1]}]"
                 if self.pinned_model_version_ else ""))
        print(f"  hyperparams       : {arrow_(h_cur, self.new_hyperparams_)}"
              + (f"  [pinned -> v{self.pinned_hyperparam_version_[0]}.{self.pinned_hyperparam_version_[1]}]"
                 if self.pinned_hyperparam_version_ else ""))
        def ver_arrow_(current, nxt):
            if current == nxt:
                return current
            return f"{current}  ->  {nxt}"

        sfx_note = f"  [suffix overridden -> '{self.pinned_raw_suffix_}']" if self.pinned_raw_suffix_ else ""
        print(f"  raw_version       : {ver_arrow_(self.raw_version, self.next_raw_version)}{sfx_note}")
        print(f"  next_final_version: {self.next_final_version}")
        print(f"  model_version     : {ver_arrow_(self.model_version, self.next_model_version)}")
        print(f"  baselines_version : {ver_arrow_(self.baselines_version, self.next_baselines_version)}")
        print(f"  hyperparam_version: {ver_arrow_(self.hyperparam_version, self.next_hyperparam_version)}")
        print(f"  use_synthetic     : {self.use_synthetic_}")
        if self.flags_[Flag.TUNE]:
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
                "major": self.new_data_[0],
                "minor": self.new_data_[1],
                "raw_suffix": self.state_["data"]["raw_suffix"],
            },
            "model": {"major": self.new_model_[0], "minor": self.new_model_[1]},
            "baselines": {"major": self.new_baselines_[0], "minor": self.new_baselines_[1]},
            "hyperparams": {"major": self.new_hyperparams_[0], "minor": self.new_hyperparams_[1]},
            "last_updated": datetime.utcnow().isoformat(),
            "last_snapshot_types": [k.value for k, v in self.flags_.items() if v],
        }

        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)
        bucket.blob(VERSIONS_BLOB_).upload_from_string(
            json.dumps(new_state, indent=2),
            content_type="application/json",
        )
        d = new_state["data"]
        m = new_state["model"]
        b = new_state["baselines"]
        h = new_state["hyperparams"]
        print(
            f"Committed versions.json -> "
            f"data v{d['major']}.{d['minor']}, "
            f"model v{m['major']}.{m['minor']}, "
            f"baselines v{b['major']}.{b['minor']}, "
            f"hyperparams v{h['major']}.{h['minor']}"
        )

    # =========================================================================
    # Flag properties
    # =========================================================================

    @property
    def take_snapshot_raw(self) -> bool:
        return self.flags_[Flag.RAW]

    @property
    def take_snapshot_final(self) -> bool:
        return self.flags_[Flag.FINAL]

    @property
    def take_snapshot_models(self) -> bool:
        return self.flags_[Flag.MODELS]
    
    @property
    def take_snapshot_baselines(self) -> bool:
        return self.flags_[Flag.BASELINES_MAJOR]

    @property
    def take_snapshot_hyperparams(self) -> bool:
        return self.flags_[Flag.HYPERPARAMS]

    @property
    def tune_models(self) -> bool:
        return self.flags_[Flag.TUNE]

    @property
    def use_synthetic(self) -> bool:
        return self.use_synthetic_

    @property
    def target_real_pct(self) -> float:
        """Fraction of real rows in the final train mix (e.g. 0.8 = 80% real).
        None when use_synthetic=False.
        """
        return self.target_real_pct_

    @property
    def search_config(self) -> dict:
        """Search config dict; read by PipelineFactory when wiring ModelTrainer."""
        return {
            "strategy": self.search_strategy,
            "n_iter": self.search_n_iter,
            "cv": self.search_cv,
            "scoring": self.search_scoring,
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
        return VersionConfig.parse_version_(version)

    def __repr__(self) -> str:
        if self.built_:
            active = [k.value for k, v in self.flags_.items() if v]
            return (f"VersionConfig(data={self.new_data_}, model={self.new_model_}, "
                    f"baselines={self.new_baselines_}, "
                    f"hyperparams={self.new_hyperparams_}, "
                    f"use_synthetic={self.use_synthetic_}, active={active})")
        return "VersionConfig(not built — call .build())"
