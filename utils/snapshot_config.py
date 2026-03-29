"""
SnapshotConfig — builder-pattern orchestrator for all GCS snapshot operations.

Manages snapshot flags and version numbers in one place. Reads/writes a
`config/versions.json` file in GCS as the single source of truth for the
current version state.

Usage (notebook config cell):

    from utils.snapshot_config import SnapshotConfig

    # Use a preset:
    config = SnapshotConfig.load().preset("feature_engineering").build()

    # Or chain methods explicitly:
    config = SnapshotConfig.load().snapshot_mixed().snapshot_models().snapshot_hyperparams().build()

    # Dry run (no snapshots, just loads current version strings for reading):
    config = SnapshotConfig.load().build()

Available presets:
    "dry_run"            — no snapshots; loads current versions for reading
    "model_tuning"       — models + hyperparams          → minor version bump
    "feature_engineering"— mixed + models + hyperparams  → minor version bump
    "new_raw_data"       — all four                      → major version bump

After all snapshots succeed, call config.commit() to persist the new version
number back to GCS so the next session picks it up automatically.
"""

import json
import re
from datetime import datetime

from google.cloud import storage

PROJECT_ID     = "maduros-dolce"
BUCKET_NAME    = "maduros-dolce-capstone-data"
_VERSIONS_BLOB = "config/versions.json"

_PRESETS = {
    "dry_run":             [],
    "model_tuning":        ["models", "hyperparams"],
    "feature_engineering": ["mixed", "models", "hyperparams"],
    "new_raw_data":        ["raw", "mixed", "models", "hyperparams"],
}

# Default state used when no versions.json exists in GCS yet
_DEFAULT_STATE = {
    "version":      "3.1",
    "raw_suffix":   "real",
    "mixed_suffix": "mixed_80real",
    "last_updated": None,
    "last_snapshot_types": [],
}


class SnapshotConfig:

    def __init__(self, state: dict):
        self._state   = state
        self._flags   = {k: False for k in ["raw", "mixed", "models", "hyperparams"]}
        self._built   = False
        self._new_version = None

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
        blob = bucket.blob(_VERSIONS_BLOB)

        if blob.exists():
            state = json.loads(blob.download_as_text())
        else:
            state = _DEFAULT_STATE.copy()
            print(
                "No versions.json found in GCS — using defaults "
                f"(v{_DEFAULT_STATE['version']}). "
                "Call config.commit() after your first snapshot to persist versions."
            )

        print(
            f"SnapshotConfig loaded: v{state['version']} "
            f"(raw_suffix='{state['raw_suffix']}', "
            f"mixed_suffix='{state['mixed_suffix']}')"
        )
        return cls(state)

    # =========================================================================
    # Builder methods — each returns self for chaining
    # =========================================================================

    def snapshot_raw(self, suffix: str = None) -> "SnapshotConfig":
        """Mark raw BQ pull for snapshotting. Triggers a major version bump."""
        self._flags["raw"] = True
        if suffix:
            self._state["raw_suffix"] = suffix
        return self

    def snapshot_mixed(self, suffix: str = None) -> "SnapshotConfig":
        """Mark mixed (real + synthetic) dataset for snapshotting."""
        self._flags["mixed"] = True
        if suffix:
            self._state["mixed_suffix"] = suffix
        return self

    def snapshot_models(self) -> "SnapshotConfig":
        """Mark all three model snapshots."""
        self._flags["models"] = True
        return self

    def snapshot_hyperparams(self) -> "SnapshotConfig":
        """Mark hyperparameter set for snapshotting."""
        self._flags["hyperparams"] = True
        return self

    def preset(self, name: str) -> "SnapshotConfig":
        """
        Apply a named preset. See module docstring for available presets.
        Can be combined with explicit flag methods.
        """
        if name not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. "
                f"Available: {list(_PRESETS.keys())}"
            )
        for flag in _PRESETS[name]:
            self._flags[flag] = True
        return self

    # =========================================================================
    # Build — computes version strings, does NOT write to GCS
    # =========================================================================

    def build(self) -> "SnapshotConfig":
        """
        Compute new version strings based on active flags.

        Bumping rules:
          - Raw snapshot active  → major bump (X.Y → X+1.0); signals new dataset
          - Any other snapshot   → minor bump (X.Y → X.Y+1)
          - No snapshots (dry)   → version unchanged; useful for read-only runs
        """
        current = self._state["version"]
        major, minor = self._parse_version(current)
        any_active = any(self._flags.values())

        if self._flags["raw"]:
            new_version = f"{major + 1}.0"
        elif any_active:
            new_version = f"{major}.{minor + 1}"
        else:
            new_version = current   # dry run

        raw_sfx   = self._state["raw_suffix"]
        mixed_sfx = self._state["mixed_suffix"]

        self.raw_version        = f"v{new_version}_{raw_sfx}"
        self.mixed_version      = f"v{new_version}_{mixed_sfx}"
        self.model_version      = f"v{new_version}"
        self.hyperparam_version = f"v{new_version}"
        self._new_version       = new_version
        self._built             = True

        active = [k for k, v in self._flags.items() if v]
        print("\nSnapshotConfig ready:")
        print(f"  Active snapshots : {active if active else ['none (dry run)']}")
        print(f"  Version          : v{current} → v{new_version}")
        print(f"  raw_version      : {self.raw_version}")
        print(f"  mixed_version    : {self.mixed_version}")
        print(f"  model_version    : {self.model_version}")
        print(f"  hyperparam_version: {self.hyperparam_version}")
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
        if not self._built:
            raise RuntimeError("Call .build() before .commit().")
        if not any(self._flags.values()):
            print("No snapshots were active — nothing to commit.")
            return

        new_state = {
            "version":             self._new_version,
            "raw_suffix":          self._state["raw_suffix"],
            "mixed_suffix":        self._state["mixed_suffix"],
            "last_updated":        datetime.utcnow().isoformat(),
            "last_snapshot_types": [k for k, v in self._flags.items() if v],
        }

        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)
        bucket.blob(_VERSIONS_BLOB).upload_from_string(
            json.dumps(new_state, indent=2),
            content_type="application/json",
        )
        print(f"Committed versions.json → v{self._new_version}")

    # =========================================================================
    # Flag properties
    # =========================================================================

    @property
    def take_snapshot_raw(self) -> bool:
        return self._flags["raw"]

    @property
    def take_snapshot_mixed(self) -> bool:
        return self._flags["mixed"]

    @property
    def take_snapshot_models(self) -> bool:
        return self._flags["models"]

    @property
    def take_snapshot_hyperparams(self) -> bool:
        return self._flags["hyperparams"]

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _parse_version(version_str: str) -> tuple:
        m = re.match(r"^(\d+)\.(\d+)$", str(version_str).strip())
        if not m:
            raise ValueError(
                f"Cannot parse version '{version_str}'. Expected format 'X.Y' (e.g. '3.1')."
            )
        return int(m.group(1)), int(m.group(2))

    def __repr__(self) -> str:
        if self._built:
            active = [k for k, v in self._flags.items() if v]
            return f"SnapshotConfig(v{self._new_version}, active={active})"
        return "SnapshotConfig(not built — call .build())"
