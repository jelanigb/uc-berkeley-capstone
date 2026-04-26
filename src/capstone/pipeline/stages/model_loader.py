"""
ModelLoader — load saved model families from GCS.

Auto-discovery
--------------
Models are saved by `utils.snapshot_model.save_model` under
`models/<full_tag>/`, where `full_tag = f"{base_version}_{model_name}"`. A
single bump produces multiple directories like `v3.1_lr_l1`, `v3.1_rf`,
`v3.1_xgb`, `v3.1_ensemble`. Loading "v3.1" therefore means listing
`models/v3.1_*/` and reading each match. The model name is recovered by
stripping the base-version-and-underscore prefix from the full tag.

Keying convention
-----------------
`run.models` is keyed by the model name (e.g. "lr_l1") in the single-base-version
case so downstream Validator code reads naturally. When multiple base versions
are loaded (the `retro_validate` scenario), keys collide across versions, so we
prefix with the base version: "v3.0/lr_l1", "v3.1/lr_l1", etc. This keeps the
single-version case ergonomic without sacrificing the cross-version comparison.

State carried per model
-----------------------
Each entry stores model + scaler + feature_cols + metadata in a dict so the
validator can reproduce the original training-time alignment exactly. This
shape may differ from what `ModelTrainer` eventually populates; the validator
is responsible for unpacking either shape (see Validator design notes).

Read-only
---------
Writing models back to GCS is the job of `ModelSnapshotter` (next pass), gated
by `config.take_snapshot_models`. `ModelLoader` never writes.
"""

from google.cloud import storage

from constants import BUCKET_NAME, PROJECT_ID
from pipeline.pipeline_run import PipelineRun
from pipeline.run_config import VersionConfig
from utils.snapshot_model import load_model


class ModelLoader:
    """Load one or more families of saved models into `run.models`."""

    def __init__(self, config: VersionConfig, versions: list[str] = None):
        self.config = config
        # `validate_current` passes None and gets the current model version;
        # `retro_validate` passes an explicit list of base versions to compare.
        self.versions = list(versions) if versions else [config.model_version]
        if not self.versions or any(not v for v in self.versions):
            raise ValueError(
                "ModelLoader requires at least one non-empty base version. "
                "Did you call config.build() before constructing the loader?"
            )

    def run(self, run: PipelineRun) -> PipelineRun:
        gcs_client = storage.Client(project=PROJECT_ID)
        bucket = gcs_client.bucket(BUCKET_NAME)

        multi_version = len(self.versions) > 1

        for base_version in self.versions:
            full_tags = self._discover_full_tags(bucket, base_version)
            if not full_tags:
                raise FileNotFoundError(
                    f"No model directories matched "
                    f"gs://{BUCKET_NAME}/models/{base_version}_*"
                )

            print(
                f"Loading {len(full_tags)} models for base version "
                f"'{base_version}': {full_tags}"
            )
            for full_tag in full_tags:
                model, scaler, feature_cols, metadata = load_model(full_tag)
                model_name = full_tag[len(base_version) + 1:]
                key = f"{base_version}/{model_name}" if multi_version else model_name
                run.models[key] = {
                    "model": model,
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    "metadata": metadata,
                }

        return run

    @staticmethod
    def _discover_full_tags(bucket, base_version: str) -> list[str]:
        """Return sorted unique `<base>_<suffix>` directory names under models/.

        Listing by `models/<base>_` (note trailing underscore) prevents
        `v3.1` from matching `v3.10_*` etc.
        """
        prefix = f"models/{base_version}_"
        seen: set[str] = set()
        for blob in bucket.list_blobs(prefix=prefix):
            # blob.name format: "models/<full_tag>/<artifact_filename>"
            parts = blob.name.split("/", 2)
            if len(parts) >= 2 and parts[0] == "models":
                full_tag = parts[1]
                if full_tag.startswith(f"{base_version}_"):
                    seen.add(full_tag)
        return sorted(seen)
