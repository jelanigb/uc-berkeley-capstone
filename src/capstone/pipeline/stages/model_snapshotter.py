"""
ModelSnapshotter — persist fitted models from run.models to GCS.

Write-side counterpart to ModelLoader. Gated by config.take_snapshot_models;
a no-op when that flag is False. Calls utils.snapshot_model.save_model once
per entry in run.models, building the full GCS version tag as
f"{config.model_version}_{model_name}" (e.g. "v4.0_lr_l1").

Expected shape of each run.models entry (populated by ModelTrainer):

    run.models[model_name] = {
        "model":         fitted sklearn-compatible model,
        "scaler":        StandardScaler fitted on X_train,
        "feature_cols":  list[str] of feature column names,
        "training_data": utils.snapshot_model.TrainingData instance,
        "result":        utils.snapshot_model.ModelResult instance,
        "model_config":  utils.snapshot_model.ModelConfig instance,
    }

Note: "model_config" is the ModelConfig dataclass from utils.snapshot_model,
not VersionConfig. The key name avoids ambiguity.

Never modifies PipelineRun state — reads only.
"""

from pipeline.pipeline_run import PipelineRun
from pipeline.version_config import VersionConfig
from utils.snapshot_model import save_model


class ModelSnapshotter:
    """Persist each fitted model in run.models to GCS under model_version_*."""

    def __init__(self, config: VersionConfig):
        self.config = config

    def run(self, run: PipelineRun) -> PipelineRun:
        if not self.config.take_snapshot_models:
            print("[ModelSnapshotter] take_snapshot_models=False — skipping.")
            return run

        if not run.models:
            raise RuntimeError(
                "ModelSnapshotter: run.models is empty. "
                "Did ModelTrainer.run() complete? Cannot snapshot with no models."
            )

        for model_name, entry in run.models.items():
            version_tag = f"{self.config.model_version}_{model_name}"
            save_model(
                model=entry["model"],
                scaler=entry["scaler"],
                feature_cols=entry["feature_cols"],
                version_tag=version_tag,
                data_snapshot_tag=self.config.final_version,
                training_data=entry["training_data"],
                result=entry["result"],
                config=entry["model_config"],
            )

        print(
            f"[ModelSnapshotter] Saved {len(run.models)} model(s) "
            f"under '{self.config.model_version}_*'."
        )
        return run
