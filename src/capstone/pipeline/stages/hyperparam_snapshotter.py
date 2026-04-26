"""
HyperparamSnapshotter — persist hyperparameters from run.models to GCS.

Gated by config.take_snapshot_hyperparams; a no-op when that flag is False.
Builds a params dict by reading model_config.model_type and
model_config.hyperparameters from each run.models entry, then calls
utils.snapshot_hyperparameters.save_hyperparams.

When config.tune_models is True the recorded search_config reflects the
search strategy and settings; otherwise params are recorded as a
pre-tuning baseline (search_config=None in save_hyperparams).

Never modifies PipelineRun state — reads only.
"""

from pipeline.pipeline_run import PipelineRun
from pipeline.version_config import VersionConfig
from utils.snapshot_hyperparameters import save_hyperparams


class HyperparamSnapshotter:
    """Persist hyperparameters for each model in run.models to GCS."""

    def __init__(self, config: VersionConfig):
        self.config = config

    def run(self, run: PipelineRun) -> PipelineRun:
        if not self.config.take_snapshot_hyperparams:
            print("[HyperparamSnapshotter] take_snapshot_hyperparams=False — skipping.")
            return run

        if not run.models:
            raise RuntimeError(
                "HyperparamSnapshotter: run.models is empty. "
                "Did ModelTrainer.run() complete? Cannot snapshot with no models."
            )

        params = {
            entry["model_config"].model_type: entry["model_config"].hyperparameters
            for entry in run.models.values()
        }
        search_config = self.config.search_config if self.config.tune_models else None

        save_hyperparams(
            params=params,
            version_tag=self.config.hyperparam_version,
            search_config=search_config,
        )
        print(
            f"[HyperparamSnapshotter] Saved hyperparams '{self.config.hyperparam_version}' "
            f"for {list(params.keys())}."
        )
        return run
