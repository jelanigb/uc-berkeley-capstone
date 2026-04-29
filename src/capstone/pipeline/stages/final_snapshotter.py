"""
FinalSnapshotter — persist the six per-split modeling artifacts to GCS.

Sits after SyntheticAugmenter in the stage sequence, so X_train / y_train
include synthetic rows when config.use_synthetic is True. The snapshot
therefore captures the exact training mixture that ModelTrainer will see.

Gated by config.take_snapshot_final; a no-op when that flag is False.
Delegates to utils.snapshot_data.save_splits_snapshot, which writes each
split as a separate parquet file plus a metadata sidecar.

Never modifies PipelineRun state — reads only.
"""

from pipeline.pipeline_run import PipelineRun
from pipeline.version_config import VersionConfig
from utils.snapshot_data import save_splits_snapshot


class FinalSnapshotter:
    """Persist per-split X/y arrays from a completed FeatureEngineer run to GCS."""

    def __init__(self, config: VersionConfig):
        self.config = config

    def run(self, run: PipelineRun) -> PipelineRun:
        if not self.config.take_snapshot_final:
            print("[FinalSnapshotter] take_snapshot_final=False — skipping.")
            return run

        required = ["X_train", "y_train", "X_test", "y_test", "X_val", "y_val"]
        missing = [f for f in required if getattr(run, f) is None]
        if missing:
            raise RuntimeError(
                f"FinalSnapshotter: run fields {missing} are None. "
                "Did FeatureEngineer.run() (and SyntheticAugmenter.run()) complete?"
            )

        version_tag = self.config.next_final_version
        save_splits_snapshot(
            run.X_train, run.y_train,
            run.X_test, run.y_test,
            run.X_val, run.y_val,
            version_tag=version_tag,
        )
        print(f"[FinalSnapshotter] Splits snapshot '{version_tag}' saved to GCS.")
        return run
