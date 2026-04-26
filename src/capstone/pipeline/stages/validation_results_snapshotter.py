"""
ValidationResultsSnapshotter — persist run.results to GCS after validation.

Appends one JSON line to models/{model_version}/validation_results.jsonl
so every validation run for a given model version is preserved. The file
is append-only; use load_validation_results() to reconstruct history.
"""

from pipeline.pipeline_run import PipelineRun
from pipeline.version_config import VersionConfig
from utils.snapshot_model import save_validation_results


class ValidationResultsSnapshotter:
    """Stage — append run.results to a JSONL sidecar adjacent to model artifacts."""

    def __init__(self, config: VersionConfig):
        self.config = config

    def run(self, run: PipelineRun) -> PipelineRun:
        if not run.results:
            raise RuntimeError(
                "ValidationResultsSnapshotter: run.results is empty. "
                "Did Validator.run() complete?"
            )
        save_validation_results(run.results, self.config)
        return run
