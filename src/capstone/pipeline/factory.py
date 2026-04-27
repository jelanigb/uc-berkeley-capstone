"""
PipelineFactory — wiring and scenario assembly.

Single place where stage classes (and their inner Logic collaborators) are
instantiated and packaged into a `PipelineStages` bundle for the notebook
to consume. Each `@staticmethod` on `PipelineFactory` corresponds to one
named scenario in the design doc; switching scenarios in the notebook is
a one-line change to the factory call.

PipelineStages
--------------
A plain class, not a dataclass. Stages absent from a given scenario are
never set as instance attributes; Python's `__getattr__` then fires on
access and raises a descriptive error that names both the stage and the
factory method that built the bundle. This avoids the two failure modes a
dataclass-with-None-defaults design would have:

  - Lookup of a `None` field succeeds, so `__getattr__` would never run.
  - Overriding `__getattribute__` instead carries a real recursion risk
    (every attribute access inside the override re-enters the override).

The `scenario` attribute is set by each factory method and quoted back in
the absent-stage error message.

Stage construction
------------------
Stage classes pull everything they need from `VersionConfig` at construction
time, so each factory method is a flat list of constructor calls. A few
stages also take per-run arguments passed positionally / keyword as
appropriate.

Stage order (all scenarios)
---------------------------
loader → preprocessor → engineer → splitter → scaler
  → [augmenter → trainer]         (training scenarios only)
  → [model_loader]                (optional in training; required in validation)
  → validator → validation_results_snapshotter
"""

from pipeline.version_config import VersionConfig
from pipeline.stages.data_loader import DataLoader
from pipeline.stages.data_preprocessor import DataPreprocessor
from pipeline.stages.data_splitter import DataSplitter
from pipeline.stages.feature_engineer import FeatureEngineer, FeatureEngineerLogic
from pipeline.stages.scaler import Scaler
from pipeline.stages.synthetic_augmenter import SyntheticAugmenter
from pipeline.stages.raw_snapshotter import RawSnapshotter
from pipeline.stages.final_snapshotter import FinalSnapshotter
from pipeline.stages.model_trainer import ModelTrainer
from pipeline.stages.model_loader import ModelLoader
from pipeline.stages.model_snapshotter import ModelSnapshotter
from pipeline.stages.hyperparam_snapshotter import HyperparamSnapshotter
from pipeline.stages.validator import Validator, RetroValidatorLogic
from pipeline.stages.validation_results_snapshotter import ValidationResultsSnapshotter


# Stage attribute names recognized on PipelineStages. Anything not in this
# set passed to __init__ is a typo and is rejected up front.
VALID_STAGE_NAMES_ = (
    "loader",
    "preprocessor",
    "raw_snapshotter",
    "engineer",
    "splitter",
    "scaler",
    "augmenter",
    "final_snapshotter",
    "trainer",
    "hyperparam_snapshotter",
    "model_snapshotter",
    "model_loader",
    "validator",
    "validation_results_snapshotter",
)


class PipelineStages:
    """Container for the stage instances wired up by `PipelineFactory`.

    Stages absent from the current scenario are not set as attributes;
    accessing them raises a descriptive error via `__getattr__`.
    """

    def __init__(self, scenario: str, **stages):
        unknown = [k for k in stages if k not in VALID_STAGE_NAMES_]
        if unknown:
            raise ValueError(
                f"Unknown stage name(s) {unknown} passed to PipelineStages. "
                f"Valid names: {list(VALID_STAGE_NAMES_)}."
            )
        # Use object.__setattr__ to make intent explicit; absent stages
        # remain genuinely missing so __getattr__ catches their access.
        object.__setattr__(self, "scenario", scenario)
        for name, stage in stages.items():
            if stage is not None:
                object.__setattr__(self, name, stage)

    def __getattr__(self, name: str):
        # Only invoked when normal attribute lookup fails. If `name` is a
        # known stage, the scenario simply did not wire it up.
        if name in VALID_STAGE_NAMES_:
            raise AttributeError(
                f"{self._stage_class_name_(name)} is not part of this pipeline "
                f"scenario ({self.scenario}). Check your PipelineFactory method."
            )
        raise AttributeError(
            f"'PipelineStages' object has no attribute {name!r}"
        )

    def __repr__(self) -> str:
        present = [n for n in VALID_STAGE_NAMES_ if n in self.__dict__]
        return f"PipelineStages(scenario={self.scenario!r}, present={present})"

    @staticmethod
    def _stage_class_name_(attr_name: str) -> str:
        """Map an attribute name to its class name for the absent-stage error."""
        return {
            "loader": "DataLoader",
            "preprocessor": "DataPreprocessor",
            "raw_snapshotter": "RawSnapshotter",
            "engineer": "FeatureEngineer",
            "splitter": "DataSplitter",
            "scaler": "Scaler",
            "augmenter": "SyntheticAugmenter",
            "final_snapshotter": "FinalSnapshotter",
            "trainer": "ModelTrainer",
            "hyperparam_snapshotter": "HyperparamSnapshotter",
            "model_snapshotter": "ModelSnapshotter",
            "model_loader": "ModelLoader",
            "validator": "Validator",
            "validation_results_snapshotter": "ValidationResultsSnapshotter",
        }[attr_name]


class PipelineFactory:
    """Static methods, one per named scenario, returning a `PipelineStages`."""

    @staticmethod
    def full_run(config: VersionConfig) -> PipelineStages:
        """Load fresh data from BQ, preprocess, engineer, split, scale,
        augment, train, validate.

        ModelLoader is wired but optional — callers that don't need
        warm-start / baseline-comparison simply do not invoke
        stages.model_loader.run(run).
        """
        logic = FeatureEngineerLogic()
        scaler = Scaler(config)
        return PipelineStages(
            scenario="full_run",
            loader=DataLoader(config, source=DataLoader.SOURCE_BQ),
            preprocessor=DataPreprocessor(config),
            raw_snapshotter=RawSnapshotter(config),
            engineer=FeatureEngineer(config, logic=logic),
            splitter=DataSplitter(config),
            scaler=scaler,
            augmenter=SyntheticAugmenter(config, scaler=scaler, logic=logic),
            final_snapshotter=FinalSnapshotter(config),
            trainer=ModelTrainer(config, scaler=scaler),
            hyperparam_snapshotter=HyperparamSnapshotter(config),
            model_snapshotter=ModelSnapshotter(config),
            model_loader=ModelLoader(config),
            validator=Validator(config),
            validation_results_snapshotter=ValidationResultsSnapshotter(config),
        )

    @staticmethod
    def retrain_existing_data(config: VersionConfig) -> PipelineStages:
        """Load from GCS parquet snapshot (no BQ). Preprocess, engineer,
        split, scale, augment, train, validate. Same stage set as full_run;
        DataLoader reads from GCS instead of BigQuery.
        """
        logic = FeatureEngineerLogic()
        scaler = Scaler(config)
        return PipelineStages(
            scenario="retrain_existing_data",
            loader=DataLoader(config, source=DataLoader.SOURCE_GCS),
            preprocessor=DataPreprocessor(config),
            raw_snapshotter=RawSnapshotter(config),
            engineer=FeatureEngineer(config, logic=logic),
            splitter=DataSplitter(config),
            scaler=scaler,
            augmenter=SyntheticAugmenter(config, scaler=scaler, logic=logic),
            final_snapshotter=FinalSnapshotter(config),
            trainer=ModelTrainer(config, scaler=scaler),
            hyperparam_snapshotter=HyperparamSnapshotter(config),
            model_snapshotter=ModelSnapshotter(config),
            model_loader=ModelLoader(config),
            validator=Validator(config),
            validation_results_snapshotter=ValidationResultsSnapshotter(config),
        )

    @staticmethod
    def tune_hyperparams(config: VersionConfig) -> PipelineStages:
        """Retrain with hyperparameter search enabled on existing data snapshot.

        Functionally identical to retrain_existing_data. Search behavior is
        controlled by config.tune_models / config.search_config, which
        ModelTrainer reads at run time — no factory-side override needed.
        """
        logic = FeatureEngineerLogic()
        scaler = Scaler(config)
        return PipelineStages(
            scenario="tune_hyperparams",
            loader=DataLoader(config, source=DataLoader.SOURCE_GCS),
            preprocessor=DataPreprocessor(config),
            raw_snapshotter=RawSnapshotter(config),
            engineer=FeatureEngineer(config, logic=logic),
            splitter=DataSplitter(config),
            scaler=scaler,
            augmenter=SyntheticAugmenter(config, scaler=scaler, logic=logic),
            final_snapshotter=FinalSnapshotter(config),
            trainer=ModelTrainer(config, scaler=scaler),
            hyperparam_snapshotter=HyperparamSnapshotter(config),
            model_snapshotter=ModelSnapshotter(config),
            model_loader=ModelLoader(config),
            validator=Validator(config),
            validation_results_snapshotter=ValidationResultsSnapshotter(config),
        )

    @staticmethod
    def validate_current(config: VersionConfig) -> PipelineStages:
        """Validation only against the current model version.

        No trainer, no augmenter — synthetic data has no place in a
        validation-only flow. ModelLoader loads config.model_version;
        Validator consumes run.models populated by it.
        """
        return PipelineStages(
            scenario="validate_current",
            loader=DataLoader(config, source=DataLoader.SOURCE_GCS),
            preprocessor=DataPreprocessor(config),
            engineer=FeatureEngineer(config),
            splitter=DataSplitter(config),
            scaler=Scaler(config),
            model_loader=ModelLoader(config),
            validator=Validator(config),
            validation_results_snapshotter=ValidationResultsSnapshotter(config),
        )

    @staticmethod
    def retro_validate(
        config: VersionConfig,
        model_versions: list[str],
    ) -> PipelineStages:
        """Replay multiple saved model versions against the locked validation set.

        ModelLoader is configured with the explicit list of versions and
        populates run.models with all of them. Validator is wired with
        RetroValidatorLogic, which iterates over the loaded versions and
        produces a per-version metrics breakdown for cross-version comparison.
        Like validate_current, no trainer / augmenter.
        """
        return PipelineStages(
            scenario="retro_validate",
            loader=DataLoader(config, source=DataLoader.SOURCE_GCS),
            preprocessor=DataPreprocessor(config),
            engineer=FeatureEngineer(config),
            splitter=DataSplitter(config),
            scaler=Scaler(config),
            model_loader=ModelLoader(config, versions=model_versions),
            validator=Validator(config, logic=RetroValidatorLogic()),
            validation_results_snapshotter=ValidationResultsSnapshotter(config),
        )
