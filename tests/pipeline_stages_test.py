"""Unit tests for PipelineStages in pipeline/factory.py.

PipelineFactory methods are not tested here because they call stage
constructors that require GCS-backed VersionConfig. PipelineStages itself
is testable in isolation by passing plain sentinel objects as stages.
"""

import pytest

from pipeline.factory import PipelineStages, VALID_STAGE_NAMES_


class FakeStage:
    """Stand-in for any real stage instance."""


# =========================================================================
# PipelineStages.__init__
# =========================================================================

def test_init_accepts_all_valid_stage_names():
    stages = {name: FakeStage() for name in VALID_STAGE_NAMES_}
    ps = PipelineStages(scenario="test", **stages)
    for name in VALID_STAGE_NAMES_:
        assert hasattr(ps, name)


def test_init_unknown_stage_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown stage name"):
        PipelineStages(scenario="test", totally_fake_stage=FakeStage())


def test_init_none_valued_stage_is_not_set_as_attribute():
    ps = PipelineStages(scenario="test", loader=None)
    assert "loader" not in ps.__dict__


def test_init_non_none_stage_is_set():
    stage = FakeStage()
    ps = PipelineStages(scenario="test", loader=stage)
    assert ps.loader is stage


def test_init_scenario_is_stored():
    ps = PipelineStages(scenario="full_run")
    assert ps.scenario == "full_run"


def test_init_subset_of_valid_stages_accepted():
    ps = PipelineStages(
        scenario="validate_current",
        loader=FakeStage(),
        validator=FakeStage(),
    )
    assert hasattr(ps, "loader")
    assert hasattr(ps, "validator")


# =========================================================================
# PipelineStages.__getattr__ — absent stages
# =========================================================================

def test_getattr_absent_known_stage_raises_attribute_error():
    ps = PipelineStages(scenario="validate_current", loader=FakeStage())
    with pytest.raises(AttributeError):
        _ = ps.trainer


def test_getattr_absent_stage_error_mentions_scenario():
    ps = PipelineStages(scenario="validate_current", loader=FakeStage())
    with pytest.raises(AttributeError, match="validate_current"):
        _ = ps.trainer


def test_getattr_absent_stage_error_mentions_class_name():
    ps = PipelineStages(scenario="validate_current", loader=FakeStage())
    with pytest.raises(AttributeError, match="ModelTrainer"):
        _ = ps.trainer


def test_getattr_completely_unknown_attr_raises_generic_attribute_error():
    ps = PipelineStages(scenario="test")
    with pytest.raises(AttributeError, match="no attribute"):
        _ = ps.nonexistent_attr_xyz


# =========================================================================
# PipelineStages.__repr__
# =========================================================================

def test_repr_includes_scenario():
    ps = PipelineStages(scenario="full_run", loader=FakeStage())
    assert "full_run" in repr(ps)


def test_repr_lists_present_stages():
    ps = PipelineStages(scenario="test", loader=FakeStage(), validator=FakeStage())
    r = repr(ps)
    assert "loader" in r
    assert "validator" in r


def test_repr_excludes_absent_stages():
    ps = PipelineStages(scenario="test", loader=FakeStage())
    r = repr(ps)
    assert "trainer" not in r


# =========================================================================
# PipelineStages._stage_class_name_
# =========================================================================

@pytest.mark.parametrize("attr,cls_name", [
    ("loader",                        "DataLoader"),
    ("preprocessor",                  "DataPreprocessor"),
    ("engineer",                      "FeatureEngineer"),
    ("splitter",                      "DataSplitter"),
    ("scaler",                        "Scaler"),
    ("augmenter",                     "SyntheticAugmenter"),
    ("trainer",                       "ModelTrainer"),
    ("model_loader",                  "ModelLoader"),
    ("validator",                     "Validator"),
    ("validation_results_snapshotter","ValidationResultsSnapshotter"),
])
def test_stage_class_name_mapping(attr, cls_name):
    assert PipelineStages._stage_class_name_(attr) == cls_name
