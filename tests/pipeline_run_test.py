"""Unit tests for PipelineRun, Stage, and describe_field_."""

import pandas as pd
import pytest

from pipeline.pipeline_run import PipelineRun, Stage, describe_field_


# =========================================================================
# describe_field_
# =========================================================================

def test_describe_field_none():
    assert describe_field_(None) == "None"


def test_describe_field_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = describe_field_(df)
    assert "DataFrame" in result
    assert "2" in result   # rows
    assert "2" in result   # cols


def test_describe_field_series():
    s = pd.Series([1, 2, 3])
    result = describe_field_(s)
    assert "Series" in result
    assert "3" in result


def test_describe_field_empty_dict():
    assert describe_field_({}) == "empty dict"


def test_describe_field_populated_dict():
    result = describe_field_({"lr": 0.9, "rf": 0.8})
    assert "lr" in result
    assert "rf" in result


def test_describe_field_other_type():
    result = describe_field_(42)
    assert "int" in result


# =========================================================================
# PipelineRun.assert_ready_for — happy path
# =========================================================================

def test_assert_ready_for_load_always_passes():
    run = PipelineRun(config=None)
    run.assert_ready_for(Stage.LOAD)   # no required fields


def test_assert_ready_for_passes_when_fields_populated():
    run = PipelineRun(config=None)
    run.df_videos = pd.DataFrame({"x": [1]})
    run.df_medians = pd.DataFrame({"y": [1]})
    run.assert_ready_for(Stage.PREPROCESS)


def test_assert_ready_for_split_passes_when_engineered_present():
    run = PipelineRun(config=None)
    run.df_engineered = pd.DataFrame({"a": [1]})
    run.assert_ready_for(Stage.SPLIT)


# =========================================================================
# PipelineRun.assert_ready_for — failure path
# =========================================================================

def test_assert_ready_for_raises_runtime_error_when_field_missing():
    run = PipelineRun(config=None)
    with pytest.raises(RuntimeError, match="df_clean"):
        run.assert_ready_for(Stage.ENGINEER)


def test_assert_ready_for_error_message_names_missing_field():
    run = PipelineRun(config=None)
    with pytest.raises(RuntimeError, match="df_engineered"):
        run.assert_ready_for(Stage.SPLIT)


def test_assert_ready_for_raises_type_error_on_non_stage():
    run = PipelineRun(config=None)
    with pytest.raises(TypeError):
        run.assert_ready_for("split")


def test_assert_ready_for_type_error_mentions_stage_import():
    run = PipelineRun(config=None)
    with pytest.raises(TypeError, match="Stage"):
        run.assert_ready_for(42)


# =========================================================================
# PipelineRun.__repr__
# =========================================================================

class FakeConfig_:
    model_version = "v5.1"
    def __repr__(self): return "FakeConfig()"


def test_repr_includes_model_version_when_config_has_it():
    run = PipelineRun(config=FakeConfig_())
    assert "v5.1" in repr(run)


def test_repr_lists_populated_fields():
    run = PipelineRun(config=FakeConfig_())
    run.df_clean = pd.DataFrame({"a": [1]})
    assert "df_clean" in repr(run)


def test_repr_excludes_empty_dicts():
    run = PipelineRun(config=FakeConfig_())
    result = repr(run)
    assert "models" not in result
    assert "results" not in result


def test_repr_includes_num_synth_rows():
    run = PipelineRun(config=FakeConfig_())
    run.num_synth_rows = 500
    assert "500" in repr(run)
