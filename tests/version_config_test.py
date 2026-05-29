"""Unit tests for VersionConfig builder / compute methods.

VersionConfig.load() and .commit() touch GCS and are not tested here.
Everything else — builder methods, build(), compute_new_version_,
parse_version_, validate_pin_conflicts_, migrate_flat_to_nested_ — is
pure in-memory logic and is exercised below.
"""

import copy
import pytest

from pipeline.version_config import (
    VersionConfig,
    Flag,
    DEFAULT_STATE_,
)


# =========================================================================
# Helpers
# =========================================================================

def config_(use_synthetic=False, **kwargs) -> VersionConfig:
    """Construct a VersionConfig from DEFAULT_STATE_ without GCS."""
    return VersionConfig(
        state=copy.deepcopy(DEFAULT_STATE_),
        use_synthetic=use_synthetic,
        **kwargs,
    )


# =========================================================================
# parse_version_
# =========================================================================

@pytest.mark.parametrize("version_str,expected", [
    ("3.1",  (3, 1)),
    ("v3.1", (3, 1)),
    ("0.0",  (0, 0)),
    ("10.20",(10, 20)),
    ("v10.20",(10,20)),
])
def test_parse_version_valid(version_str, expected):
    assert VersionConfig.parse_version_(version_str) == expected


@pytest.mark.parametrize("bad", ["3", "3.", ".1", "abc", "v3", ""])
def test_parse_version_invalid_raises(bad):
    with pytest.raises(ValueError, match="Cannot parse version"):
        VersionConfig.parse_version_(bad)


# =========================================================================
# compute_new_version_
# =========================================================================

def test_compute_new_version_no_write_flag_unchanged():
    assert VersionConfig.compute_new_version_((3, 1), write_flag=False, major_flag=False) == (3, 1)


def test_compute_new_version_write_increments_minor():
    assert VersionConfig.compute_new_version_((3, 1), write_flag=True, major_flag=False) == (3, 2)


def test_compute_new_version_major_bump_resets_minor():
    assert VersionConfig.compute_new_version_((3, 1), write_flag=True, major_flag=True) == (4, 0)


def test_compute_new_version_major_flag_without_write_is_no_op():
    # major_flag alone has no effect — write_flag gates the bump
    assert VersionConfig.compute_new_version_((3, 1), write_flag=False, major_flag=True) == (3, 1)


# =========================================================================
# migrate_flat_to_nested_
# =========================================================================

def test_migrate_flat_to_nested_produces_nested_schema():
    flat = {"version": "3.1", "raw_suffix": "real", "mixed_suffix": "mixed_80real"}
    result = VersionConfig.migrate_flat_to_nested_(flat)
    assert result["data"]["major"] == 3
    assert result["data"]["minor"] == 1
    assert result["model"]["major"] == 3
    assert result["hyperparams"] == {"major": 1, "minor": 0}


def test_migrate_flat_to_nested_preserves_raw_suffix():
    flat = {"version": "3.1", "raw_suffix": "custom", "mixed_suffix": "mixed_80real"}
    result = VersionConfig.migrate_flat_to_nested_(flat)
    assert result["data"]["raw_suffix"] == "custom"


def test_migrate_flat_to_nested_missing_suffix_defaults():
    flat = {"version": "2.0"}
    result = VersionConfig.migrate_flat_to_nested_(flat)
    assert result["data"]["raw_suffix"] == "real"


# =========================================================================
# Builder methods — flag state
# =========================================================================

def test_snapshot_raw_sets_raw_and_baselines_flags():
    c = config_()
    c.snapshot_raw()
    assert c.flags_[Flag.RAW] is True
    assert c.flags_[Flag.BASELINES_MAJOR] is True


def test_snapshot_final_sets_final_flag_only():
    c = config_()
    c.snapshot_final()
    assert c.flags_[Flag.FINAL] is True
    assert c.flags_[Flag.RAW] is False


def test_snapshot_schema_change_sets_data_major_and_baselines():
    c = config_()
    c.snapshot_schema_change()
    assert c.flags_[Flag.DATA_MAJOR] is True
    assert c.flags_[Flag.BASELINES_MAJOR] is True


def test_snapshot_models_sets_models_flag():
    c = config_()
    c.snapshot_models()
    assert c.flags_[Flag.MODELS] is True
    assert c.flags_[Flag.MODEL_MAJOR] is False


def test_snapshot_models_new_data_sets_both_model_flags():
    c = config_()
    c.snapshot_models_new_data()
    assert c.flags_[Flag.MODELS] is True
    assert c.flags_[Flag.MODEL_MAJOR] is True


def test_snapshot_hyperparams_sets_hyperparams_flag():
    c = config_()
    c.snapshot_hyperparams()
    assert c.flags_[Flag.HYPERPARAMS] is True
    assert c.flags_[Flag.HYPERPARAMS_MAJOR] is False


def test_snapshot_hyperparams_new_grid_sets_both_hyperparam_flags():
    c = config_()
    c.snapshot_hyperparams_new_grid()
    assert c.flags_[Flag.HYPERPARAMS] is True
    assert c.flags_[Flag.HYPERPARAMS_MAJOR] is True


def test_tune_sets_tune_flag_and_stores_search_config():
    c = config_()
    c.tune(strategy="halving", n_iter=25, cv=3, scoring="f1")
    assert c.flags_[Flag.TUNE] is True
    assert c.search_strategy == "halving"
    assert c.search_n_iter == 25
    assert c.search_cv == 3
    assert c.search_scoring == "f1"


def test_builder_methods_return_self_for_chaining():
    c = config_()
    result = c.snapshot_raw().snapshot_hyperparams()
    assert result is c


# =========================================================================
# use_*_version — pinning
# =========================================================================

def test_use_data_version_stores_parsed_tuple():
    c = config_()
    c.use_data_version("v2.5")
    assert c.pinned_data_version_ == (2, 5)


def test_use_model_version_stores_parsed_tuple():
    c = config_()
    c.use_model_version("4.2")
    assert c.pinned_model_version_ == (4, 2)


def test_use_hyperparam_version_stores_parsed_tuple():
    c = config_()
    c.use_hyperparam_version("v1.3")
    assert c.pinned_hyperparam_version_ == (1, 3)


# =========================================================================
# validate_pin_conflicts_
# =========================================================================

def test_pin_data_with_snapshot_raw_raises():
    c = config_()
    c.use_data_version("v2.5")
    c.snapshot_raw()
    with pytest.raises(ValueError, match="pin"):
        c.validate_pin_conflicts_()


def test_pin_data_with_snapshot_final_raises():
    c = config_()
    c.use_data_version("v2.5")
    c.snapshot_final()
    with pytest.raises(ValueError):
        c.validate_pin_conflicts_()


def test_pin_model_with_snapshot_models_raises():
    c = config_()
    c.use_model_version("v4.1")
    c.snapshot_models()
    with pytest.raises(ValueError, match="pin"):
        c.validate_pin_conflicts_()


def test_pin_hyperparams_with_snapshot_hyperparams_raises():
    c = config_()
    c.use_hyperparam_version("v1.2")
    c.snapshot_hyperparams()
    with pytest.raises(ValueError):
        c.validate_pin_conflicts_()


def test_no_conflict_with_no_flags():
    c = config_()
    c.use_data_version("v2.5")
    c.validate_pin_conflicts_()  # should not raise


# =========================================================================
# build() — version string computation
# =========================================================================

def test_build_no_flags_versions_unchanged():
    c = config_().build()
    state = DEFAULT_STATE_
    d, m = state["data"], state["model"]
    assert c.raw_version == f"v{d['major']}.{d['minor']}_{d['raw_suffix']}"
    assert c.model_version == f"v{m['major']}.{m['minor']}"


def test_build_snapshot_models_minor_bump():
    c = config_().snapshot_models().build()
    state = DEFAULT_STATE_
    m = state["model"]
    assert c.next_model_version == f"v{m['major']}.{m['minor'] + 1}"


def test_build_snapshot_models_new_data_major_bump():
    c = config_().snapshot_models_new_data().build()
    state = DEFAULT_STATE_
    m = state["model"]
    assert c.next_model_version == f"v{m['major'] + 1}.0"


def test_build_snapshot_raw_minor_data_bump():
    c = config_().snapshot_raw().build()
    state = DEFAULT_STATE_
    d = state["data"]
    assert c.next_raw_version == f"v{d['major']}.{d['minor'] + 1}_{d['raw_suffix']}"


def test_build_synthetic_final_version_contains_mixed_suffix():
    c = config_(use_synthetic=True, target_real_pct=0.8)
    c.snapshot_final().build()
    assert "mixed_80real" in c.next_final_version


def test_build_no_synthetic_final_version_is_100real():
    c = config_().snapshot_final().build()
    assert "100real" in c.next_final_version


def test_build_pinned_model_version_overrides_bump():
    c = config_()
    c.use_model_version("v2.0").build()
    assert c.model_version == "v2.0"


def test_build_sets_built_flag():
    c = config_().build()
    assert c.built_ is True


def test_commit_raises_if_not_built():
    c = config_()
    with pytest.raises(RuntimeError, match="build"):
        c.commit()


# =========================================================================
# target_real_pct validation
# =========================================================================

def test_target_real_pct_zero_raises():
    with pytest.raises(ValueError, match="target_real_pct"):
        config_(use_synthetic=True, target_real_pct=0.0)


def test_target_real_pct_one_raises():
    with pytest.raises(ValueError, match="target_real_pct"):
        config_(use_synthetic=True, target_real_pct=1.0)


def test_target_real_pct_valid_range_accepted():
    c = config_(use_synthetic=True, target_real_pct=0.8)
    assert c.target_real_pct == 0.8


def test_target_real_pct_none_when_not_synthetic():
    c = config_(use_synthetic=False)
    assert c.target_real_pct is None


# =========================================================================
# dry_run guardrail
# =========================================================================

def test_dry_run_defaults_false():
    assert config_().is_dry_run is False


def test_dry_run_sets_flag_and_returns_self():
    c = config_()
    result = c.dry_run()
    assert result is c
    assert c.is_dry_run is True


def test_dry_run_explicit_false():
    c = config_().dry_run(True).dry_run(False)
    assert c.is_dry_run is False


def test_commit_in_dry_run_skips_without_gcs(capsys):
    # dry_run commit must short-circuit before any GCS access and not raise.
    c = config_().snapshot_models().dry_run().build()
    c.commit()
    out = capsys.readouterr().out
    assert "dry_run" in out.lower()


def test_dry_run_chains_with_other_builders():
    c = config_().snapshot_models().dry_run().build()
    assert c.is_dry_run is True
    assert c.flags_[Flag.MODELS] is True


# =========================================================================
# tune(models=...) — per-model search subset
# =========================================================================

def test_tune_models_none_tunes_all():
    c = config_().tune()
    assert c.tune_model_names is None


def test_tune_models_normalizes_aliases():
    c = config_().tune(models=["xgb", "rf"])
    assert c.tune_model_names == ["XGBoost", "RandomForest"]


def test_tune_models_accepts_lr_l1_alias():
    c = config_().tune(models=["lr_l1"])
    assert c.tune_model_names == ["LogisticRegression"]


def test_tune_models_dedupes():
    c = config_().tune(models=["xgb", "XGBoost", "xgbclassifier"])
    assert c.tune_model_names == ["XGBoost"]


def test_tune_models_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model name"):
        config_().tune(models=["catboost"])


def test_normalize_model_names_none_returns_none():
    assert VersionConfig.normalize_model_names_(None) is None


# =========================================================================
# search_config property
# =========================================================================

def test_search_config_returns_dict_with_expected_keys():
    c = config_().tune(strategy="grid", n_iter=10, cv=3, scoring="accuracy")
    sc = c.search_config
    assert sc["strategy"] == "grid"
    assert sc["n_iter"] == 10
    assert sc["cv"] == 3
    assert sc["scoring"] == "accuracy"
