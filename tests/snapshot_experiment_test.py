"""Unit tests for utils.snapshot_experiment pure-logic helpers.

Per the repo convention, GCS-touching functions (save_tier_split_experiment,
load_tier_split_experiments) are exercised manually; these tests cover the
record assembly and run-id formatting, plus that the assembled record is
JSON-serializable (it is written to GCS verbatim via json.dumps).
"""

import json
from datetime import datetime

import pandas as pd

from utils.snapshot_experiment import build_tier_split_record_, run_id_from_


def test_run_id_from_formats_compact_utc():
    assert run_id_from_(datetime(2026, 5, 30, 4, 35, 52)) == "20260530T043552Z"


def _record():
    pooled = {
        "xgb": {"roc_auc": 0.9200, "accuracy": 0.8432, "confusion_matrix": [[1, 2], [3, 4]]},
        "xgb_tier_full": {"roc_auc": 0.9160, "accuracy": 0.8415},
    }
    audit = pd.DataFrame([
        {"model": "xgb", "segment_type": "tier", "segment_value": "S", "roc_auc": 0.8821},
        {"model": "xgb_tier_full", "segment_type": "tier", "segment_value": "S", "roc_auc": 0.8720},
    ])
    return build_tier_split_record_(
        pooled_metrics=pooled,
        segment_audit=audit,
        feature_cols=["a", "b", "c"],
        data_version="v3.5_real",
        basis_model_version="v6.2",
        train_rows_per_tier={"S": 5773, "M": 6833, "L": 6983},
        now=datetime(2026, 5, 30, 4, 35, 52),
        models_path="gs://bucket/experiments/tier_split/models/20260530T043552Z",
    )


def test_build_record_has_expected_shape():
    rec = _record()
    assert rec["run_id"] == "20260530T043552Z"
    assert rec["run_timestamp"] == "2026-05-30T04:35:52Z"
    assert rec["experiment"] == "tier_split"
    assert rec["status"] == "rejected"  # default
    assert rec["hyperparams"] == "transferred_from_global_untuned"  # default
    assert rec["basis_model_version"] == "v6.2"
    assert rec["data_version"] == "v3.5_real"
    assert rec["train_rows_per_tier"]["S"] == 5773
    assert rec["models_path"].endswith("20260530T043552Z")


def test_build_record_flattens_segment_audit_to_records():
    rec = _record()
    assert isinstance(rec["segment_audit"], list)
    assert rec["segment_audit"][0] == {
        "model": "xgb", "segment_type": "tier", "segment_value": "S", "roc_auc": 0.8821,
    }


def test_build_record_is_json_serializable():
    # The record is written to GCS via json.dumps — it must round-trip cleanly.
    rec = _record()
    restored = json.loads(json.dumps(rec))
    assert restored["pooled_metrics"]["xgb"]["roc_auc"] == 0.9200
