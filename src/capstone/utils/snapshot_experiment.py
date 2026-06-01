"""
snapshot_experiment — preserve exploratory experiment outputs to GCS.

Experiments that are *not* promoted to a model version (e.g. the tier-split
comparison, see docs/tier_split_models.md) still need to be preserved for later
comparison — kept separate from the canonical models/{version}/ history so they
never pollute the cross-version model trajectory. This module writes them under a
parallel `experiments/` prefix in the same bucket, reusing the append-only-JSONL
idiom of utils.snapshot_model.save_validation_results.

Layout
------
    experiments/{experiment}/results.jsonl
        Append-only; one JSON line per run (run_timestamp-keyed). Each record
        carries pooled metrics, the segment audit, and provenance.
    experiments/{experiment}/models/{run_id}/
        Pickled sub-models (joblib), the fitted scaler, and a manifest.json —
        written only when save_models=True. The record's "models_path" points
        here so results and artifacts stay linked.

Split design: build_tier_split_record_ assembles the JSON record with no GCS
calls (unit-tested); save_tier_split_experiment is the thin GCS pass-through
(tested manually, per the repo convention for GCS-touching functions).
"""

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from google.cloud import storage

from constants import PROJECT_ID, BUCKET_NAME

EXPERIMENTS_PREFIX_ = "experiments"


def run_id_from_(now: datetime) -> str:
    """GCS-safe run id derived from a UTC timestamp, e.g. '20260530T043552Z'."""
    return now.strftime("%Y%m%dT%H%M%SZ")


def build_tier_split_record_(
    pooled_metrics: dict,
    segment_audit: pd.DataFrame,
    feature_cols: list,
    data_version: str,
    basis_model_version: str,
    train_rows_per_tier: dict,
    now: datetime,
    models_path: str = None,
    status: str = "rejected",
    hyperparams: str = "transferred_from_global_untuned",
    notes: str = "",
    experiment: str = "tier_split",
) -> dict:
    """Assemble the JSON-serializable experiment record. No GCS calls — pure
    logic so it can be unit-tested. `segment_audit` is flattened to records;
    `models_path` is the gs:// dir for the pickled sub-models (None if not saved).
    """
    return {
        "run_timestamp": now.isoformat() + "Z",
        "run_id": run_id_from_(now),
        "experiment": experiment,
        "status": status,
        "basis_model_version": basis_model_version,
        "data_version": data_version,
        "hyperparams": hyperparams,
        "train_rows_per_tier": train_rows_per_tier,
        "feature_cols": feature_cols,
        "models_path": models_path,
        "pooled_metrics": pooled_metrics,
        "segment_audit": segment_audit.to_dict(orient="records"),
        "notes": notes,
    }


def save_tier_split_experiment(
    pooled_metrics: dict,
    segment_audit: pd.DataFrame,
    tier_models: dict,
    feature_cols: list,
    data_version: str,
    basis_model_version: str,
    train_rows_per_tier: dict,
    scaler=None,
    status: str = "rejected",
    hyperparams: str = "transferred_from_global_untuned",
    notes: str = "",
    experiment: str = "tier_split",
    save_models: bool = True,
) -> dict:
    """Append one experiment record to experiments/{experiment}/results.jsonl and
    (optionally) pickle the per-tier sub-models + scaler.

    Parameters
    ----------
    pooled_metrics : dict
        ``{model_name: metrics_dict}`` — the routed-prediction comparison table
        (globals + tier variants).
    segment_audit : pd.DataFrame
        The SegmentAuditor output for the same models.
    tier_models : dict
        ``{family: {tier: fitted_model}}`` — the sub-models to pickle.
    feature_cols : list
        Column order the sub-models expect (scaled with `scaler`).
    scaler : optional
        The fitted StandardScaler the sub-models were trained against. Saved
        alongside the models so the experiment dir is self-contained for later
        re-evaluation (e.g. OOB). Skipped if None.
    save_models : bool
        When False, only the JSONL record is written (no pickles).

    Returns the written record dict.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    now = datetime.utcnow()
    run_id = run_id_from_(now)

    models_path = None
    if save_models:
        models_path = _save_sub_models_(
            bucket, experiment, run_id, tier_models, feature_cols, scaler
        )

    record = build_tier_split_record_(
        pooled_metrics=pooled_metrics,
        segment_audit=segment_audit,
        feature_cols=feature_cols,
        data_version=data_version,
        basis_model_version=basis_model_version,
        train_rows_per_tier=train_rows_per_tier,
        now=now,
        models_path=models_path,
        status=status,
        hyperparams=hyperparams,
        notes=notes,
        experiment=experiment,
    )

    blob_path = f"{EXPERIMENTS_PREFIX_}/{experiment}/results.jsonl"
    blob = bucket.blob(blob_path)
    existing = ""
    if blob.exists():
        existing = blob.download_as_text()
        if existing and not existing.endswith("\n"):
            existing += "\n"
    blob.upload_from_string(
        existing + json.dumps(record) + "\n",
        content_type="application/json",
    )

    print(f"Experiment record appended → gs://{BUCKET_NAME}/{blob_path}")
    if models_path:
        print(f"Sub-models + scaler saved → {models_path}")
    return record


def _save_sub_models_(bucket, experiment, run_id, tier_models, feature_cols, scaler):
    """Pickle each sub-model (and the scaler) to GCS; write a manifest. Returns
    the gs:// dir path."""
    base = f"{EXPERIMENTS_PREFIX_}/{experiment}/models/{run_id}"
    local_dir = f"/tmp/{base}"
    os.makedirs(local_dir, exist_ok=True)

    saved = []
    for family, per_tier in tier_models.items():
        for tier, model in per_tier.items():
            fname = f"{family}_{tier}.pkl"
            local_path = f"{local_dir}/{fname}"
            joblib.dump(model, local_path)
            bucket.blob(f"{base}/{fname}").upload_from_filename(local_path, timeout=600)
            saved.append(fname)

    if scaler is not None:
        local_scaler = f"{local_dir}/scaler.pkl"
        joblib.dump(scaler, local_scaler)
        bucket.blob(f"{base}/scaler.pkl").upload_from_filename(local_scaler)

    manifest = {
        "run_id": run_id,
        "experiment": experiment,
        "families": sorted(tier_models),
        "tiers": sorted({t for per_tier in tier_models.values() for t in per_tier}),
        "sub_models": saved,
        "scaler": "scaler.pkl" if scaler is not None else None,
        "feature_cols": feature_cols,
        "note": (
            "Each sub-model is fitted on a single tier's rows. Reconstruct routing "
            "with modeling.tier_routing.TierRoutedClassifier; inputs must be the "
            "feature_cols above in order, scaled with scaler.pkl."
        ),
    }
    local_manifest = f"{local_dir}/manifest.json"
    with open(local_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    bucket.blob(f"{base}/manifest.json").upload_from_filename(local_manifest)

    return f"gs://{BUCKET_NAME}/{base}"


def load_tier_split_experiments(experiment: str = "tier_split") -> pd.DataFrame:
    """Load experiment history into a flat DataFrame: one row per (run, model)
    from each record's pooled_metrics, with run_timestamp / data_version /
    basis_model_version / status attached for comparison. The wide fields
    (segment_audit, feature_cols) are omitted here — read the raw JSONL for those.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{EXPERIMENTS_PREFIX_}/{experiment}/results.jsonl")
    if not blob.exists():
        return pd.DataFrame()

    rows = []
    for line in blob.download_as_text().strip().split("\n"):
        if not line:
            continue
        rec = json.loads(line)
        for model_name, metrics in rec.get("pooled_metrics", {}).items():
            rows.append({
                "run_timestamp": rec["run_timestamp"],
                "basis_model_version": rec.get("basis_model_version"),
                "data_version": rec.get("data_version"),
                "status": rec.get("status"),
                "model": model_name,
                **{k: v for k, v in metrics.items()
                   if k not in ("confusion_matrix", "top_features")},
            })
    return pd.DataFrame(rows)
