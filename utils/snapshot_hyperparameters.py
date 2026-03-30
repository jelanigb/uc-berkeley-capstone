"""
Hyperparameter snapshot utility.
Saves and loads the best hyperparameter sets per model to/from GCS,
versioned alongside data and model snapshots.

Usage:
    from utils.snapshot_hyperparameters import save_hyperparams, load_hyperparams, list_hyperparams

    params = {
        'LogisticRegression':     {'C': 0.1, 'penalty': 'l1', ...},
        'RandomForestClassifier': {'n_estimators': 300, ...},
        'XGBClassifier':          {'max_depth': 5, ...},
    }
    search_config = {'strategy': 'random', 'n_iter': 50, 'cv': 5, 'scoring': 'roc_auc'}
    save_hyperparams(params, version_tag="v3.2", search_config=search_config)
    stored = load_hyperparams("v3.2")
"""

import json
from datetime import datetime
from google.cloud import storage

PROJECT_ID  = "maduros-dolce"
BUCKET_NAME = "maduros-dolce-capstone-data"
GCS_PREFIX_ = "hyperparams"


def version_tag_exists_(bucket, version_tag: str) -> bool:
    blobs = list(bucket.list_blobs(prefix=f"{GCS_PREFIX_}/{version_tag}.json"))
    return len(blobs) > 0


def save_hyperparams(
    params: dict,
    version_tag: str,
    search_config: dict = None,
    notes: str = "",
    overwrite: bool = False,
) -> dict:
    """
    Save a hyperparameter set to GCS.

    Parameters
    ----------
    params        : {model_class_name: {param: value, ...}, ...}
    version_tag   : version string, e.g. "v3.2"
    search_config : dict capturing how the search was run, e.g.
                    {'strategy': 'random', 'n_iter': 50, 'cv': 5, 'scoring': 'roc_auc'}
                    Pass None (default) when recording pre-tuning baseline params.
    notes         : free-text notes about the tuning run
    overwrite     : bypass existence check (use after interrupted runs)

    Returns the metadata dict.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    if not overwrite and version_tag_exists_(bucket, version_tag):
        raise ValueError(
            f"Hyperparameter snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag, pass overwrite=True, or delete the existing snapshot first."
        )

    now = datetime.utcnow()
    payload = {
        "version_tag":   version_tag,
        "saved_at":      now.isoformat(),
        "notes":         notes,
        "search_config": search_config or {"strategy": "none", "note": "pre-tuning baseline"},
        "models":        list(params.keys()),
        "params":        params,
    }

    blob_path = f"{GCS_PREFIX_}/{version_tag}.json"
    bucket.blob(blob_path).upload_from_string(
        json.dumps(payload, indent=2),
        content_type="application/json",
    )
    print(f"Saved hyperparams '{version_tag}' → gs://{BUCKET_NAME}/{blob_path}")
    print(f"  Models: {list(params.keys())}")
    print(f"  Search: {payload['search_config']}")
    return payload


def load_hyperparams(version_tag: str) -> dict:
    """
    Load a hyperparameter set from GCS.
    Returns the full payload dict; access params via result['params']['XGBClassifier'].

    Raises FileNotFoundError if the version tag does not exist.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{GCS_PREFIX_}/{version_tag}.json")

    if not blob.exists():
        raise FileNotFoundError(
            f"No hyperparameter snapshot found for version tag '{version_tag}'. "
            "Run with tune_models=True first, or check available versions with list_hyperparams()."
        )

    payload = json.loads(blob.download_as_text())
    print(f"Loaded hyperparams '{version_tag}' (saved {payload['saved_at'][:10]})")
    print(f"  Models: {payload['models']}")
    print(f"  Search: {payload.get('search_config', 'not recorded')}")
    return payload


def list_hyperparams():
    """List all available hyperparameter snapshots in GCS."""
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = [b for b in bucket.list_blobs(prefix=f"{GCS_PREFIX_}/")
             if b.name.endswith(".json")]

    print(f"Found {len(blobs)} hyperparameter snapshots:\n")
    for blob in sorted(blobs, key=lambda b: b.name):
        payload = json.loads(blob.download_as_text())
        sc = payload.get('search_config', {})
        print(f"  {payload['version_tag']}  |  {payload['saved_at'][:10]}  |  search: {sc.get('strategy', '?')}")
        if payload.get('notes'):
            print(f"    Notes: {payload['notes']}")
        for model, p in payload.get('params', {}).items():
            print(f"    {model}: {p}")
        print()
