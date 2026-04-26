import json
from datetime import datetime
from google.cloud import bigquery, storage
import pandas as pd

from constants import PROJECT_ID, BUCKET_NAME

DATASET_ID = "capstone_youtube"

# Add this query helper near the top, after the constants

BASELINE_QUERY = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.channel_baseline_videos`
"""

BASELINE_MEDIANS_QUERY = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.channel_baseline_medians`
"""


def _version_tag_exists(bucket, version_tag: str, snapshot_type: str) -> bool:
    """
    Check whether a snapshot with this version_tag already exists in GCS.

    snapshot_type options:
      "raw_video"  — BQ video_snapshots pulls (snapshot_video_data / save_video_snapshot)
      "final"      — final training dataset saves (save_snapshot)
      "baselines"  — channel baseline pulls (snapshot_baselines / save_baselines_snapshot)
      "splits"     — per-split parquet saves (save_splits_snapshot)
    """
    prefix_map = {
        "raw_video": f"snapshots/snapshots_{version_tag}_",
        "final":     f"snapshots/snapshots_{version_tag}_",
        "baselines": f"snapshots/baselines_{version_tag}_",
        "splits":    f"snapshots/splits_{version_tag}_",
    }
    prefix = prefix_map[snapshot_type]
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0


def snapshot_baselines(version_tag: str, notes: str = ""):
    """
    Pull channel_baseline_videos and channel_baseline_medians from BQ,
    save as Parquet, upload to GCS under the same version tag.

    Usage:
        df_baselines, df_medians, meta = snapshot_baselines("v1.0_real")
    """
    bq_client = bigquery.Client(project=PROJECT_ID)
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    if _version_tag_exists(bucket, version_tag, "baselines"):
        raise ValueError(
            f"Baseline snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag or delete the existing snapshot first."
        )

    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # --- Pull baseline videos ---
    df_baselines = bq_client.query(BASELINE_QUERY).to_dataframe()
    print(
        f"Pulled {len(df_baselines)} baseline video rows "
        f"({df_baselines['channel_id'].nunique()} channels)"
    )

    # --- Pull baseline medians ---
    df_medians = bq_client.query(BASELINE_MEDIANS_QUERY).to_dataframe()
    print(f"Pulled {len(df_medians)} baseline median rows")

    # --- Save and upload baseline videos ---
    baselines_name = (
        f"baselines_{version_tag}_{len(df_baselines)}rows_{timestamp}"
    )
    local_baselines = f"/tmp/{baselines_name}.parquet"
    df_baselines.to_parquet(local_baselines, index=False)
    bucket.blob(f"snapshots/{baselines_name}.parquet").upload_from_filename(
        local_baselines
    )
    print(f"Uploaded gs://{BUCKET_NAME}/snapshots/{baselines_name}.parquet")

    # --- Save and upload baseline medians ---
    medians_name = f"medians_{version_tag}_{len(df_medians)}rows_{timestamp}"
    local_medians = f"/tmp/{medians_name}.parquet"
    df_medians.to_parquet(local_medians, index=False)
    bucket.blob(f"snapshots/{medians_name}.parquet").upload_from_filename(
        local_medians
    )
    print(f"Uploaded gs://{BUCKET_NAME}/snapshots/{medians_name}.parquet")

    # --- Metadata ---
    metadata = {
        "version_tag": version_tag,
        "snapshot_timestamp": now.isoformat(),
        "baseline_video_rows": len(df_baselines),
        "baseline_median_rows": len(df_medians),
        "unique_channels": df_baselines["channel_id"].nunique(),
        "baselines_file": f"gs://{BUCKET_NAME}/snapshots/{baselines_name}.parquet",
        "medians_file": f"gs://{BUCKET_NAME}/snapshots/{medians_name}.parquet",
        "notes": notes,
    }

    meta_name = f"baselines_{version_tag}_{timestamp}_meta.json"
    local_meta = f"/tmp/{meta_name}"
    with open(local_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    bucket.blob(f"snapshots/{meta_name}").upload_from_filename(local_meta)

    print(f"\n--- Baseline Snapshot {version_tag} ---")
    print(
        f"  Baseline videos: {len(df_baselines)} ({df_baselines['channel_id'].nunique()} channels)"
    )
    print(f"  Baseline medians: {len(df_medians)}")

    return df_baselines, df_medians, metadata


def load_baselines(version_tag: str):
    """
    Load baseline videos and medians from GCS by version tag.
    Returns (df_baselines, df_medians, metadata).

    Usage:
        df_baselines, df_medians, meta = load_baselines("v1.0_real")
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="snapshots/"))

    # Find matching files
    baseline_blobs = [
        b
        for b in blobs
        if b.name.endswith(".parquet") and f"baselines_{version_tag}_" in b.name
    ]
    median_blobs = [
        b
        for b in blobs
        if b.name.endswith(".parquet") and f"medians_{version_tag}_" in b.name
    ]
    meta_blobs = [
        b
        for b in blobs
        if b.name.endswith("_meta.json")
        and f"baselines_{version_tag}_" in b.name
    ]

    if not baseline_blobs or not median_blobs:
        raise FileNotFoundError(
            f"No baseline snapshot found for version tag '{version_tag}'"
        )

    # Take most recent of each
    baseline_blob = sorted(baseline_blobs, key=lambda b: b.name)[-1]
    median_blob = sorted(median_blobs, key=lambda b: b.name)[-1]

    local_baselines = f"/tmp/{baseline_blob.name.split('/')[-1]}"
    baseline_blob.download_to_filename(local_baselines)
    df_baselines = pd.read_parquet(local_baselines)

    local_medians = f"/tmp/{median_blob.name.split('/')[-1]}"
    median_blob.download_to_filename(local_medians)
    df_medians = pd.read_parquet(local_medians)

    # Load metadata if available
    metadata = {}
    if meta_blobs:
        meta_blob = sorted(meta_blobs, key=lambda b: b.name)[-1]
        metadata = json.loads(meta_blob.download_as_text())

    print(
        f"Loaded baselines '{version_tag}': "
        f"{len(df_baselines)} baseline videos, "
        f"{len(df_medians)} median rows "
        f"({df_baselines['channel_id'].nunique()} channels)"
    )

    return df_baselines, df_medians, metadata


def snapshot_video_data(version_tag: str, notes: str = ""):
    """
    Pull current video_snapshots from BQ, save as Parquet,
    upload to GCS with a metadata sidecar JSON.

    Usage:
        snapshot_video_data("v1.0_real", notes="First real-only dataset")
        snapshot_video_data("v1.2_mixed30", notes="30% real, 70% synthetic")
    """
    bq_client = bigquery.Client(project=PROJECT_ID)
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    if _version_tag_exists(bucket, version_tag, "raw_video"):
        raise ValueError(
            f"Video snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag or delete the existing snapshot first."
        )

    now = datetime.utcnow()

    # --- Pull data ---
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.video_snapshots`
        WHERE poll_label = '7d'
    """
    df = bq_client.query(query).to_dataframe()

    # --- Build filenames ---
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    base_name = f"snapshots_{version_tag}_{len(df)}rows_{timestamp}"
    parquet_filename = f"{base_name}.parquet"
    meta_filename = f"{base_name}_meta.json"

    # --- Save Parquet locally ---
    local_parquet = f"/tmp/{parquet_filename}"
    df.to_parquet(local_parquet, index=False)
    print(f"Saved {len(df)} rows to {local_parquet}")

    # --- Build metadata ---
    poll_counts = df["poll_label"].value_counts().to_dict()
    vertical_counts = df["vertical"].value_counts().to_dict()
    tier_counts = df["tier"].value_counts().to_dict()

    metadata = {
        "version_tag": version_tag,
        "snapshot_timestamp": now.isoformat(),
        "total_rows": len(df),
        "unique_videos": df["video_id"].nunique(),
        "unique_channels": df["channel_id"].nunique(),
        "poll_label_counts": poll_counts,
        "vertical_counts": vertical_counts,
        "tier_counts": tier_counts,
        "date_range": {
            "earliest_publish": (
                str(df["published_at"].dropna().min())
                if "published_at" in df.columns
                else ""
            ),
            "latest_publish": (
                str(df["published_at"].dropna().max())
                if "published_at" in df.columns
                else ""
            ),
        },
        "columns": df.columns.tolist(),
        "parquet_file": f"gs://{BUCKET_NAME}/snapshots/{parquet_filename}",
        "notes": notes,
    }

    local_meta = f"/tmp/{meta_filename}"
    with open(local_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Upload both to GCS ---
    gcs_prefix = "snapshots"

    blob_parquet = bucket.blob(f"{gcs_prefix}/{parquet_filename}")
    blob_parquet.upload_from_filename(local_parquet)
    print(f"Uploaded gs://{BUCKET_NAME}/{gcs_prefix}/{parquet_filename}")

    blob_meta = bucket.blob(f"{gcs_prefix}/{meta_filename}")
    blob_meta.upload_from_filename(local_meta)
    print(f"Uploaded gs://{BUCKET_NAME}/{gcs_prefix}/{meta_filename}")

    # --- Summary ---
    print(f"\n--- Snapshot {version_tag} ---")
    print(f"  Rows: {len(df)}")
    print(f"  Polls: {poll_counts}")
    print(f"  Verticals: {vertical_counts}")
    print(f"  GCS path: gs://{BUCKET_NAME}/{gcs_prefix}/{parquet_filename}")

    return df, metadata


def list_snapshots():
    """List all available snapshots in GCS."""
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix="snapshots/")

    meta_files = [b.name for b in blobs if b.name.endswith("_meta.json")]

    print(f"Found {len(meta_files)} snapshots:\n")
    for mf in sorted(meta_files):
        blob = bucket.blob(mf)
        meta = json.loads(blob.download_as_text())
        tag = meta.get("version_tag", "unknown")
        ts = meta.get("snapshot_timestamp", "")[:10]

        if "total_rows" in meta:
            # Video snapshot metadata
            print(f"  [videos]    {tag}  |  {meta['total_rows']} rows  |  {ts}")
            print(f"              Polls: {meta.get('poll_label_counts', {})}")
            print(f"              File:  {meta.get('parquet_file', '')}\n")
        elif "baseline_video_rows" in meta:
            # Baseline metadata
            print(
                f"  [baselines] {tag}  |  {meta['baseline_video_rows']} baseline videos, "
                f"{meta['baseline_median_rows']} medians  |  {ts}"
            )
            print(f"              Files: {meta.get('baselines_file', '')}")
            print(f"                     {meta.get('medians_file', '')}\n")
        else:
            print(f"  [unknown]   {tag}  |  {ts}\n")


def load_videos(version_tag: str):
    """
    Load an existing snapshot from GCS by version tag.
    Returns (DataFrame, metadata dict).

    Usage:
        df, meta = load_snapshot("v1.0_real")
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    # Find the matching parquet file
    blobs = list(bucket.list_blobs(prefix="snapshots/"))
    parquet_blobs = [
        b
        for b in blobs
        if b.name.endswith(".parquet") and f"snapshots_{version_tag}_" in b.name
    ]

    if not parquet_blobs:
        raise FileNotFoundError(
            f"No snapshot found for version tag '{version_tag}'"
        )

    # Take the most recent if multiple exist for this tag
    parquet_blob = sorted(parquet_blobs, key=lambda b: b.name)[-1]
    meta_blob_name = parquet_blob.name.replace(".parquet", "_meta.json")

    # Download parquet to local cache
    local_path = f"/tmp/{parquet_blob.name.split('/')[-1]}"
    parquet_blob.download_to_filename(local_path)
    df = pd.read_parquet(local_path)

    # Download metadata
    meta = json.loads(bucket.blob(meta_blob_name).download_as_text())

    print(
        f"Loaded snapshot '{version_tag}': {len(df)} rows from {meta['snapshot_timestamp'][:10]}"
    )
    print(f"  Polls: {meta['poll_label_counts']}")

    return df, meta


def save_snapshot(df: pd.DataFrame, version_tag: str, notes: str = "", overwrite: bool = False):
    """
    Save an arbitrary DataFrame as a versioned Parquet snapshot to GCS.
    Use this for combined real+synthetic datasets.
    Returns metadata object about the snapshot.

    overwrite=True skips the existence check. Use when re-running after an
    interrupted snapshot where the data landed but downstream steps (e.g. models) did not.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    if not overwrite and _version_tag_exists(bucket, version_tag, "final"):
        raise ValueError(
            f"Final-dataset snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag, pass overwrite=True, or delete the existing snapshot first."
        )

    now = datetime.utcnow()

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    base_name = f"snapshots_{version_tag}_{len(df)}rows_{timestamp}"
    parquet_filename = f"{base_name}.parquet"
    meta_filename = f"{base_name}_meta.json"

    local_parquet = f"/tmp/{parquet_filename}"
    df.to_parquet(local_parquet, index=False)

    poll_counts = (
        df["poll_label"].value_counts().to_dict()
        if "poll_label" in df.columns
        else {}
    )
    vertical_counts = (
        df["vertical"].value_counts().to_dict()
        if "vertical" in df.columns
        else {}
    )
    tier_counts = (
        df["tier"].value_counts().to_dict() if "tier" in df.columns else {}
    )

    synthetic_row_count = (
        int(df["contains_synthetic_data"].sum())
        if "contains_synthetic_data" in df.columns
        else 0
    )
    real_count = len(df) - synthetic_row_count

    metadata = {
        "version_tag": version_tag,
        "snapshot_timestamp": now.isoformat(),
        "total_rows": len(df),
        "real_rows": real_count,
        "synthetic_rows": synthetic_row_count,
        "synthetic_pct": (
            round(synthetic_row_count / len(df) * 100, 1) if len(df) > 0 else 0
        ),
        "unique_videos": df["video_id"].nunique(),
        "unique_channels": df["channel_id"].nunique(),
        "poll_label_counts": poll_counts,
        "vertical_counts": vertical_counts,
        "tier_counts": tier_counts,
        "date_range": {
            "earliest_publish": (
                str(df["published_at"].dropna().min())
                if "published_at" in df.columns
                else ""
            ),
            "latest_publish": (
                str(df["published_at"].dropna().max())
                if "published_at" in df.columns
                else ""
            ),
        },
        "columns": df.columns.tolist(),
        "parquet_file": f"gs://{BUCKET_NAME}/snapshots/{parquet_filename}",
        "notes": notes,
    }

    local_meta = f"/tmp/{meta_filename}"
    with open(local_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    gcs_prefix = "snapshots"
    bucket.blob(f"{gcs_prefix}/{parquet_filename}").upload_from_filename(
        local_parquet
    )
    bucket.blob(f"{gcs_prefix}/{meta_filename}").upload_from_filename(
        local_meta
    )

    print(f"\n--- Snapshot {version_tag} ---")
    print(
        f"  Rows: {len(df)} ({real_count} real, {synthetic_row_count} synthetic)"
    )
    print(f"  Polls: {poll_counts}")
    print(f"  GCS: gs://{BUCKET_NAME}/{gcs_prefix}/{parquet_filename}")

    return metadata


def save_video_snapshot(
    df: pd.DataFrame,
    version_tag: str,
    notes: str = "",
    overwrite: bool = False,
) -> dict:
    """
    Save a pre-loaded df_videos DataFrame to GCS as a versioned parquet snapshot.

    Called by RawSnapshotter after DataLoader has populated run.df_videos.
    Use snapshot_video_data() if you want to pull-and-save in one step.
    Raises if the version tag already exists unless overwrite=True.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    if not overwrite and _version_tag_exists(bucket, version_tag, "raw_video"):
        raise ValueError(
            f"Video snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag or delete the existing snapshot first."
        )

    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    base_name = f"snapshots_{version_tag}_{len(df)}rows_{timestamp}"
    parquet_filename = f"{base_name}.parquet"
    meta_filename = f"{base_name}_meta.json"

    local_parquet = f"/tmp/{parquet_filename}"
    df.to_parquet(local_parquet, index=False)

    poll_counts = df["poll_label"].value_counts().to_dict() if "poll_label" in df.columns else {}
    vertical_counts = df["vertical"].value_counts().to_dict() if "vertical" in df.columns else {}
    tier_counts = df["tier"].value_counts().to_dict() if "tier" in df.columns else {}

    metadata = {
        "version_tag": version_tag,
        "snapshot_timestamp": now.isoformat(),
        "total_rows": len(df),
        "unique_videos": int(df["video_id"].nunique()) if "video_id" in df.columns else None,
        "unique_channels": int(df["channel_id"].nunique()) if "channel_id" in df.columns else None,
        "poll_label_counts": poll_counts,
        "vertical_counts": vertical_counts,
        "tier_counts": tier_counts,
        "date_range": {
            "earliest_publish": (
                str(df["published_at"].dropna().min()) if "published_at" in df.columns else ""
            ),
            "latest_publish": (
                str(df["published_at"].dropna().max()) if "published_at" in df.columns else ""
            ),
        },
        "columns": df.columns.tolist(),
        "parquet_file": f"gs://{BUCKET_NAME}/snapshots/{parquet_filename}",
        "notes": notes,
    }

    local_meta = f"/tmp/{meta_filename}"
    with open(local_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    bucket.blob(f"snapshots/{parquet_filename}").upload_from_filename(local_parquet)
    bucket.blob(f"snapshots/{meta_filename}").upload_from_filename(local_meta)

    print(f"\n--- Video Snapshot {version_tag} ---")
    print(f"  Rows: {len(df)}")
    print(f"  Polls: {poll_counts}")
    print(f"  GCS: gs://{BUCKET_NAME}/snapshots/{parquet_filename}")

    return metadata


def save_baselines_snapshot(
    df_baselines: pd.DataFrame,
    df_medians: pd.DataFrame,
    version_tag: str,
    notes: str = "",
    overwrite: bool = False,
) -> dict:
    """
    Save pre-loaded df_baselines and df_medians to GCS.

    Called by RawSnapshotter after DataLoader has populated run.df_baselines
    and run.df_medians. Use snapshot_baselines() for the pull-and-save variant.
    Raises if the version tag already exists unless overwrite=True.
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    if not overwrite and _version_tag_exists(bucket, version_tag, "baselines"):
        raise ValueError(
            f"Baseline snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag or delete the existing snapshot first."
        )

    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    baselines_name = f"baselines_{version_tag}_{len(df_baselines)}rows_{timestamp}"
    local_baselines = f"/tmp/{baselines_name}.parquet"
    df_baselines.to_parquet(local_baselines, index=False)
    bucket.blob(f"snapshots/{baselines_name}.parquet").upload_from_filename(local_baselines)
    print(f"Uploaded gs://{BUCKET_NAME}/snapshots/{baselines_name}.parquet")

    medians_name = f"medians_{version_tag}_{len(df_medians)}rows_{timestamp}"
    local_medians = f"/tmp/{medians_name}.parquet"
    df_medians.to_parquet(local_medians, index=False)
    bucket.blob(f"snapshots/{medians_name}.parquet").upload_from_filename(local_medians)
    print(f"Uploaded gs://{BUCKET_NAME}/snapshots/{medians_name}.parquet")

    metadata = {
        "version_tag": version_tag,
        "snapshot_timestamp": now.isoformat(),
        "baseline_video_rows": len(df_baselines),
        "baseline_median_rows": len(df_medians),
        "unique_channels": int(df_baselines["channel_id"].nunique()),
        "baselines_file": f"gs://{BUCKET_NAME}/snapshots/{baselines_name}.parquet",
        "medians_file": f"gs://{BUCKET_NAME}/snapshots/{medians_name}.parquet",
        "notes": notes,
    }

    meta_name = f"baselines_{version_tag}_{timestamp}_meta.json"
    local_meta = f"/tmp/{meta_name}"
    with open(local_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    bucket.blob(f"snapshots/{meta_name}").upload_from_filename(local_meta)

    print(f"\n--- Baseline Snapshot {version_tag} ---")
    print(
        f"  Baseline videos: {len(df_baselines)} "
        f"({df_baselines['channel_id'].nunique()} channels)"
    )
    print(f"  Baseline medians: {len(df_medians)}")

    return metadata


def save_splits_snapshot(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    version_tag: str,
    notes: str = "",
    overwrite: bool = False,
) -> dict:
    """
    Save the six per-split modeling artifacts to GCS as separate parquet files.

    Called by FinalSnapshotter after FeatureEngineer (and SyntheticAugmenter).
    X_train / y_train include synthetic rows if SyntheticAugmenter ran.
    Raises if the version tag already exists unless overwrite=True.

    GCS paths follow the pattern:
        snapshots/splits_{version_tag}_{split_name}_{nrows}rows_{timestamp}.parquet
        snapshots/splits_{version_tag}_{timestamp}_meta.json
    """
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    if not overwrite and _version_tag_exists(bucket, version_tag, "splits"):
        raise ValueError(
            f"Splits snapshot '{version_tag}' already exists in GCS. "
            "Use a new version tag or delete the existing snapshot first."
        )

    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    splits = {
        "X_train": X_train,
        "y_train": y_train.to_frame(),
        "X_test": X_test,
        "y_test": y_test.to_frame(),
        "X_val": X_val,
        "y_val": y_val.to_frame(),
    }

    split_meta = {}
    for split_name, df in splits.items():
        filename = f"splits_{version_tag}_{split_name}_{len(df)}rows_{timestamp}.parquet"
        local_path = f"/tmp/{filename}"
        df.to_parquet(local_path, index=False)
        bucket.blob(f"snapshots/{filename}").upload_from_filename(local_path)
        gcs_uri = f"gs://{BUCKET_NAME}/snapshots/{filename}"
        split_meta[split_name] = {"rows": len(df), "file": gcs_uri}
        if df.shape[1] > 1:
            split_meta[split_name]["cols"] = df.shape[1]
        print(f"  Uploaded {gcs_uri}")

    metadata = {
        "version_tag": version_tag,
        "snapshot_timestamp": now.isoformat(),
        "splits": split_meta,
        "notes": notes,
    }

    meta_filename = f"splits_{version_tag}_{timestamp}_meta.json"
    local_meta = f"/tmp/{meta_filename}"
    with open(local_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    bucket.blob(f"snapshots/{meta_filename}").upload_from_filename(local_meta)

    print(f"\n--- Splits Snapshot {version_tag} ---")
    for split_name, info in split_meta.items():
        print(f"  {split_name}: {info['rows']} rows")

    return metadata
