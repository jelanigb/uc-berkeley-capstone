import json
from datetime import datetime
from google.cloud import bigquery, storage
import pandas as pd

PROJECT_ID = "maduros-dolce"
DATASET_ID = "capstone_youtube"
BUCKET_NAME = "maduros-dolce-capstone-data"

def snapshot_training_data(version_tag: str, notes: str = ""):
    """
    Pull current video_snapshots from BQ, save as Parquet,
    upload to GCS with a metadata sidecar JSON.
    
    Usage:
        snapshot_training_data("v1.0_real", notes="First real-only dataset")
        snapshot_training_data("v1.2_mixed30", notes="30% real, 70% synthetic")
    """
    bq_client = bigquery.Client(project=PROJECT_ID)
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    now = datetime.utcnow()

    # --- Pull data ---
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.video_snapshots`
        WHERE poll_label IS NOT NULL
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
    poll_counts = df['poll_label'].value_counts().to_dict()
    vertical_counts = df['vertical'].value_counts().to_dict()
    tier_counts = df['tier'].value_counts().to_dict()
    
    metadata = {
        "version_tag": version_tag,
        "snapshot_timestamp": now.isoformat(),
        "total_rows": len(df),
        "unique_videos": df['video_id'].nunique(),
        "unique_channels": df['channel_id'].nunique(),
        "poll_label_counts": poll_counts,
        "vertical_counts": vertical_counts,
        "tier_counts": tier_counts,
        "date_range": {
            "earliest_publish": str(df['published_at'].min()),
            "latest_publish": str(df['published_at'].max()),
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
