"""
Pipeline Validation — Cloud Run Function

Checks the health of the harvester pipeline across all three poll stages:
upload, 24h, and 7d. Reports completion rates, spot-checks snapshots,
and flags mismatches.

Can also be run locally: python main.py
"""

import os
import json
from google.cloud import bigquery
import functions_framework

PROJECT_ID = os.environ.get("PROJECT_ID", "maduros-dolce")
DATASET_ID = "capstone_youtube"

# Each poll stage: (flag column, poll_label, min hours old to be "eligible")
POLL_STAGES = [
    {
        "name": "upload",
        "flag_col": "upload_poll_done",
        "poll_label": "upload",
        "min_hours": 6,
        "prereq_flag": None,
    },
    {
        "name": "24h",
        "flag_col": "day1_poll_done",
        "poll_label": "24h",
        "min_hours": 30,
        "prereq_flag": "upload_poll_done",
    },
    {
        "name": "7d",
        "flag_col": "day7_poll_done",
        "poll_label": "7d",
        "min_hours": 180,
        "prereq_flag": "day1_poll_done",
    },
]


def run_check(bq_client):
    results = {
        "status": "ok",
        "stages": {},
        "warnings": [],
    }

    for stage in POLL_STAGES:
        name = stage["name"]
        flag_col = stage["flag_col"]
        poll_label = stage["poll_label"]
        min_hours = stage["min_hours"]
        prereq_flag = stage["prereq_flag"]

        # Build prerequisite clause
        if prereq_flag:
            prereq_clause = f"AND {prereq_flag} = TRUE"
        else:
            prereq_clause = ""

        # --- Check 1: Completion rate ---
        completion_query = f"""
            WITH eligible AS (
                SELECT
                    video_id,
                    {flag_col},
                    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), published_at, HOUR) AS hours_old
                FROM `{PROJECT_ID}.{DATASET_ID}.videos_to_track`
                WHERE TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), published_at, HOUR) > {min_hours}
                    {prereq_clause}
            )
            SELECT
                COUNT(*) AS total_eligible,
                COUNTIF({flag_col} = TRUE) AS completed,
                COUNTIF({flag_col} = FALSE) AS missing,
                ROUND(SAFE_DIVIDE(COUNTIF({flag_col} = TRUE), COUNT(*)) * 100, 1) AS completion_pct
            FROM eligible
        """
        row = list(bq_client.query(completion_query).result())[0]

        stage_result = {
            "total_eligible": row.total_eligible,
            "completed": row.completed,
            "missing": row.missing,
            "completion_pct": row.completion_pct or 0.0,
        }

        if row.total_eligible > 0 and (row.completion_pct or 0) < 90:
            results["warnings"].append(
                f"{name}: only {row.completion_pct}% completion "
                f"({row.missing} missing out of {row.total_eligible})"
            )

        # --- Check 2: Spot-check snapshots ---
        spot_query = f"""
            SELECT
                vt.video_id,
                vt.published_at,
                vs.poll_label,
                vs.hours_since_publish,
                vs.view_count,
                vs.like_count,
                vs.comment_count
            FROM `{PROJECT_ID}.{DATASET_ID}.videos_to_track` vt
            LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.video_snapshots` vs
                ON vt.video_id = vs.video_id AND vs.poll_label = '{poll_label}'
            WHERE {flag_col} = TRUE
            ORDER BY vt.published_at DESC
            LIMIT 5
        """
        spot_rows = list(bq_client.query(spot_query).result())
        stage_result["spot_check"] = [
            {
                "video_id": r.video_id,
                "published_at": r.published_at.isoformat() if r.published_at else None,
                "poll_label": r.poll_label,
                "hours_since_publish": r.hours_since_publish,
                "view_count": r.view_count,
                "like_count": r.like_count,
                "comment_count": r.comment_count,
            }
            for r in spot_rows
        ]

        # --- Check 3: Mismatches (flag=TRUE but no snapshot) ---
        mismatch_query = f"""
            SELECT
                vt.video_id,
                vt.published_at,
                vt.channel_id
            FROM `{PROJECT_ID}.{DATASET_ID}.videos_to_track` vt
            LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.video_snapshots` vs
                ON vt.video_id = vs.video_id AND vs.poll_label = '{poll_label}'
            WHERE vt.{flag_col} = TRUE
                AND vs.video_id IS NULL
        """
        mismatch_rows = list(bq_client.query(mismatch_query).result())
        stage_result["mismatches"] = len(mismatch_rows)

        if mismatch_rows:
            results["warnings"].append(
                f"{name}: {len(mismatch_rows)} videos flagged as done "
                f"but missing snapshots"
            )
            stage_result["mismatch_examples"] = [
                {
                    "video_id": r.video_id,
                    "channel_id": r.channel_id,
                    "published_at": r.published_at.isoformat() if r.published_at else None,
                }
                for r in mismatch_rows[:5]
            ]

        # --- Check 4: Orphan check (snapshot exists but no tracking row) ---
        orphan_query = f"""
            SELECT COUNT(*) AS orphan_count
            FROM `{PROJECT_ID}.{DATASET_ID}.video_snapshots` vs
            LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.videos_to_track` vt
                ON vs.video_id = vt.video_id
            WHERE vs.poll_label = '{poll_label}'
                AND vt.video_id IS NULL
        """
        orphan_row = list(bq_client.query(orphan_query).result())[0]
        stage_result["orphaned_snapshots"] = orphan_row.orphan_count

        if orphan_row.orphan_count > 0:
            results["warnings"].append(
                f"{name}: {orphan_row.orphan_count} snapshots with no "
                f"matching tracking row (likely from old harvester)"
            )

        results["stages"][name] = stage_result

    # --- Overall tracking table health ---
    health_query = f"""
        SELECT
            COUNT(*) AS total_tracked,
            COUNTIF(upload_poll_done = TRUE) AS upload_done,
            COUNTIF(day1_poll_done = TRUE) AS day1_done,
            COUNTIF(day7_poll_done = TRUE) AS day7_done,
            MIN(first_seen_at) AS earliest_tracked,
            MAX(first_seen_at) AS latest_tracked
        FROM `{PROJECT_ID}.{DATASET_ID}.videos_to_track`
    """
    health = list(bq_client.query(health_query).result())[0]
    results["tracking_summary"] = {
        "total_tracked": health.total_tracked,
        "upload_done": health.upload_done,
        "day1_done": health.day1_done,
        "day7_done": health.day7_done,
        "earliest_tracked": health.earliest_tracked.isoformat() if health.earliest_tracked else None,
        "latest_tracked": health.latest_tracked.isoformat() if health.latest_tracked else None,
    }

    # --- Snapshot volume by poll_label ---
    volume_query = f"""
        SELECT
            poll_label,
            COUNT(*) AS snapshot_count,
            MIN(poll_timestamp) AS earliest,
            MAX(poll_timestamp) AS latest
        FROM `{PROJECT_ID}.{DATASET_ID}.video_snapshots`
        WHERE poll_label IS NOT NULL
        GROUP BY poll_label
        ORDER BY poll_label
    """
    volume_rows = list(bq_client.query(volume_query).result())
    results["snapshot_volumes"] = {
        r.poll_label: {
            "count": r.snapshot_count,
            "earliest": r.earliest.isoformat() if r.earliest else None,
            "latest": r.latest.isoformat() if r.latest else None,
        }
        for r in volume_rows
    }

    if results["warnings"]:
        results["status"] = "warnings"

    return results


def format_report(results):
    """Format results as a readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PIPELINE VALIDATION REPORT")
    lines.append("=" * 60)

    status = results["status"]
    lines.append(f"\nOverall status: {'✅ OK' if status == 'ok' else '⚠️  WARNINGS'}\n")

    # Tracking summary
    ts = results["tracking_summary"]
    lines.append("--- Tracking Table ---")
    lines.append(f"  Total videos tracked:  {ts['total_tracked']}")
    lines.append(f"  Upload polls done:     {ts['upload_done']}")
    lines.append(f"  24h polls done:        {ts['day1_done']}")
    lines.append(f"  7d polls done:         {ts['day7_done']}")
    lines.append(f"  Tracking since:        {ts['earliest_tracked']}")
    lines.append(f"  Latest tracked:        {ts['latest_tracked']}")

    # Snapshot volumes
    lines.append("\n--- Snapshot Volumes ---")
    for label, vol in results.get("snapshot_volumes", {}).items():
        lines.append(f"  {label}: {vol['count']} snapshots "
                     f"({vol['earliest']} → {vol['latest']})")

    # Per-stage details
    for stage_name, stage in results["stages"].items():
        lines.append(f"\n--- Stage: {stage_name} ---")
        lines.append(f"  Eligible:    {stage['total_eligible']}")
        lines.append(f"  Completed:   {stage['completed']}")
        lines.append(f"  Missing:     {stage['missing']}")
        lines.append(f"  Completion:  {stage['completion_pct']}%")
        lines.append(f"  Mismatches:  {stage['mismatches']}")
        lines.append(f"  Orphans:     {stage['orphaned_snapshots']}")

        if stage.get("spot_check"):
            lines.append(f"  Spot-check (latest {len(stage['spot_check'])} videos):")
            for s in stage["spot_check"]:
                lines.append(
                    f"    {s['video_id']} | "
                    f"{s['hours_since_publish']}h | "
                    f"views={s['view_count']} likes={s['like_count']}"
                )

        if stage.get("mismatch_examples"):
            lines.append(f"  Mismatch examples:")
            for m in stage["mismatch_examples"]:
                lines.append(f"    {m['video_id']} | {m['channel_id']}")

    # Warnings
    if results["warnings"]:
        lines.append("\n--- Warnings ---")
        for w in results["warnings"]:
            lines.append(f"  ⚠️  {w}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# --- Cloud Run Entry Point ---

@functions_framework.http
def run_validation(request):
    try:
        bq_client = bigquery.Client(project=PROJECT_ID)
        results = run_check(bq_client)

        output_format = request.args.get("format", "text")

        if output_format == "json":
            return json.dumps(results, indent=2, default=str), 200, {
                "Content-Type": "application/json"
            }
        else:
            report = format_report(results)
            print(report)  # Also log it
            return report, 200, {"Content-Type": "text/plain"}

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return f"Validation error: {e}", 500


# --- Local execution ---

if __name__ == "__main__":
    bq_client = bigquery.Client(project=PROJECT_ID)
    results = run_check(bq_client)
    print(format_report(results))

